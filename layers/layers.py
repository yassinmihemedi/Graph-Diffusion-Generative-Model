import math
import numpy as np
import torch_geometric as pyg
import torch
try:
    from torch_scatter import scatter as _scatter
except (ImportError, OSError):
    from torch_geometric.utils import scatter as _scatter
from diffusion.diffusion_base import index_to_log_onehot
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree

norm_dict = {
    'Batch': lambda d: torch.nn.BatchNorm1d(d),
    'None': lambda d: torch.nn.Identity(),
    "Inst": lambda d: pyg.nn.norm.InstanceNorm(d),
    "Graph": lambda d: pyg.nn.norm.GraphNorm(d),
}

class SelEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out


class TimeEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out
        

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NodeModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, seq_lens, n_layers=6):
        super().__init__()
        self.num_bits = num_bits
        self.max_num_nodes = max_num_nodes
        self.seq_lens = seq_lens
        self.n_layers = n_layers
        self.embedding = nn.Linear(num_bits, 64)
        self.pos_embedding = SinusoidalPosEmb(64, seq_lens)
        self.g_embedding = nn.Linear(1, 256)
        self.res_embedding = nn.Linear(1, 64)
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x, g_v, res_count):
        x = self.embedding(x)
        g = g_v / self.max_num_nodes
        r = res_count / self.max_num_nodes

        g = self.g_embedding(g)
        r = self.res_embedding(r[..., None])
        t = torch.arange(0,x.shape[1])[None,:].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0],-1, 64)
        x = torch.cat([x, t, r], dim=-1)
        g = g[None,:,:].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (g, g))
        x = self.linear(self.dropout(x))
        return x

class BitModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(3, 64)
        self.max_num_nodes = max_num_nodes
        self.n_layers = n_layers
        self.num_bits = num_bits
        self.pos_embedding = SinusoidalPosEmb(64, num_bits)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.res_embedding = nn.Linear(1, 128) 

    def forward(self, bits, hidden_nodes, res_count):
        x = self.embedding(bits)
        r = res_count/self.max_num_nodes
        r = self.res_embedding(r)
        t = torch.arange(0,x.shape[1])[None, :].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0], -1, 64)
        x = torch.cat([x, t], dim=-1)
        hidden_nodes = torch.cat([hidden_nodes, r],dim=-1)
        hidden_nodes = hidden_nodes[None,...].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (hidden_nodes, hidden_nodes))
        x = self.linear(self.dropout(x))
        return x

class MiniAttentionLayer(torch.nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, d_model, num_heads=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model*num_heads, num_heads, batch_first=True)
        self.qkv_node = torch.nn.Linear(node_dim, d_model * 3 * num_heads)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, d_model * 3 * num_heads)
        self.edge_linear = torch.nn.Sequential(torch.nn.Linear(d_model * num_heads, d_model),
                                                torch.nn.SiLU(),
                                                torch.nn.Linear(d_model, out_edge_dim))
    def forward(self, node_us, node_vs, edges):

        # node_us/vs: (B, D)
        q_node_us, k_node_us, v_node_us = self.qkv_node(node_us).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_node_vs, k_node_vs, v_node_vs = self.qkv_node(node_vs).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_edges, k_edges, v_edges = self.qkv_edge(edges).chunk(3, -1) # (B, D*num_heads) for q/k/v

        q = torch.stack([q_node_us, q_node_vs, q_edges], 1) # (B, 3, D*num_heads)
        k = torch.stack([k_node_us, k_node_vs, k_edges], 1) # (B, 3, D*num_heads)
        v = torch.stack([v_node_us, v_node_vs, v_edges], 1) # (B, 3, D*num_heads)

        h, _ = self.multihead_attn(q, k, v)
        h_edge = h[:, -1, :]
        h_edge = self.edge_linear(h_edge)

        return h_edge


# ─────────────────────────────────────────────────────────────────────────────
# FiLM + MiniAt_Film
# Reference: revise_generation_citation_GAN_degree_bit.ipynb
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(torch.nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Adaptively scales and shifts feature tensor *x* conditioned on *cond*:
        output = γ(cond) * x + β(cond)

    Used in MiniAt_Film to let the raw edge context directly modulate the
    attention output, providing a shortcut pathway for edge-level conditioning.
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = torch.nn.Linear(cond_dim, feat_dim)
        self.beta  = torch.nn.Linear(cond_dim, feat_dim)

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cond : (*, cond_dim)  conditioning signal (e.g. raw edge context)
        x    : (*, feat_dim)  features to modulate
        """
        return self.gamma(cond) * x + self.beta(cond)


class MiniAt_Film(torch.nn.Module):
    """MiniAttentionLayer extended with FiLM conditioning.

    Jointly attends over the (node_u, node_v, edge_context) triple —
    identical to MiniAttentionLayer — then applies FiLM conditioning using
    the original edge features as the modulation signal:

        h_attn   = MultiheadAttention(node_u, node_v, edge)[edge_token]
        h_linear = edge_linear(h_attn)                     # (E, out_edge_dim)
        output   = γ(edge) * h_linear + β(edge)            # FiLM

    The FiLM pathway lets the model adaptively re-scale predictions based
    on raw edge context without relying solely on the attention bottleneck,
    improving expressivity especially at early diffusion timesteps when
    edge signals are sparse.

    Reference: ``MiniAt_Film`` class in
    ``revise_generation_citation_GAN_degree_bit.ipynb``.
    """
    def __init__(self, node_dim: int, in_edge_dim: int, out_edge_dim: int,
                 d_model: int, num_heads: int = 2):
        super().__init__()
        embed_dim = d_model * num_heads
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.qkv_node = torch.nn.Linear(node_dim,    embed_dim * 3)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, embed_dim * 3)
        self.edge_linear = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, out_edge_dim),
        )
        # FiLM: condition on the raw concatenated edge features
        self.film = FiLM(cond_dim=in_edge_dim, feat_dim=out_edge_dim)

    def forward(self, node_us: torch.Tensor, node_vs: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_us : (E, node_dim)    source node features
        node_vs : (E, node_dim)    target node features
        edges   : (E, in_edge_dim) edge context (typically cat[node_u, node_v])

        Returns
        -------
        (E, out_edge_dim)
        """
        q_u, k_u, v_u = self.qkv_node(node_us).chunk(3, -1)
        q_v, k_v, v_v = self.qkv_node(node_vs).chunk(3, -1)
        q_e, k_e, v_e = self.qkv_edge(edges).chunk(3, -1)

        q = torch.stack([q_u, q_v, q_e], 1)   # (E, 3, embed_dim)
        k = torch.stack([k_u, k_v, k_e], 1)
        v = torch.stack([v_u, v_v, v_e], 1)

        h, _ = self.multihead_attn(q, k, v)    # (E, 3, embed_dim)
        h_edge = self.edge_linear(h[:, -1, :]) # (E, out_edge_dim)

        # FiLM conditioning: γ(edges) * h_edge + β(edges)
        return self.film(edges, h_edge)

class TGNN(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))
        self.node_interaction = MiniAttentionLayer(node_dim=dim, in_edge_dim=dim, out_edge_dim=dim, d_model=dim, num_heads=2)
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]
        
        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        nodes = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()

        nodes = nodes[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes = self.embedding_t(nodes)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts


        # SparseDiff subsampling: see GIN.forward's comment — score only the
        # attached query subset when present (training only).
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = query_edge_index if query_edge_index is not None else pyg_data.full_edge_index
        row, col = edge_query_index[0], edge_query_index[1]
        edge_emb = torch.cat([nodes[row], nodes[col]], -1)
        edge_class = self.final_out(edge_emb)

        return pyg_data.log_node_attr, edge_class

class AttentionGINConv(torch.nn.Module):
    """Attention-gated sum-aggregation operator (GIN + GAT hybrid).

    h_i = MLP( (1+ε)·h_i  +  Σ_j  softmax_j(α_{ij}) · h_j )

    * SUM aggregation keeps GIN's injective (WL-test) expressiveness.
    * Softmax normalization per destination node bounds the total incoming
      signal regardless of node degree, preventing hub-node explosion.
    * Multi-head attention (att_heads) captures diverse relational patterns.
    """
    def __init__(self, mlp, in_dim, att_heads=4):
        super().__init__()
        self.mlp = mlp
        # Project (src‖dst) features to att_heads logits, then combine to one weight
        self.att_fc  = torch.nn.Linear(in_dim * 2, att_heads, bias=False)
        self.att_out = torch.nn.Linear(att_heads, 1, bias=False)
        self.eps = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:          # no edges → self-loop only
            return self.mlp((1.0 + self.eps) * x)
        row, col = edge_index
        # Multi-head attention logits (E, att_heads)
        alpha = F.leaky_relu(
            self.att_fc(torch.cat([x[row], x[col]], dim=-1)), negative_slope=0.2
        )
        # Combine heads → scalar attention weight per edge (E,)
        alpha = self.att_out(alpha).squeeze(-1)
        # Softmax per destination: for each dst node the weights over its sources sum to 1
        alpha = pyg.utils.softmax(alpha, col, num_nodes=x.size(0))
        # Attention-weighted SUM (GIN style)
        agg = _scatter(
            alpha.unsqueeze(-1) * x[row], col, dim=0, dim_size=x.size(0), reduce='sum'
        )
        return self.mlp((1.0 + self.eps) * x + agg)


class GIN(torch.nn.Module):
    """GIN with attention-gated sum aggregation.

    Replaces TransformerConv with AttentionGINConv: attention-normalised sum
    aggregation preserves GIN's WL expressivity while being as stable as mean
    aggregation for high-degree hub nodes.  All other architectural choices
    (GRU update, global context, sinusoidal time embedding) match TGNN.
    """
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps,
                 num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False,
                 augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(), torch.nn.Linear(dim * 4, dim)
        )
        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(), torch.nn.Linear(dim * 4, dim)
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(), torch.nn.Linear(dim * 4, dim)
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        if kwargs.get('gru', False):
            self.gru = torch.nn.GRU(dim, dim)

        for i in range(len(num_heads)):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            gin_nn = torch.nn.Sequential(
                torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(),
                torch.nn.Linear(dim * 4, dim)
            )
            # AttentionGINConv: sum aggregation + attention normalisation
            self.layers[f'conv{i}'] = AttentionGINConv(gin_nn, in_dim=dim * 2, att_heads=4)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}']  = torch.nn.SiLU()

        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 2), torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),     torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )

    def forward(self, pyg_data, t_node, t_edge):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        nodes = pyg.utils.degree(edge_index[0], num_nodes=pyg_data.num_nodes).clamp(
            max=self.max_degree + 1).long()

        nodes = nodes[..., None] / self.max_degree
        nodes = self.embedding_t(nodes)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)

        h = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)

        for i in range(len(self.num_heads)):
            t_emb = self.layers[f'time{i}'](t)
            nodes = torch.cat([nodes, t_emb], dim=-1)          # (N, dim*2)
            nodes = self.layers[f'conv{i}'](nodes, edge_index)  # (N, dim) – attention GIN
            nodes = torch.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)

            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(
                contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)
            nodes = nodes + contexts

        # SparseDiff subsampling: if a reduced candidate set was attached for
        # this step (training only — see BinomialDiffusionVanilla._p_pred),
        # score only that subset instead of all N(N-1)/2 pairs. Message passing
        # above is unaffected — it already used the full current graph state.
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = query_edge_index if query_edge_index is not None else pyg_data.full_edge_index
        row, col = edge_query_index[0], edge_query_index[1]
        edge_emb = torch.cat([nodes[row], nodes[col]], -1)
        edge_class = self.final_out(edge_emb)
        return pyg_data.log_node_attr, edge_class


class GIN_Film(torch.nn.Module):
    """GIN with MiniAt_Film edge prediction (--arch GIN_Film).

    Identical to GIN in all message-passing stages (AttentionGINConv,
    GRU update, global context, sinusoidal time embedding) but replaces
    the final 3-layer MLP edge decoder with ``MiniAt_Film``:

        edge_class = MiniAt_Film(node_u, node_v, cat[node_u, node_v])

    MiniAt_Film jointly attends over the (node_u, node_v, edge_context)
    triple and then applies FiLM conditioning:

        h_attn  = MultiheadAttention(node_u, node_v, edge)[edge_token]
        output  = γ(edge_ctx) · edge_linear(h_attn) + β(edge_ctx)

    This gives the denoiser a direct adaptive pathway to condition edge
    predictions on the concatenated node-pair context without relying
    solely on the attention bottleneck, significantly improving
    expressivity at early timesteps when the diffusion signal is sparse.

    Reference: ``GDisc`` / ``Graphdenoising`` in
    ``revise_generation_citation_GAN_degree_bit.ipynb``.
    """
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps,
                 num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False,
                 augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree   = max_degree
        self.num_classes  = num_edge_classes
        self.num_heads    = num_heads
        self.dim          = dim
        self.num_steps    = num_steps
        self.embedding_t  = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers       = torch.nn.ModuleDict()
        self.norm         = norm
        self.gru          = torch.nn.Identity()

        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        if kwargs.get('gru', False):
            self.gru = torch.nn.GRU(dim, dim)

        for i in range(len(num_heads)):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            gin_nn = torch.nn.Sequential(
                torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(),
                torch.nn.Linear(dim * 4, dim)
            )
            # AttentionGINConv: sum aggregation + attention normalisation (WL-expressive)
            self.layers[f'conv{i}'] = AttentionGINConv(gin_nn, in_dim=dim * 2, att_heads=4)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}']  = torch.nn.SiLU()

        # ── MiniAt_Film replaces the plain MLP decoder ──────────────────────
        # Attends over (node_u, node_v, edge_context) and applies FiLM
        self.edge_attention = MiniAt_Film(
            node_dim=dim,                 # node features after GRU (dim)
            in_edge_dim=dim * 2,          # edge context = cat(node_u, node_v)
            out_edge_dim=num_edge_classes,
            d_model=dim,
            num_heads=4,
        )

    def forward(self, pyg_data, t_node, t_edge):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)

        nodes = pyg.utils.degree(edge_index[0], num_nodes=pyg_data.num_nodes).clamp(
            max=self.max_degree + 1).long()
        nodes = nodes[..., None] / self.max_degree
        nodes = self.embedding_t(nodes)

        t = self.time_pos_emb(t_node)
        t = self.mlp(t)

        h = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)

        for i in range(len(self.num_heads)):
            t_emb = self.layers[f'time{i}'](t)
            nodes = torch.cat([nodes, t_emb], dim=-1)           # (N, dim*2)
            nodes = self.layers[f'conv{i}'](nodes, edge_index)  # AttentionGINConv
            nodes = torch.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h      = self.dropout(h)
            nodes  = nodes.squeeze(0)

            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(
                contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)
            nodes = nodes + contexts

        # ── MiniAt_Film edge prediction ──────────────────────────────────────
        # SparseDiff subsampling: see GIN.forward's comment — score only the
        # attached query subset when present (training only).
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = query_edge_index if query_edge_index is not None else pyg_data.full_edge_index
        row, col = edge_query_index[0], edge_query_index[1]
        edge_emb  = torch.cat([nodes[row], nodes[col]], -1)          # (E, dim*2)
        edge_class = _chunked_edge_attention(self.edge_attention, nodes[row], nodes[col], edge_emb)  # (E, C)
        return pyg_data.log_node_attr, edge_class


class GCN(torch.nn.Module):
    """GCN variant of TGNN: same depth, activations, and global context; GCNConv replaces TransformerConv."""
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i in range(len(num_heads)):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.GCNConv(in_channels=dim * 2, out_channels=dim)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))
        self.node_interaction = MiniAttentionLayer(node_dim=dim, in_edge_dim=dim, out_edge_dim=dim, d_model=dim, num_heads=2)
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )

    def forward(self, pyg_data, t_node, t_edge):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        nodes = pyg.utils.degree(edge_index[0], num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree + 1).long()

        nodes = nodes[..., None] / self.max_degree
        nodes = self.embedding_t(nodes)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)

        h = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)

        for i in range(len(self.num_heads)):
            t_emb = self.layers[f'time{i}'](t)
            nodes = torch.cat([nodes, t_emb], dim=-1)
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = torch.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)

            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)
            nodes = nodes + contexts

        # SparseDiff subsampling: see GIN.forward's comment — score only the
        # attached query subset when present (training only).
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = query_edge_index if query_edge_index is not None else pyg_data.full_edge_index
        row, col = edge_query_index[0], edge_query_index[1]
        edge_emb = torch.cat([nodes[row], nodes[col]], -1)
        edge_class = self.final_out(edge_emb)
        return pyg_data.log_node_attr, edge_class


# ─────────────────────────────────────────────────────────────────────────────
# GraphDenoising  –  scalable denoising backbone
# Reference: Graphdenoising class in revise_generation_citation_GAN_degree_bit.ipynb
# ─────────────────────────────────────────────────────────────────────────────

class SoftWeightedGINConv(torch.nn.Module):
    """GIN aggregation with externally-supplied soft edge weights.

    Aggregates neighbour messages weighted by the provided *edge_weight*
    tensor (typically P(edge=1) from the diffusion noisy state), then
    applies a learnable MLP identical to standard GIN:

        agg_i = Σ_{j ∈ N(i)}  w_{ij} · h_j
        h_i'  = MLP( (1 + ε) · h_i + agg_i )

    Unlike AttentionGINConv, the weights are NOT learned — they come
    directly from the diffusion process, giving the denoiser a privileged
    read of the current edge-probability landscape without having to
    infer it through attention.
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        if edge_index.size(1) == 0:
            return self.mlp((1.0 + self.eps) * x)
        row, col = edge_index
        msgs = (edge_weight.unsqueeze(-1) * x[row]
                if edge_weight is not None else x[row])
        agg = _scatter(msgs, col, dim=0, dim_size=x.size(0), reduce='sum')
        return self.mlp((1.0 + self.eps) * x + agg)


def _compute_rwse(edge_index, edge_weights, num_nodes, K):
    """Random Walk Structural Encoding from the current (noisy) graph state.

    rwse[i, k] = diag(T^{k+1})[i]  where  T = row-normalised weighted adjacency.
    Dynamic: computed from edge_index at each forward pass, so it adapts to the
    current noise level t.  No eigendecomposition — uses K sparse→dense matmuls.
    Cost: O(K · |E_t| · N).  Returns (N, K) float32, same device as inputs.
    """
    device = edge_index.device
    if edge_index.shape[1] == 0:
        return torch.zeros(num_nodes, K, device=device)
    deg    = torch.zeros(num_nodes, device=device).scatter_add_(
                 0, edge_index[0], edge_weights)
    norm_w = edge_weights / deg[edge_index[0]].clamp(min=1e-8)
    T_sp   = torch.sparse_coo_tensor(
                 edge_index, norm_w, (num_nodes, num_nodes)).coalesce()
    P      = torch.eye(num_nodes, device=device)
    rwse   = torch.zeros(num_nodes, K, device=device)
    for k in range(K):
        P          = torch.sparse.mm(T_sp, P)
        rwse[:, k] = P.diagonal()
    return rwse


class GraphDenoising(torch.nn.Module):
    """Scalable graph denoising backbone for binomial diffusion generation.

    Two key improvements over GIN / TGNN:

    1.  **Soft-edge-weighted message passing** — neighbour aggregation is
        weighted by P(edge_ij = 1) read from log_full_edge_attr_t. The model
        therefore sees a continuous, probabilistic view of the noisy graph
        rather than a binary hard-thresholded one, yielding richer gradient
        signal and faster convergence.

    2.  **Fully sparse message-passing graph** — only edges whose argmax
        label is 1 ("currently on") participate in MP. The effective number
        of messages is O(rho · N²) instead of O(N²); for sparse graphs
        (rho ≪ 1) this is a large constant-factor speedup during both
        training and sampling.

    The edge decoder is identical to GIN_Film: MiniAt_Film jointly attends
    over (node_u, node_v, cat[node_u, node_v]) then applies FiLM conditioning,
    producing expressive per-edge logits over the full upper-triangular set.

    This model is intended to REPLACE GIN / TGNN for scalable generation,
    keeping the same forward signature and diffusion interface.
    """
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim,
                 num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None',
                 degree=False, augmented_features={},
                 rwse_k=0, pred_edge_classes=None, **kwargs):
        super().__init__()
        num_layers         = len(num_heads)
        self.max_degree    = max_degree
        # pred_edge_classes: number of classes the decoder head produces.
        # Defaults to num_edge_classes (2-class binary case).
        # For PA-DACA: num_edge_classes=3 (state space) but pred=2 (x_0 binary).
        out_classes        = pred_edge_classes if pred_edge_classes is not None else num_edge_classes
        self.num_classes   = out_classes
        self.num_steps     = num_steps
        self.dim           = dim
        self.rwse_k        = rwse_k

        self.node_embed    = torch.nn.Linear(1, dim)

        # RWSE projection: 2-layer MLP maps K walk-length features → dim.
        # Added to the degree embedding before the first GIN layer (same style
        # as DiGress / DIRECTO).  Disabled when rwse_k=0 (default).
        if rwse_k > 0:
            self.rwse_proj = torch.nn.Sequential(
                torch.nn.Linear(rwse_k, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim))
        self.time_pos_emb  = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.time_mlp      = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim))
        self.global_mlp    = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim))
        self.context_mlp   = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim))
        self.dropout = torch.nn.Dropout(p=dropout)

        self.convs     = torch.nn.ModuleList()
        self.norms     = torch.nn.ModuleList()
        self.time_embs = torch.nn.ModuleList()
        for _ in range(num_layers):
            gin_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim * 2, dim * 4), torch.nn.SiLU(),
                torch.nn.Linear(dim * 4, dim))
            self.convs.append(SoftWeightedGINConv(gin_mlp))
            self.norms.append(norm_dict[norm](dim))
            self.time_embs.append(TimeEmb(dim, dim, Mish()))

        self.gru = (torch.nn.GRU(dim, dim)
                    if kwargs.get('gru', False) else torch.nn.Identity())

        # MiniAt_Film decoder (same as GIN_Film)
        self.edge_attention = MiniAt_Film(
            node_dim=dim,
            in_edge_dim=dim * 2,
            out_edge_dim=out_classes,
            d_model=dim,
            num_heads=4)

    def forward(self, pyg_data, t_node, t_edge):
        # ── Soft edge probabilities from current noisy state ────────────────
        # P(edge=1) for every candidate edge pair, shape (E_full,)
        edge_probs = torch.softmax(pyg_data.log_full_edge_attr_t, dim=-1)[:, 1]

        # ── Sparse bidirectional edge_index for message passing ──────────────
        # Only edges currently "on" (argmax==1) are used, so MP cost is O(rho·N²).
        # Use == 1 (not .bool()) to correctly exclude state-2 (M, mask) in PA-DACA.
        is_edge    = (pyg_data.log_full_edge_attr_t.argmax(-1) == 1)
        sparse_idx = is_edge.nonzero(as_tuple=True)[0]
        sparse_ei  = pyg_data.full_edge_index[:, sparse_idx]     # (2, E_sp)
        sparse_ep  = edge_probs[sparse_idx]                       # (E_sp,)  soft weights
        edge_index = torch.cat([sparse_ei, sparse_ei.flip(0)], dim=-1)  # bidirectional
        ep_bidi    = sparse_ep.repeat(2)

        # ── Node features: target_degree (PA-DACA) or noisy-graph degree ─────
        # Use target_degree* when available: permanent ground-truth conditioning
        # that is never masked, providing a structural scaffold at all noise levels.
        if hasattr(pyg_data, 'target_degree') and pyg_data.target_degree is not None:
            nodes = pyg_data.target_degree.float().clamp(max=self.max_degree + 1)
        else:
            nodes = pyg.utils.degree(
                edge_index[0], num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree + 1)
        nodes = self.node_embed(nodes.unsqueeze(-1) / self.max_degree)    # (N, dim)

        # ── RWSE: dynamic random-walk features from the current noisy graph ────
        # Computed from edge_index (already built above from x_t), so it adapts
        # to the noise level: at t≈T the model sees RWSE of a near-random graph;
        # at t≈0 it sees the true graph's local clustering and community density.
        # Cost: K sparse→dense matmuls, O(K·|E_t|·N), <5 ms on GPU.
        if self.rwse_k > 0:
            with torch.no_grad():
                rwse = _compute_rwse(edge_index, ep_bidi, pyg_data.num_nodes, self.rwse_k)
            nodes = nodes + self.rwse_proj(rwse)

        t        = self.time_pos_emb(t_node)
        t        = self.time_mlp(t)
        h        = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)

        for conv, norm_l, time_emb in zip(self.convs, self.norms, self.time_embs):
            t_emb  = time_emb(t)
            x_in   = torch.cat([nodes, t_emb], dim=-1)                    # (N, dim*2)
            nodes  = conv(x_in, edge_index, ep_bidi)                       # (N, dim)
            nodes  = torch.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
            nodes  = norm_l(nodes)
            nodes  = F.silu(nodes)
            nodes  = self.dropout(nodes)

            if not isinstance(self.gru, torch.nn.Identity):
                nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
                h     = self.dropout(h)
                nodes = nodes.squeeze(0)

            ctx      = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(contexts + ctx, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)
            nodes    = nodes + contexts

        # ── Edge prediction via MiniAt_Film over full upper-triangular set ───
        # SparseDiff subsampling: see GIN.forward's comment — score only the
        # attached query subset when present (training only).
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = query_edge_index if query_edge_index is not None else pyg_data.full_edge_index
        row, col   = edge_query_index
        edge_emb   = torch.cat([nodes[row], nodes[col]], dim=-1)            # (E_query, dim*2)
        edge_class = _chunked_edge_attention(self.edge_attention, nodes[row], nodes[col], edge_emb)  # (E_query, C)
        return pyg_data.log_node_attr, edge_class


# ─────────────────────────────────────────────────────────────────────────────
# GraphDenoisingERGM  –  GraphDenoising + ERGM-3 triangle feature τ̂_ij(t)
# Used with ERGMAnchoredDiffusion (diffusion/diffusion_ergm.py).
# ─────────────────────────────────────────────────────────────────────────────

class GraphDenoisingERGM(GraphDenoising):
    """GraphDenoising extended with the ERGM-3 triangle feature τ̂_ij(t).

    The only structural difference from GraphDenoising is the edge decoder:
      GraphDenoising:     edge_emb = cat[node_u, node_v]          → dim*2
      GraphDenoisingERGM: edge_emb = cat[node_u, node_v, tau_emb] → dim*3

    τ̂_ij(t) = # common revealed neighbors at noise level t.
    It is log-scaled (log(1+τ̂)), projected via a 2-layer MLP to dim,
    and concatenated before MiniAt_Film.  This gives the denoiser a direct
    read of the local triangle count, enabling ERGM-3 recovery at the
    denoising optimum (Theorem 1 in ergm_ddpm.tex).

    tau_feat is passed via pyg_data.tau_feat (shape = |Q_t|) by
    ERGMAnchoredDiffusion before each denoiser call; falls back to zeros
    if absent (e.g. when used standalone or in evaluation without tau).
    """

    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim,
                 num_steps, num_heads=None, dropout=0., norm='None',
                 degree=False, augmented_features=None,
                 rwse_k=0, pred_edge_classes=None,
                 legacy_ergm_embed=False, **kwargs):
        if num_heads is None:
            num_heads = [4, 4, 4, 1]
        if augmented_features is None:
            augmented_features = {}
        super().__init__(
            max_degree, num_node_classes, num_edge_classes, dim, num_steps,
            num_heads, dropout, norm, degree, augmented_features,
            rwse_k, pred_edge_classes, **kwargs)

        out_classes = pred_edge_classes if pred_edge_classes is not None else num_edge_classes

        # legacy_ergm_embed=True reconstructs the pre-node_embed_t architecture
        # so that older checkpoints load without any weight loss — see
        # eval_utils/evaluate.py:_detect_legacy_ergm_embed. Do NOT randomly
        # reinitialize these modules the way deg_head/tri_head are handled:
        # unlike those (dead at eval time), node_embed/tau_proj feed every
        # forward pass, so a shape/weight mismatch here silently corrupts
        # sampling instead of raising.
        self.legacy_ergm_embed = legacy_ergm_embed

        if legacy_ergm_embed:
            # Old architecture: single-Linear node_embed, no degree-deficit
            # node_embed_t signal, 2-layer tau_proj.
            self.node_embed = torch.nn.Linear(1, dim)
            self.node_embed_t = None
            self.tau_proj = torch.nn.Sequential(
                torch.nn.Linear(1, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim))
        else:
            # 3-layer MLP for permanent target degree d* (replaces parent's 1-layer Linear).
            # Log-scale input: log(1+d*)/log(1+max_degree) ∈ [0,1] — better for power-law.
            self.node_embed = torch.nn.Sequential(
                torch.nn.Linear(1, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim))

            # 3-layer MLP for dynamic noisy-graph degree d_t (degree deficit signal).
            # d* − d_t encodes how many edges remain to be formed for each node.
            self.node_embed_t = torch.nn.Sequential(
                torch.nn.Linear(1, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim))

            # τ̂_ij projection: log(1+τ̂) scalar → dim — 3-layer MLP
            self.tau_proj = torch.nn.Sequential(
                torch.nn.Linear(1, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim), torch.nn.SiLU(),
                torch.nn.Linear(dim, dim))

        # Replace the dim*2 edge decoder with one that accepts dim*3
        self.edge_attention = MiniAt_Film(
            node_dim=dim,
            in_edge_dim=dim * 3,   # cat[node_u(dim), node_v(dim), tau_emb(dim)]
            out_edge_dim=out_classes,
            d_model=dim,
            num_heads=4)

        # Normalisation constant for log-degree embedding (computed once).
        self._log_max_deg = math.log1p(float(max_degree))

        # Node-level auxiliary regression heads.
        # deg_head: linear(h_i) → log(1+d*_i)/log(1+d_max) ∈ [0,1].
        # tri_head: linear(h_i) → log(1+τ_node_i)/log(1+d_max²) ∈ [0,1].
        # One Linear layer each — small enough not to compete with the edge
        # decoder for model capacity, but sufficient to create a direct
        # gradient path through all GNN layers for structural encoding.
        self.deg_head = torch.nn.Linear(dim, 1)
        self.tri_head = torch.nn.Linear(dim, 1)

    def forward(self, pyg_data, t_node, t_edge):
        """Identical to GraphDenoising.forward() except for τ̂_ij in edge emb."""
        # ── Soft edge probs + sparse bidirectional MP graph ──────────────────
        edge_probs = torch.softmax(pyg_data.log_full_edge_attr_t, dim=-1)[:, 1]
        is_edge    = (pyg_data.log_full_edge_attr_t.argmax(-1) == 1)
        sparse_idx = is_edge.nonzero(as_tuple=True)[0]
        sparse_ei  = pyg_data.full_edge_index[:, sparse_idx]
        sparse_ep  = edge_probs[sparse_idx]
        edge_index = torch.cat([sparse_ei, sparse_ei.flip(0)], dim=-1)
        ep_bidi    = sparse_ep.repeat(2)

        # ── Node features: d* (permanent) + d_t (dynamic degree deficit) ───────
        # d* — true target degree, never masked, provides permanent scaffold.
        if hasattr(pyg_data, 'target_degree') and pyg_data.target_degree is not None:
            d_star = pyg_data.target_degree.float().clamp(max=self.max_degree + 1)
        else:
            d_star = degree(
                edge_index[0], num_nodes=pyg_data.num_nodes
            ).clamp(max=self.max_degree + 1)
        # d_t — current noisy-graph degree from visible state=1 edges.
        # degree deficit = d* − d_t: how many edges each node still needs.
        d_t = degree(
            edge_index[0], num_nodes=pyg_data.num_nodes
        ).clamp(max=self.max_degree + 1)
        # Log-scale normalization: better discriminability for power-law degree ranges.
        nodes  = self.node_embed(torch.log1p(d_star).unsqueeze(-1) / self._log_max_deg)
        if self.node_embed_t is not None:
            nodes = nodes + self.node_embed_t(torch.log1p(d_t).unsqueeze(-1) / self._log_max_deg)

        if self.rwse_k > 0:
            with torch.no_grad():
                rwse = _compute_rwse(edge_index, ep_bidi, pyg_data.num_nodes, self.rwse_k)
            nodes = nodes + self.rwse_proj(rwse)

        t        = self.time_pos_emb(t_node)
        t        = self.time_mlp(t)
        h        = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)

        for conv, norm_l, time_emb in zip(self.convs, self.norms, self.time_embs):
            t_emb  = time_emb(t)
            x_in   = torch.cat([nodes, t_emb], dim=-1)
            nodes  = conv(x_in, edge_index, ep_bidi)
            nodes  = torch.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
            nodes  = norm_l(nodes)
            nodes  = F.silu(nodes)
            nodes  = self.dropout(nodes)

            if not isinstance(self.gru, torch.nn.Identity):
                nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
                h     = self.dropout(h)
                nodes = nodes.squeeze(0)

            ctx      = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(contexts + ctx, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph, dim=0)
            nodes    = nodes + contexts

        # ── Node-level auxiliary head predictions (training only) ────────────
        # tau_node_true is set on pyg_data by ERGMAnchoredDiffusion._loss_impl
        # BEFORE the denoiser call, only during training when node-level losses
        # are active.  At sampling time it is absent (None), so this is a no-op.
        # deg_pred / tri_pred are written back to pyg_data and read by _loss_impl
        # to compute L_deg_node and L_tri_node.
        if getattr(pyg_data, 'tau_node_true', None) is not None:
            pyg_data.deg_pred = self.deg_head(nodes).squeeze(-1)   # (N,)
            pyg_data.tri_pred = self.tri_head(nodes).squeeze(-1)   # (N,)

        # ── Edge prediction with τ̂_ij conditioning ──────────────────────────
        query_edge_index = getattr(pyg_data, 'query_edge_index', None)
        edge_query_index = (query_edge_index if query_edge_index is not None
                            else pyg_data.full_edge_index)
        row, col = edge_query_index

        tau_feat = getattr(pyg_data, 'tau_feat', None)
        # Chunked pipeline: tau_proj → edge_emb → edge_attention — avoids
        # allocating (E_q, dim) all at once. Critical for PubMed where
        # E_q ≈ 388M at sampling → tau_emb alone would be 99 GB.
        edge_class = _chunked_edge_predict(
            self.tau_proj, self.edge_attention, self.dim,
            nodes, row, col, tau_feat)
        return pyg_data.log_node_attr, edge_class


# ─────────────────────────────────────────────────────────────────────────────
# GDiscPowerful  –  stronger discriminator based on GDisc notebook architecture
# Reference: GDisc class in revise_generation_citation_GAN_degree_bit.ipynb
# ─────────────────────────────────────────────────────────────────────────────

class GDiscPowerful(torch.nn.Module):
    """Powerful graph discriminator based on GDisc
    (revise_generation_citation_GAN_degree_bit.ipynb).

    Architecture:
      1. **Edge-to-node attention** — for every (directed) message edge, form
         cat[node_u, node_v, edge_logit] and pass through a deep MLP, then
         scatter-mean back to each node. This lets node representations absorb
         rich edge-level context before GCN message passing starts.
      2. **GCN layers** with soft P(edge=1) edge weights — expressive node
         refinement using the full probabilistic adjacency.
      3. **MiniAt_Film** per-edge representation pooled per graph, concatenated
         with the node-mean pool, fed to a sigmoid readout.

    Compared with GraphDiscriminator / GraphDiscriminatorFilm:
      • GCN replaces the pure attention-sum aggregation.
      • Edge log-probabilities explicitly modulate node updates (e_att block).
      • More expressive per-edge readout inherits FiLM conditioning.
    """
    def __init__(self, num_node_classes, num_edge_classes, dim=64, num_layers=3, att_heads=4):
        super().__init__()
        self.dim             = dim
        self.num_edge_classes = num_edge_classes

        self.node_embed = torch.nn.Linear(num_node_classes, dim)

        # Edge-to-node attention block (mirrors GDisc's e_att)
        e_att_in = dim * 2 + num_edge_classes   # cat[node_u, node_v, edge_logits]
        self.e_att = torch.nn.Sequential(
            torch.nn.Linear(e_att_in, dim * 4), torch.nn.ELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(dim * 4, dim * 2),  torch.nn.ELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(dim * 2, dim))
        self.edge_proj = torch.nn.Linear(dim, dim)

        # GCN layers (no self-loops; we manage the graph manually)
        self.gcn_layers = torch.nn.ModuleList(
            [pyg.nn.GCNConv(dim, dim, add_self_loops=False) for _ in range(num_layers)])
        self.gcn_norms  = torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim) for _ in range(num_layers)])

        # Per-edge representation via MiniAt_Film
        self.edge_attn = MiniAt_Film(
            node_dim=dim, in_edge_dim=dim * 2, out_edge_dim=dim,
            d_model=dim, num_heads=att_heads)
        self.edge_norm = torch.nn.LayerNorm(dim)

        # Readout: cat[node_pool, edge_pool, density] → scalar score ∈ (0,1).
        # +1: explicit per-graph edge density (see _graph_edge_density) — mean
        # pooling alone can't reliably signal "this graph is far too dense/sparse".
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(dim * 2 + 1, dim), torch.nn.SiLU(),
            torch.nn.Linear(dim, 1),       torch.nn.Sigmoid())

    def forward(self, batched_graph, log_node_attr_t, log_full_edge_attr_t, sparse=False):
        """
        sparse=False (default / p_sample guidance):
            Soft edge set — all candidate pairs, weighted by P(edge=1).
            Fully differentiable w.r.t. log_full_edge_attr_t.
        sparse=True  (disc_loss training):
            Hard edge set — only argmax=1 edges. Much smaller tensor,
            no gradient needed through log_full_edge_attr_t.
        """
        node_probs = torch.softmax(log_node_attr_t, dim=-1)      # (N, C_node)
        nodes      = self.node_embed(node_probs)                  # (N, dim)

        r0, c0 = batched_graph.full_edge_index
        N      = batched_graph.num_nodes

        # ── Build bidirectional edge set ──────────────────────────────────────
        if sparse:
            is_edge     = log_full_edge_attr_t.argmax(-1).bool()
            r_s, c_s    = r0[is_edge], c0[is_edge]
            row         = torch.cat([r_s, c_s])
            col         = torch.cat([c_s, r_s])
            ep          = torch.ones(row.size(0), dtype=nodes.dtype, device=nodes.device)
            e_attr      = log_full_edge_attr_t[is_edge]                    # (E_sp, 2)
            e_attr_bidi = torch.cat([e_attr, e_attr], dim=0)               # (2*E_sp, 2)
            r_e, c_e    = r_s, c_s
        else:
            edge_prob   = torch.softmax(log_full_edge_attr_t, dim=-1)[:, 1]
            row         = torch.cat([r0, c0])
            col         = torch.cat([c0, r0])
            ep          = edge_prob.repeat(2)
            e_attr_bidi = torch.cat(
                [log_full_edge_attr_t, log_full_edge_attr_t], dim=0)       # (2*E_full, 2)
            r_e, c_e    = r0, c0

        # ── Edge-to-node attention: edge logits modulate node representations ─
        if row.size(0) > 0:
            edge_feat  = torch.cat([nodes[row], nodes[col], e_attr_bidi], dim=-1)
            edge_att   = self.e_att(edge_feat)                             # (2E, dim)
            n_e_agg    = _scatter(edge_att, col, dim=0, dim_size=N, reduce='mean')
            nodes      = nodes + self.edge_proj(n_e_agg)

        # ── GCN layers with soft edge-probability weights ─────────────────────
        if row.size(0) > 0:
            edge_index = torch.stack([row, col], dim=0)
            for gcn, lnorm in zip(self.gcn_layers, self.gcn_norms):
                nodes = F.elu(lnorm(gcn(nodes, edge_index, edge_weight=ep)))

        # ── Node-level graph pooling ──────────────────────────────────────────
        node_repr  = _scatter(nodes, batched_graph.batch, reduce='mean', dim=0)  # (B, dim)
        num_graphs = node_repr.size(0)

        # ── Edge-level pooling via MiniAt_Film ────────────────────────────────
        # Soft mode (sparse=False, used by p_sample guidance) scores the full
        # candidate set every reverse step — chunk to cap peak memory.
        if r_e.size(0) > 0:
            edge_emb  = torch.cat([nodes[r_e], nodes[c_e]], dim=-1)        # (E, dim*2)
            edge_repr = F.silu(self.edge_norm(
                _chunked_edge_attention(self.edge_attn, nodes[r_e], nodes[c_e], edge_emb)))  # (E, dim)
            if not sparse:
                ep_w      = torch.softmax(log_full_edge_attr_t, dim=-1)[:, 1]
                edge_repr = ep_w.unsqueeze(-1) * edge_repr
            edge_batch      = batched_graph.batch[r_e]
            graph_edge_repr = _scatter(
                edge_repr, edge_batch, reduce='mean', dim=0,
                dim_size=num_graphs)                                         # (B, dim)
        else:
            graph_edge_repr = torch.zeros(
                num_graphs, self.dim, device=nodes.device, dtype=nodes.dtype)

        # ── Combined readout ──────────────────────────────────────────────────
        density  = _graph_edge_density(row, col, batched_graph.batch,
                                        batched_graph.nodes_per_graph, num_graphs)  # (B,)
        combined = torch.cat([node_repr, graph_edge_repr, density.unsqueeze(-1)], dim=-1)  # (B, 2*dim+1)
        return self.readout(combined).squeeze(-1)                            # (B,)


# ─────────────────────────────────────────────────────────────────────────────

def _chunked_edge_attention(edge_attention, node_u, node_v, edge_emb, chunk_size=50000):
    """Run a MiniAt_Film-style edge decoder in chunks to cap peak memory.

    MiniAt_Film's internal nn.MultiheadAttention allocates several
    (E, 3, embed_dim) intermediate tensors per call; at the full E≈700K+
    scale needed for dense evaluation/sampling (subsampling only applies to
    training), this can exceed even a large GPU's memory in one shot. Chunking
    the edge dimension keeps memory bounded by chunk_size regardless of E,
    at no behavioural difference since edges are scored independently.
    """
    E = node_u.shape[0]
    if E <= chunk_size:
        return edge_attention(node_u, node_v, edge_emb)
    outputs = []
    for start in range(0, E, chunk_size):
        end = start + chunk_size
        outputs.append(edge_attention(node_u[start:end], node_v[start:end], edge_emb[start:end]))
    return torch.cat(outputs, dim=0)


def _chunked_edge_predict(tau_proj, edge_attention, dim, nodes, row, col,
                          tau_feat, chunk_size=50_000):
    """Chunk the full τ_proj → edge_emb → edge_attention pipeline.

    Builds edge embeddings and runs the decoder in chunks of chunk_size pairs.
    Peak memory per chunk: chunk_size × 3×dim × 4 bytes (≈ 38 MB at 50K/dim=64),
    regardless of total |Q_t|. Critical for PubMed sampling where |Q_t|≈388M
    pairs would require 99 GB for tau_emb alone if processed in one shot.
    """
    E_q = row.shape[0]
    if E_q <= chunk_size:
        if tau_feat is not None:
            tau_log = torch.log1p(tau_feat.float()).unsqueeze(-1)
            tau_emb = tau_proj(tau_log)
        else:
            tau_emb = torch.zeros(E_q, dim, device=nodes.device)
        edge_emb = torch.cat([nodes[row], nodes[col], tau_emb], dim=-1)
        return edge_attention(nodes[row], nodes[col], edge_emb)

    out_chunks = []
    for start in range(0, E_q, chunk_size):
        end = min(start + chunk_size, E_q)
        r_c, c_c = row[start:end], col[start:end]
        n_u, n_v = nodes[r_c], nodes[c_c]
        if tau_feat is not None:
            tau_log_c = torch.log1p(tau_feat[start:end].float()).unsqueeze(-1)
            tau_emb_c = tau_proj(tau_log_c)
        else:
            tau_emb_c = torch.zeros(end - start, dim, device=nodes.device)
        edge_emb_c = torch.cat([n_u, n_v, tau_emb_c], dim=-1)
        out_chunks.append(edge_attention(n_u, n_v, edge_emb_c))
    return torch.cat(out_chunks, dim=0)


def _graph_edge_density(row, col, batch, nodes_per_graph, num_graphs):
    """Per-graph edge density from a bidirectional (row, col) edge list.

    Mean/attention pooling over node embeddings is largely invariant to raw
    edge count (it averages, not sums), so discriminators built only from such
    pooling have no reliable channel for detecting "this graph is far too
    dense/sparse overall" — exactly the failure mode this is meant to guard
    against. row/col list each undirected edge twice (u->v and v->u), so the
    true edge count per graph is half the scatter-summed count.
    """
    device = nodes_per_graph.device
    if row.numel() == 0:
        return torch.zeros(num_graphs, device=device)
    edge_batch = batch[row]
    n_edges = _scatter(torch.ones(row.size(0), device=device), edge_batch,
                        dim=0, dim_size=num_graphs, reduce='sum') / 2.0
    n_nodes = nodes_per_graph.float().to(device)
    max_pairs = (n_nodes * (n_nodes - 1) / 2).clamp(min=1.0)
    return n_edges / max_pairs


class GraphDiscriminator(torch.nn.Module):
    """Attention-based graph discriminator for GAN-style diffusion guidance.

    Distinguishes G_t ("real", less noisy) from G_{t+1} ("fake", more noisy).

    Uses the same AttentionGINConv attention design as the denoiser:
      - Multi-head attention gating normalised per destination node
      - Soft edge probabilities as an additional multiplicative gate
      - Bidirectional message passing over the full (upper-triangular) edge set
    Fully differentiable w.r.t. log_node_attr_t and log_full_edge_attr_t,
    which is required for the gradient guidance step in p_sample.
    """
    def __init__(self, num_node_classes, num_edge_classes, dim=64, num_layers=3, att_heads=4):
        super().__init__()
        self.node_embed = torch.nn.Linear(num_node_classes, dim)
        self.att_heads  = att_heads
        # Per-layer attention + message projections
        self.att_fc  = torch.nn.ModuleList(
            [torch.nn.Linear(dim * 2, att_heads, bias=False) for _ in range(num_layers)])
        self.att_out = torch.nn.ModuleList(
            [torch.nn.Linear(att_heads, 1, bias=False) for _ in range(num_layers)])
        self.msg_layers = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in range(num_layers)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim) for _ in range(num_layers)])
        # +1: explicit per-graph edge density (see _graph_edge_density) — mean
        # pooling alone can't reliably signal "this graph is far too dense/sparse".
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, batched_graph, log_node_attr_t, log_full_edge_attr_t, sparse=False):
        """
        sparse=False (default / p_sample guidance):
            Full soft edge set — differentiable w.r.t. log_full_edge_attr_t.
        sparse=True  (disc_loss training):
            Only the actual (hard-argmax) edges — drastically smaller edge
            tensor, no gradient needed through log_full_edge_attr_t.
        """
        node_probs = torch.softmax(log_node_attr_t, dim=-1)           # (N, C_node)
        nodes = self.node_embed(node_probs)                            # (N, dim)

        r0, c0 = batched_graph.full_edge_index
        N = batched_graph.num_nodes

        if sparse:
            # Hard-sample: keep only positions where argmax = 1 (edge exists)
            is_edge = log_full_edge_attr_t.argmax(-1).bool()          # (E_full,)
            r0_s, c0_s = r0[is_edge], c0[is_edge]
            row = torch.cat([r0_s, c0_s])
            col = torch.cat([c0_s, r0_s])
            # All selected edges are actual edges → uniform weight 1
            ep = torch.ones(row.size(0), dtype=nodes.dtype, device=nodes.device)
        else:
            # Soft mode: use P(edge=1) as a differentiable gate
            edge_prob = torch.softmax(log_full_edge_attr_t, dim=-1)[:, 1]  # (E_full,)
            row = torch.cat([r0, c0])
            col = torch.cat([c0, r0])
            ep  = edge_prob.repeat(2)                                  # (2*E_full,)

        for att_fc, att_out, msg_layer, norm in zip(
                self.att_fc, self.att_out, self.msg_layers, self.norms):
            if row.size(0) == 0:                                       # empty graph
                break
            # Multi-head attention (2E, att_heads) → scalar weight per edge
            alpha = F.leaky_relu(
                att_fc(torch.cat([nodes[row], nodes[col]], dim=-1)), negative_slope=0.2)
            alpha = att_out(alpha).squeeze(-1)                         # (2E,)
            alpha = pyg.utils.softmax(alpha, col, num_nodes=N)        # (2E,) per-dst softmax
            weight = alpha * ep                                        # (2E,)
            agg = _scatter(
                weight.unsqueeze(-1) * nodes[row], col,
                dim=0, dim_size=N, reduce='sum')                       # (N, dim)
            nodes = nodes + F.silu(norm(msg_layer(agg)))

        graph_repr = _scatter(nodes, batched_graph.batch,
                                           reduce='mean', dim=0)       # (B, dim)
        density = _graph_edge_density(row, col, batched_graph.batch,
                                       batched_graph.nodes_per_graph,
                                       graph_repr.size(0))              # (B,)
        graph_repr = torch.cat([graph_repr, density.unsqueeze(-1)], dim=-1)
        return self.readout(graph_repr).squeeze(-1)                    # (B,)


class GraphDiscriminatorFilm(GraphDiscriminator):
    """GraphDiscriminator enhanced with MiniAt_Film edge attention (--disc_film).

    After the same node message passing as ``GraphDiscriminator``, applies
    ``MiniAt_Film`` to compute per-edge representations and pools them per
    graph. The final discrimination score combines both node-level and
    edge-level pooled features:

        node_repr  = MeanPool(nodes per graph)                 # (B, dim)
        edge_repr  = MeanPool( MiniAt_Film(u, v, cat[u,v]) )  # (B, dim)
        score      = σ( Linear( cat[node_repr, edge_repr] ) )  # (B,)

    Edge features are weighted by P(edge=1) in soft (non-sparse) mode so
    the pooled representation focuses on likely edges, and the gradient
    through MiniAt_Film provides richer per-edge guidance during p_sample.

    Reference: ``GDisc`` class in
    ``revise_generation_citation_GAN_degree_bit.ipynb``.
    """
    def __init__(self, num_node_classes, num_edge_classes, dim=64,
                 num_layers=3, att_heads=4):
        super().__init__(num_node_classes, num_edge_classes, dim, num_layers, att_heads)
        self.dim = dim

        # MiniAt_Film for per-edge representations
        self.edge_attn = MiniAt_Film(
            node_dim=dim, in_edge_dim=dim * 2, out_edge_dim=dim,
            d_model=dim, num_heads=att_heads,
        )
        self.edge_norm = torch.nn.LayerNorm(dim)

        # Override parent readout: now takes (node_pool ‖ edge_pool) → scalar
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim), torch.nn.SiLU(),
            torch.nn.Linear(dim, 1),       torch.nn.Sigmoid(),
        )

    def forward(self, batched_graph, log_node_attr_t, log_full_edge_attr_t,
                sparse=False):
        node_probs = torch.softmax(log_node_attr_t, dim=-1)   # (N, C_node)
        nodes      = self.node_embed(node_probs)               # (N, dim)

        r0, c0 = batched_graph.full_edge_index
        N = batched_graph.num_nodes

        # ── Build bidirectional message-passing edge set ─────────────────────
        if sparse:
            is_edge = log_full_edge_attr_t.argmax(-1).bool()
            r0_s, c0_s = r0[is_edge], c0[is_edge]
            row = torch.cat([r0_s, c0_s])
            col = torch.cat([c0_s, r0_s])
            ep  = torch.ones(row.size(0), dtype=nodes.dtype, device=nodes.device)
            r0_e, c0_e = r0_s, c0_s        # subset for edge attention
        else:
            edge_prob = torch.softmax(log_full_edge_attr_t, dim=-1)[:, 1]
            row = torch.cat([r0, c0])
            col = torch.cat([c0, r0])
            ep  = edge_prob.repeat(2)
            r0_e, c0_e = r0, c0            # full set for edge attention

        # ── Node message passing (same as GraphDiscriminator) ────────────────
        for att_fc, att_out, msg_layer, norm in zip(
                self.att_fc, self.att_out, self.msg_layers, self.norms):
            if row.size(0) == 0:
                break
            alpha  = F.leaky_relu(
                att_fc(torch.cat([nodes[row], nodes[col]], dim=-1)), negative_slope=0.2)
            alpha  = att_out(alpha).squeeze(-1)
            alpha  = pyg.utils.softmax(alpha, col, num_nodes=N)
            weight = alpha * ep
            agg    = _scatter(
                weight.unsqueeze(-1) * nodes[row], col,
                dim=0, dim_size=N, reduce='sum')
            nodes  = nodes + F.silu(norm(msg_layer(agg)))

        # ── Node-level graph pooling ─────────────────────────────────────────
        node_repr = _scatter(
            nodes, batched_graph.batch, reduce='mean', dim=0)  # (B, dim)
        num_graphs = node_repr.size(0)                          # inferred B

        # ── Edge-level features via MiniAt_Film ──────────────────────────────
        # Soft mode (sparse=False, used by p_sample guidance) scores the full
        # candidate set every reverse step — chunk to cap peak memory.
        if r0_e.size(0) > 0:
            edge_emb  = torch.cat([nodes[r0_e], nodes[c0_e]], dim=-1) # (E, dim*2)
            edge_repr = F.silu(self.edge_norm(
                _chunked_edge_attention(self.edge_attn, nodes[r0_e], nodes[c0_e], edge_emb)))  # (E, dim)

            if not sparse:
                # Soft weight by P(edge=1) so only likely edges contribute
                ep_w      = torch.softmax(log_full_edge_attr_t, dim=-1)[:, 1]
                edge_repr = ep_w.unsqueeze(-1) * edge_repr             # (E, dim)

            edge_batch       = batched_graph.batch[r0_e]
            graph_edge_repr  = _scatter(
                edge_repr, edge_batch, reduce='mean', dim=0,
                dim_size=num_graphs)                                   # (B, dim)
        else:
            graph_edge_repr = torch.zeros(
                num_graphs, self.dim, device=nodes.device)

        # ── Combined readout ─────────────────────────────────────────────────
        combined = torch.cat([node_repr, graph_edge_repr], dim=-1)     # (B, 2*dim)
        return self.readout(combined).squeeze(-1)                      # (B,)


class TGNN_degree_guided(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.embedding_sel = torch.nn.Embedding(2, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*4, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim*2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim*2, dim*2)
        )
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge):
        if hasattr(pyg_data, 'edge_index_t'):
            edge_index = pyg_data.edge_index_t

        else: 
            edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
            is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

            edge_index = pyg_data.full_edge_index[:, is_edge_indices]
            edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)

        nodes_t = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()
        node_selection = torch.zeros_like(nodes_t)


        nodes_t = nodes_t[..., None] / self.max_degree  # I prefer to make it embedding later
        # Compute degree from edge_index if not pre-computed
        if hasattr(pyg_data, 'degree'):
            nodes_0 = pyg_data.degree
        else:
            nodes_0 = pyg.utils.degree(pyg_data.edge_index[0], num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()
        nodes_0 = nodes_0[..., None] / self.max_degree
        node_selection[pyg_data.active_node_indices] = 1
        node_selection = node_selection.long()
        
        nodes = torch.cat([self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_in(nodes)

        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = _scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = _scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        row = pyg_data.full_edge_index[0].index_select(0, pyg_data.active_edge_indices)
        col = pyg_data.full_edge_index[1].index_select(0, pyg_data.active_edge_indices)

        nodes = torch.cat([nodes, self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_out_mlp(nodes)

        edge_emb = nodes[row] + nodes[col]
        edge_class = self.final_out(edge_emb)

        return pyg_data.log_node_attr, edge_class


# ─────────────────────────────────────────────────────────────────────────────
#  ScalableGNN  —  f_θ denoiser  (Algorithm 2 in pseudocode)
# ─────────────────────────────────────────────────────────────────────────────

class ScalableGNN(torch.nn.Module):
    """f_θ: MLP message-passing denoiser for ScalableGraphDiffusion.

    L-layer MLP message-passing runs on the **surviving** sparse edge set E_t.
    After message passing, multi-head edge attention refines edge representations,
    then three output heads produce:
        X̂_0   (N, d_x)      predicted clean node features
        F̂_0   (|E_t|, d_f)  predicted clean edge features (surviving edges)
        p̂_ij  (|P_t|,)      edge-existence probabilities over P^(t) = full_edge_index

    Constructor signature matches the other dynamics_fn classes so that
    ``get_model()`` in model.py can instantiate it identically.

    Extra fields expected on ``pyg_data`` (set by ScalableGraphDiffusion):
        x_cont           (N, d_x)       noisy continuous node features
        surv_edge_index  (2, |E_t|)     surviving edges (upper-triangular, i < j)
        edge_feat_cont   (|E_t|, d_f)   noisy surviving-edge features
    """

    def __init__(
        self,
        num_node_classes,
        num_edge_classes,
        dim,
        num_steps,
        num_heads=None,
        dropout=0.0,
        # accepted but unused (API compatibility)
        max_degree=None,
        norm='None',
        gru=False,
        degree=False,
        augmented_features=None,
        return_node_class=False,
        **kwargs,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = [4, 4, 4, 1]

        self.num_layers = len(num_heads)
        # Attention heads for the edge-attention module (use first entry, min with dim)
        K = min(num_heads[0], dim)
        while dim % K != 0 and K > 1:
            K -= 1
        self.K = K
        self.d_h = dim // K
        self.dim = dim
        self.d_x = num_node_classes
        self.d_f = num_edge_classes

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim))

        # ── Input encoders ────────────────────────────────────────────────────
        # [X_t || γ] → H^(0)
        self.node_input = torch.nn.Linear(num_node_classes + dim, dim)
        # [F_t || γ_src] → E^(0)
        self.edge_input = torch.nn.Linear(num_edge_classes + dim, dim)

        # ── MLP message + update per layer ────────────────────────────────────
        self.msg_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim * 3, dim * 2), torch.nn.SiLU(),
                torch.nn.Linear(dim * 2, dim))
            for _ in range(self.num_layers)
        ])
        self.update_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim * 2, dim * 2), torch.nn.SiLU(),
                torch.nn.Linear(dim * 2, dim))
            for _ in range(self.num_layers)
        ])
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim) for _ in range(self.num_layers)])

        # ── Multi-head edge attention (K heads over E_t) ──────────────────────
        self.W_Q = torch.nn.ModuleList(
            [torch.nn.Linear(dim, self.d_h, bias=False) for _ in range(K)])
        self.W_K = torch.nn.ModuleList(
            [torch.nn.Linear(dim, self.d_h, bias=False) for _ in range(K)])
        # Value: MLP([h_j || e_ij]) → d_h
        self.W_V = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim * 2, self.d_h), torch.nn.SiLU())
            for _ in range(K)
        ])
        self.W_O = torch.nn.Linear(dim, dim)   # K * d_h = dim

        # ── Output heads ──────────────────────────────────────────────────────
        self.node_out = torch.nn.Linear(dim, num_node_classes)

        # F̂_0: [x̂_i || x̂_j || ê_ij] → d_f
        self.edge_feat_out = torch.nn.Sequential(
            torch.nn.Linear(num_node_classes * 2 + dim, dim), torch.nn.SiLU(),
            torch.nn.Linear(dim, num_edge_classes))

        # Symmetrized bilinear edge predictor  W_e ∈ R^{dim×dim}
        self.W_e = torch.nn.Linear(dim, dim, bias=False)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, pyg_data, t_node, t_edge):
        """
        Returns:
            X_hat   (N, d_x)
            F_hat   (|E_t|, d_f)
            p_hat   (|P_t|,)       — P_t = full_edge_index
            P_t     (2, |P_t|)
            H       (N, dim)
        """
        device = pyg_data.x_cont.device
        N = pyg_data.x_cont.shape[0]

        # Time embedding γ per node
        gamma = self.time_mlp(self.time_emb(t_node))             # (N, dim)

        # Input encoding
        X_t = pyg_data.x_cont                                    # (N, d_x)
        H = self.node_input(torch.cat([X_t, gamma], dim=-1))     # (N, dim)

        surv_ei = pyg_data.surv_edge_index                        # (2, |E_t|)
        has_edges = surv_ei.shape[1] > 0

        if has_edges:
            F_t = pyg_data.edge_feat_cont                         # (|E_t|, d_f)
            gamma_e = gamma[surv_ei[0]]                           # (|E_t|, dim)
            E = self.edge_input(torch.cat([F_t, gamma_e], dim=-1))  # (|E_t|, dim)
            # Bidirectional for message passing
            row = torch.cat([surv_ei[0], surv_ei[1]])
            col = torch.cat([surv_ei[1], surv_ei[0]])
            E_bidi = torch.cat([E, E], dim=0)                     # (2|E_t|, dim)
        else:
            E = torch.zeros(0, self.dim, device=device)

        # ── L-layer MLP message passing ───────────────────────────────────────
        for l in range(self.num_layers):
            if has_edges:
                msg_in = torch.cat([H[row], H[col], E_bidi], dim=-1)
                msgs = self.msg_fns[l](msg_in)                    # (2|E_t|, dim)
                agg = _scatter(msgs, col, dim=0, dim_size=N, reduce='sum')
            else:
                agg = torch.zeros(N, self.dim, device=device)
            H_new = self.update_fns[l](torch.cat([H, agg], dim=-1))
            H = self.layer_norms[l](H + H_new)
            # Guard against NaN/Inf compounding over long autoregressive sampling
            # rollouts (1000 sequential steps feed H back into itself via cur_x/
            # cur_ef) — same defensive pattern used in GIN / GraphDenoising.
            H = torch.nan_to_num(H, nan=0.0, posinf=10.0, neginf=-10.0)
            H = self.dropout(H)

        # ── Multi-head edge attention over E_t ───────────────────────────────
        if has_edges:
            ri, ci = surv_ei[0], surv_ei[1]
            head_outs = []
            for k in range(self.K):
                q = self.W_Q[k](H[ri])                            # (|E_t|, d_h)
                kk = self.W_K[k](H[ci])
                v = self.W_V[k](torch.cat([H[ci], E], dim=-1))   # (|E_t|, d_h)
                score = (q * kk).sum(-1) / (self.d_h ** 0.5)     # (|E_t|,)
                alpha = pyg.utils.softmax(score, ri, num_nodes=N)
                head_outs.append(alpha.unsqueeze(-1) * v)
            E_hat = self.W_O(torch.cat(head_outs, dim=-1))        # (|E_t|, dim)
        else:
            # No surviving edges: run a zero dummy through every edge parameter
            # so DDP sees gradients on all parameters every forward pass.
            dummy_f = torch.zeros(1, self.d_f,  device=device)
            dummy_g = torch.zeros(1, self.dim,  device=device)
            dummy_h = torch.zeros(1, self.dim,  device=device)
            dummy_e = self.edge_input(torch.cat([dummy_f, dummy_g], dim=-1)) * 0
            for l in range(self.num_layers):
                _m = self.msg_fns[l](torch.cat([dummy_h, dummy_h, dummy_e], dim=-1)) * 0
                _u = self.update_fns[l](torch.cat([dummy_h, _m], dim=-1)) * 0
                dummy_h = dummy_h + _u                            # all zero, grad flows
            dummy_kv = torch.zeros(1, self.dim, device=device)
            dummy_v_in = torch.cat([dummy_kv, dummy_e], dim=-1)
            _heads = [self.W_Q[k](dummy_kv) * 0
                      + self.W_K[k](dummy_kv) * 0
                      + self.W_V[k](dummy_v_in) * 0
                      for k in range(self.K)]
            E_hat = self.W_O(torch.cat(_heads, dim=-1)) * 0      # (1, dim) × 0

        # ── Output heads ──────────────────────────────────────────────────────
        X_hat = self.node_out(H)                                  # (N, d_x)
        X_hat = torch.nan_to_num(X_hat, nan=0.0, posinf=10.0, neginf=-10.0)

        if has_edges:
            F_hat = self.edge_feat_out(
                torch.cat([X_hat[ri], X_hat[ci], E_hat], dim=-1))  # (|E_t|, d_f)
        else:
            dummy_xh = torch.zeros(1, self.d_x, device=device)
            F_hat = self.edge_feat_out(
                torch.cat([dummy_xh, dummy_xh, E_hat], dim=-1)) * 0  # (1, d_f) × 0
        F_hat = torch.nan_to_num(F_hat, nan=0.0, posinf=10.0, neginf=-10.0)

        # Sparse bilinear predictor over P^(t) = full_edge_index
        P_t = pyg_data.full_edge_index                            # (2, |P|)
        pi, pj = P_t[0], P_t[1]
        Whi = self.W_e(H[pi])
        Whj = self.W_e(H[pj])
        s = ((H[pi] * Whj).sum(-1) + (H[pj] * Whi).sum(-1)) / (2.0 * (self.dim ** 0.5))
        # sigmoid(finite) is always in (0,1); only sigmoid(NaN)=NaN needs guarding —
        # fall back to 0.5 (max-entropy / "unknown") rather than letting NaN reach
        # the downstream torch.bernoulli call in sample(), which CUDA-asserts on it.
        p_hat = torch.nan_to_num(torch.sigmoid(s), nan=0.5)       # (|P|,)

        return X_hat, F_hat, p_hat, P_t, H


# ─────────────────────────────────────────────────────────────────────────────
#  ScalableDisc  —  Disc_φ discriminator  (Algorithm 3 in pseudocode)
# ─────────────────────────────────────────────────────────────────────────────

class ScalableDisc(torch.nn.Module):
    """Disc_φ: graph discriminator for ScalableGraphDiffusion (Algorithm 3).

    L' message-passing layers with spectral normalization on all linear
    layers (for GAN training stability), global mean pooling, scalar
    realness score ∈ (0, 1).

    Interface differs from the existing GraphDiscriminator: takes continuous
    tensors directly rather than log-probability vectors.

    Args:
        d_x        (int): node feature dim (= num_node_classes)
        d_f        (int): edge feature dim (= num_edge_classes)
        dim        (int): hidden dimension
        num_layers (int): number of MP layers (L')
    """

    def __init__(self, d_x, d_f, dim=64, num_layers=3):
        super().__init__()
        SN = torch.nn.utils.spectral_norm

        self.node_enc = SN(torch.nn.Linear(d_x, dim))

        self.msg_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                SN(torch.nn.Linear(dim * 2 + d_f, dim * 2)), torch.nn.SiLU(),
                SN(torch.nn.Linear(dim * 2, dim)))
            for _ in range(num_layers)
        ])
        self.update_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                SN(torch.nn.Linear(dim * 2, dim * 2)), torch.nn.SiLU(),
                SN(torch.nn.Linear(dim * 2, dim)))
            for _ in range(num_layers)
        ])
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim) for _ in range(num_layers)])

        self.readout = torch.nn.Sequential(
            SN(torch.nn.Linear(dim, dim)), torch.nn.SiLU(),
            SN(torch.nn.Linear(dim, 1)),  torch.nn.Sigmoid())

    def forward(self, x, edge_index, edge_feat, batch, num_nodes):
        """
        Args:
            x          (N, d_x)   continuous node features
            edge_index (2, |E|)   half-edge index (i < j); made bidirectional internally
            edge_feat  (|E|, d_f) continuous edge features
            batch      (N,)       graph membership
            num_nodes  int        total N (for scatter dim_size)
        Returns:
            scores (B,) ∈ (0, 1)   — 1 ≈ real, 0 ≈ fake
        """
        device = x.device
        N = num_nodes
        H = self.node_enc(x)                                      # (N, dim)

        has_edges = edge_index.shape[1] > 0
        if has_edges:
            row = torch.cat([edge_index[0], edge_index[1]])
            col = torch.cat([edge_index[1], edge_index[0]])
            ef  = torch.cat([edge_feat, edge_feat], dim=0)        # bidi features

        for msg_fn, upd_fn, lnorm in zip(
                self.msg_fns, self.update_fns, self.layer_norms):
            if has_edges:
                msg_in = torch.cat([H[row], H[col], ef], dim=-1)
                agg = _scatter(msg_fn(msg_in), col, dim=0,
                               dim_size=N, reduce='sum')
            else:
                agg = torch.zeros_like(H)
            H = lnorm(H + upd_fn(torch.cat([H, agg], dim=-1)))

        graph_repr = _scatter(H, batch, dim=0, reduce='mean')     # (B, dim)
        return self.readout(graph_repr).squeeze(-1)                # (B,)


# ─────────────────────────────────────────────────────────────────────────────
#  ScalableDiscDiffGAN  —  Diffusion-GAN time-conditioned discriminator
#
#  Reference: Wang et al., "Diffusion-GAN: Training GANs with Diffusion"
#             https://arxiv.org/abs/2206.02262
#
#  Key design difference from ScalableDisc:
#    - Takes timestep t as input and conditions ALL node representations on it
#    - Both real and fake samples are presented at the SAME noise level t
#    - The discriminator learns separate criteria for each noise level,
#      providing gradient signal from coarse (high t) to fine (low t) structure
# ─────────────────────────────────────────────────────────────────────────────

class ScalableDiscDiffGAN(torch.nn.Module):
    """Time-conditioned graph discriminator following Diffusion-GAN.

    D(G_t, t) distinguishes q(G_t | G_0^real) from q(G_t | Ĝ_0^fake),
    where both operate at the SAME noise level t. This is principled:
    - Adjacent-level discrimination (t vs t+1) gives near-identical inputs.
    - Clean-vs-noisy discrimination ignores noise-level structure.
    - Same-level discrimination forces the model to match the true reverse
      distribution at EVERY noise level simultaneously.

    Time conditioning: sinusoidal embedding of t is concatenated to each
    node's feature vector before the first linear projection. This lets the
    discriminator adapt its structural criterion to the signal-to-noise ratio
    at each step — at high t it focuses on coarse density/connectivity, at
    low t it refines to clustering and local motif patterns.

    Spectral normalisation on all linear layers for training stability.
    """

    def __init__(self, d_x, d_f, dim=64, num_layers=3, num_steps=1000):
        super().__init__()
        SN = torch.nn.utils.spectral_norm

        # Time embedding: sinusoidal → MLP → dim
        self.time_emb   = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.time_proj  = torch.nn.Sequential(
            SN(torch.nn.Linear(dim, dim)), torch.nn.SiLU())

        # Node encoder: [x || time_emb] → dim
        self.node_enc = SN(torch.nn.Linear(d_x + dim, dim))

        self.msg_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                SN(torch.nn.Linear(dim * 2 + d_f, dim * 2)), torch.nn.SiLU(),
                SN(torch.nn.Linear(dim * 2, dim)))
            for _ in range(num_layers)
        ])
        self.update_fns = torch.nn.ModuleList([
            torch.nn.Sequential(
                SN(torch.nn.Linear(dim * 2, dim * 2)), torch.nn.SiLU(),
                SN(torch.nn.Linear(dim * 2, dim)))
            for _ in range(num_layers)
        ])
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim) for _ in range(num_layers)])

        self.readout = torch.nn.Sequential(
            SN(torch.nn.Linear(dim, dim)), torch.nn.SiLU(),
            SN(torch.nn.Linear(dim, 1)),  torch.nn.Sigmoid())

    def forward(self, x, edge_index, edge_feat, batch, num_nodes, t_graph):
        """
        Args:
            x          (N, d_x)   continuous node features at noise level t
            edge_index (2, |E|)   half-edge index (i < j)
            edge_feat  (|E|, d_f) edge features at noise level t
            batch      (N,)       graph membership
            num_nodes  int        total N
            t_graph    (B,)       diffusion timestep per graph (long tensor)
        Returns:
            scores (B,) ∈ (0, 1)   — 1 ≈ real noised sample, 0 ≈ fake noised
        """
        device = x.device
        N = num_nodes

        # Per-node time embedding (broadcast from graph-level t_graph)
        t_emb = self.time_proj(self.time_emb(t_graph[batch]))     # (N, dim)
        H = self.node_enc(torch.cat([x, t_emb], dim=-1))          # (N, dim)

        has_edges = edge_index.shape[1] > 0
        if has_edges:
            row = torch.cat([edge_index[0], edge_index[1]])
            col = torch.cat([edge_index[1], edge_index[0]])
            ef  = torch.cat([edge_feat, edge_feat], dim=0)

        for msg_fn, upd_fn, lnorm in zip(
                self.msg_fns, self.update_fns, self.layer_norms):
            if has_edges:
                msg_in = torch.cat([H[row], H[col], ef], dim=-1)
                agg    = _scatter(msg_fn(msg_in), col, dim=0,
                                  dim_size=N, reduce='sum')
            else:
                agg = torch.zeros_like(H)
            H = lnorm(H + upd_fn(torch.cat([H, agg], dim=-1)))

        graph_repr = _scatter(H, batch, dim=0, reduce='mean')      # (B, dim)
        return self.readout(graph_repr).squeeze(-1)                 # (B,)