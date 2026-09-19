import torch
import random
import networkx as nx
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
from torch_geometric import transforms as T
from torch.nn import functional as F
from layers.layers import BitModel, NodeModel

FEATURE_EXTRACTOR = {}


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def unpack_deg_matrix(degs):
    res = []
    for deg in degs:
        deg = deg.long().tolist()
        r = []
        for d in deg:
            if (sum(r)==0) or (d > 0):
                r.append(d)
        res.append(r)
    return res


def deg_hist_to_deg_seq(deg_hist):
    ret = torch.zeros(sum(deg_hist))
    cum = 0
    for d, num_nodes in enumerate(deg_hist):
        ret[cum:cum+num_nodes] = d+1
        cum = cum+num_nodes
    return ret

def preprocess(g, degree=False, augmented_features=[], edge_fraction=1.0):
    """Build PyG data from a NetworkX graph.

    edge_fraction < 1.0 activates sparse mode: full_edge_index stores all true
    edges plus a random edge_fraction sample of non-edges.  This reduces every
    subsequent O(N²) tensor (q_sample, _build_edge_query scan, pos_weight
    scatter) to O(λN²).  total_cands = N(N-1)/2 is stored separately so the
    ELBO normalisation and pos_weight remain correct.  Call
    resample_neg_edges(pyg_data) each training step to refresh the non-edge
    sample and keep the Horvitz-Thompson estimator unbiased over training.
    """
    if isinstance(g, nx.Graph):
        pyg_data = pyg.utils.from_networkx(g)
        adj = torch.from_numpy(nx.to_numpy_array(g).astype(np.int64)).long()
    elif isinstance(g, pyg.data.Data):
        pyg_data = g
        if g.edge_attr is not None and g.edge_attr.numel() > 0:
            # Typed edges (e.g. ZINC bond types): preserve values in dense adj.
            # edge_attr must already be 1-indexed (0 reserved for "no edge").
            ea = g.edge_attr.view(-1).long()
            adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes, edge_attr=ea)[0].long()
        else:
            adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0].long()
    else:
        raise NotImplementedError()

    row, col = torch.triu_indices(pyg_data.num_nodes, pyg_data.num_nodes, 1)
    full_attr  = adj[row, col]                          # (N²/2,) edge labels (0=absent)

    pyg_data.total_cands = row.shape[0]                 # N(N-1)/2, always stored

    if edge_fraction < 1.0:
        pos_idx = (full_attr == 1).nonzero(as_tuple=True)[0]
        neg_idx = (full_attr == 0).nonzero(as_tuple=True)[0]
        n_neg   = max(1, int(round(edge_fraction * neg_idx.numel())))
        perm    = torch.randperm(neg_idx.numel())[:n_neg]
        sel     = torch.cat([pos_idx, neg_idx[perm]])
        pyg_data.full_edge_index = torch.stack([row[sel], col[sel]])
        pyg_data.full_edge_attr  = full_attr[sel]
        # Cache pos/neg source rows for fast per-step resampling
        pyg_data._sparse_pos_row  = row[pos_idx]
        pyg_data._sparse_pos_col  = col[pos_idx]
        pyg_data._sparse_neg_row  = row[neg_idx]
        pyg_data._sparse_neg_col  = col[neg_idx]
        pyg_data._edge_fraction   = edge_fraction
    else:
        pyg_data.full_edge_index = torch.stack([row, col])
        pyg_data.full_edge_attr  = full_attr

    if not hasattr(pyg_data, 'node_attr'):
        pyg_data.node_attr = torch.zeros(pyg_data.num_nodes, dtype=torch.long)

    # target_degree: permanent ground-truth degree for PA-DACA conditioning.
    # Must use only true edges (any bond), not all N(N-1)/2 candidate pairs.
    # full_attr > 0 works for both binary (0/1) and typed (0=no-bond, 1..K=bond types).
    edge_mask = (full_attr > 0)
    edge_row  = row[edge_mask]
    edge_col  = col[edge_mask]
    adj_bidi  = torch.cat([torch.stack([edge_row, edge_col]),
                           torch.stack([edge_col, edge_row])], dim=1)
    pyg_data.target_degree = pyg.utils.degree(
        adj_bidi[0], num_nodes=pyg_data.num_nodes).long()

    if degree:
        pyg_data.degree = pyg.utils.degree(pyg_data.edge_index[0]).long()

    for augmented_feature in augmented_features:
        setattr(pyg_data, augmented_feature, FEATURE_EXTRACTOR[augmented_feature]['func'](pyg_data))

    return pyg_data


def resample_neg_edges(pyg_data):
    """Refresh the non-edge component of a sparse full_edge_index.

    Call once per training step (e.g. in NetworkDataset.__getitem__) to keep
    the Horvitz-Thompson estimator unbiased: each non-edge pair is seen with
    probability edge_fraction over many steps.  No-ops when edge_fraction=1.0.
    """
    if not hasattr(pyg_data, '_edge_fraction'):
        return
    ef       = pyg_data._edge_fraction
    neg_row  = pyg_data._sparse_neg_row
    neg_col  = pyg_data._sparse_neg_col
    n_neg    = max(1, int(round(ef * neg_row.numel())))
    perm     = torch.randperm(neg_row.numel())[:n_neg]
    n_pos    = pyg_data._sparse_pos_row.numel()
    new_row  = torch.cat([pyg_data._sparse_pos_row, neg_row[perm]])
    new_col  = torch.cat([pyg_data._sparse_pos_col, neg_col[perm]])
    new_attr = torch.cat([torch.ones(n_pos, dtype=torch.long),
                          torch.zeros(n_neg, dtype=torch.long)])
    pyg_data.full_edge_index = torch.stack([new_row, new_col])
    pyg_data.full_edge_attr  = new_attr


def collate_fn(pyg_datas, repeat=1):
    pyg_datas = sum([[pyg_data.clone() for _ in range(repeat)]for pyg_data in pyg_datas],[])
    batched_data = pyg.data.Batch.from_data_list(pyg_datas)
    batched_data.nodes_per_graph = torch.tensor([pyg_data.num_nodes for pyg_data in pyg_datas])
    # edges_per_graph = actual size of (sparse) full_edge_index — used to size t_edge
    batched_data.edges_per_graph = torch.tensor(
        [pyg_data.full_edge_index.shape[1] for pyg_data in pyg_datas])
    # total_cands_per_graph = N(N-1)/2 — used for pos_weight and ELBO normalisation
    batched_data.total_cands_per_graph = torch.tensor(
        [getattr(pyg_data, 'total_cands', pyg_data.num_nodes*(pyg_data.num_nodes-1)//2)
         for pyg_data in pyg_datas])
    return batched_data


def _compute_sbm_blocks(pyg_data, n_clusters):
    """Spectral clustering on a training graph; returns integer block labels (N,).

    Blocks are sorted by ascending mean node degree so that block 0 is always
    the lowest-degree block — this makes the global p_kl accumulation consistent
    across graphs with different spectral-clustering label permutations.

    Returns None if clustering fails (graph too small or disconnected).
    """
    try:
        from sklearn.cluster import SpectralClustering
    except ImportError:
        return None

    n  = pyg_data.num_nodes
    if n < n_clusters * 2:  # too few nodes to form meaningful blocks
        return None

    ei   = pyg_data.full_edge_index  # upper-triangular (2, E)
    attr = pyg_data.full_edge_attr.float()  # (E,) binary

    A = np.zeros((n, n), dtype=np.float32)
    rows = ei[0].numpy()
    cols = ei[1].numpy()
    vals = attr.numpy()
    A[rows, cols] = vals
    A[cols, rows] = vals  # symmetrize

    try:
        sc     = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                    random_state=42, n_init=5)
        labels = sc.fit_predict(A)
    except Exception:
        return None

    # Align blocks by ascending mean degree so the ordering is consistent
    if hasattr(pyg_data, 'target_degree'):
        d = pyg_data.target_degree.float().numpy()
    else:
        d = A.sum(axis=1)
    mean_d = np.array([d[labels == k].mean() if (labels == k).any() else 0.0
                       for k in range(n_clusters)])
    order     = np.argsort(mean_d)          # ascending: block 0 = lowest-degree
    remap     = np.argsort(order)           # inverse permutation
    labels    = remap[labels].astype(np.int64)
    return torch.from_numpy(labels)


def _accumulate_sbm_p_kl(A_np, labels_np, K, p_counts, n_pairs):
    """Accumulate edge counts and pair counts per block pair (K x K)."""
    for k in range(K):
        mask_k = (labels_np == k)
        for l in range(k, K):
            mask_l = (labels_np == l)
            if k == l:
                A_sub  = A_np[np.ix_(mask_k, mask_k)]
                idx    = np.triu_indices(A_sub.shape[0], k=1)
                n_e    = A_sub[idx].sum()
                n_p    = mask_k.sum() * (mask_k.sum() - 1) // 2
            else:
                n_e = A_np[np.ix_(mask_k, mask_l)].sum()
                n_p = mask_k.sum() * mask_l.sum()
            p_counts[k, l] += n_e
            n_pairs[k, l]  += n_p
            if k != l:
                p_counts[l, k] += n_e
                n_pairs[l, k]  += n_p


class EmpiricalEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, degree=False, augment_features=[],
                 sbm_n_clusters=None):
        # pmf of graph size
        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)

        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()

        # empty graph table
        self.empty_graphs = {}

        # degree table
        self.degree = degree
        self.augment_features = augment_features

        self.empirical_node_feat_dist = {}
        self.sbm_p_kl = None  # populated below if sbm_n_clusters is given

        # SBM prior: accumulate block-pair statistics across all training graphs
        K = sbm_n_clusters if (sbm_n_clusters and sbm_n_clusters >= 2) else 0
        if K:
            p_counts = np.zeros((K, K), dtype=np.float64)
            n_pairs  = np.zeros((K, K), dtype=np.float64)

        for pyg_data in train_pyg_datas:
            if pyg_data.num_nodes not in self.empirical_node_feat_dist:
                self.empirical_node_feat_dist[pyg_data.num_nodes] = []
            feats = {}
            if self.degree:
                feats['degree'] = pyg.utils.degree(pyg_data.edge_index[0],num_nodes=pyg_data.num_nodes)
            # Always carry target_degree into empty-graph samples when available:
            # PA-DACA uses it for degree-aware masking even during the reverse chain.
            if hasattr(pyg_data, 'target_degree'):
                feats['target_degree'] = pyg_data.target_degree
            for feat_name in self.augment_features:
                feats[feat_name] = getattr(pyg_data, feat_name)# FEATURE_EXTRACTOR[feat_name]['func'](pyg_data)

            # SBM: compute and store block assignments for this graph
            if K and hasattr(pyg_data, 'full_edge_attr'):
                node_block = _compute_sbm_blocks(pyg_data, K)
                if node_block is not None:
                    feats['node_block'] = node_block
                    # Mutate training graph so DataLoader batches also have node_block
                    pyg_data.node_block = node_block
                    # Accumulate global block-pair edge statistics
                    ei   = pyg_data.full_edge_index
                    attr = pyg_data.full_edge_attr.float()
                    n    = pyg_data.num_nodes
                    A_np = np.zeros((n, n), dtype=np.float32)
                    A_np[ei[0].numpy(), ei[1].numpy()] = attr.numpy()
                    A_np[ei[1].numpy(), ei[0].numpy()] = attr.numpy()
                    _accumulate_sbm_p_kl(A_np, node_block.numpy(), K, p_counts, n_pairs)

            # feats['x'] = pyg_data.x
            self.empirical_node_feat_dist[pyg_data.num_nodes].append(feats)

        if K:
            self.sbm_p_kl = p_counts / np.maximum(n_pairs, 1)
            print(f'[EmpiricalEmptyGraphGenerator] SBM p_kl (K={K}):')
            print(np.round(self.sbm_p_kl, 4))


    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        ret = ret.tolist()
        xT_feats = [] 
        for n_node in ret:
            xT_feats.append(random.choice(self.empirical_node_feat_dist[n_node]))
        # xT_feats will be a list of dicts
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas

class NeuralEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, neural_attr_sampler, degree=False, device='cuda:0'):
        # now only support degree features, other features are left to future.

        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)
        # empty graph table
        self.empty_graphs = {}
        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()
        self.degree = degree
        self.neural_attr_sampler = neural_attr_sampler
        self.device = device
        
        self.node_model = NodeModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'], seq_lens=neural_attr_sampler['SEQ_LENS'])
        self.bit_model = BitModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'])
        
        self.node_model.to(self.device)
        self.bit_model.to(self.device)

        self.node_model.load_state_dict(neural_attr_sampler['modelNode'])
        self.bit_model.load_state_dict(neural_attr_sampler['modelBit'])
 
    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        if self.degree:
            x = torch.zeros(num_samples, 1, self.neural_attr_sampler['NUM_BITS'])
            g = r = ret[:, None]
            x = x.to(self.device)
            g = g.to(self.device)
            r = r.to(self.device)
            self.node_model.eval()
            self.bit_model.eval()
            with torch.no_grad():
                for i in range(self.neural_attr_sampler['SEQ_LENS']):      
                    node_hidden = self.node_model(x, g, r)[:,-1,:]
                    y = (torch.ones(num_samples, 1).long().to(self.device)*2).long()

                    for j in range(self.neural_attr_sampler['NUM_BITS']):
                        prediction = self.bit_model(y.view(-1, j+1), node_hidden.view(-1, node_hidden.shape[-1]), r[:,-1][:,None])[:,-1,:]
                        prediction = F.sigmoid(prediction)
                        index = prediction.bernoulli().long()
                        y = torch.cat([y, index],dim=-1)
                    y = y[:, 1:]
                    n_j = bin2dec(y, self.neural_attr_sampler['NUM_BITS'])-1
                    r = torch.cat([r, (r[:, -1]-n_j)[:,None]],dim=-1)
                    x = torch.cat([x, y[:,None,:]], dim=1)
                
                x = (bin2dec(x, self.neural_attr_sampler['NUM_BITS'])-1).clamp(0)[:, 1:]
            xT_feats = unpack_deg_matrix(x)
            ret = [sum(xT_feat) for xT_feat in xT_feats]
            xT_feats = [{'degree':deg_hist_to_deg_seq(xT_feat)} for xT_feat in xT_feats]
        else:
            xT_feats = [{} for _ in ret]
            ret = ret.tolist()
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas

