"""
ERGM-Anchored Discrete DDPM with Horvitz-Thompson ELBO reweighting.

Mathematical formulation: diffusion/ergm_ddpm.tex

Key improvements over PA-DACA (PartialAbsorbingDiffusion):
  1. Candidate set Q = E ∪ W(G) (ERGM-3 gradient support) — O(m·d̄) pairs
     instead of O(N²); pairs outside Q have zero triangle change statistic
     and are not needed for ERGM-3 recovery.
  2. HT reweighting: w=1 for edges, w=1/λ for wedge non-edges (λ = |W|/|non-edges|)
     — unbiased estimator of the full-graph ELBO.
  3. Dynamic triangle feature τ̂_ij(t) = (A_t²)_ij via sparse matmul — provides
     ERGM-3 signal to the denoiser at every noise level, approaching τ_ij(G) as
     chain progresses.
  4. d* via EmpiricalEmptyGraphGenerator (already implemented in data_utils.py).
  5. Corrected _beta_ij: uses sum(d_star)/N for d̄ instead of counting full_edge_attr
     (which is all-zeros during sampling from empty-graph prior).

Architecture: GraphDenoisingERGM (see layers/layers.py) — GraphDenoising backbone
with τ̂_ij log-count feature projected to dim and concatenated into MiniAt_Film edge
context (in_edge_dim = dim*3 instead of dim*2).

Theorem (ERGM-3 Recovery): Under infinite data from ERGM-3, zero ELBO, expressive
f_θ: at t=1, f_θ*(i,j; e_1) = σ(θ_1* + 2θ_2*τ_ij + θ_3*(d_i+d_j)).
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from diffusion.diffusion_base import (
    DiffusionBase, scatter,
    index_to_log_onehot, log_categorical,
    extract, log_1_min_a, log_add_exp,
    cosine_beta_schedule,
)
from diffusion.diffusion_partial_absorbing import (
    _floored_cosine_cumprod, _ABS_CLASSES, _PRED_CLASSES, _M,
)

_EPS = 1e-8
# float32 has machine epsilon ~1.19e-7, so 1.0 - 1e-8 == 1.0 in float32.
# Use a separate epsilon for probability clamping where the max side matters.
_RHO_EPS = 1e-6   # safe upper clamp: 1.0 - 1e-6 = 0.999999 in float32


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph helpers: candidate set Q and triangle feature τ̂_ij
# ─────────────────────────────────────────────────────────────────────────────

def _A2_from_edges(ei_bidi_src, ei_bidi_dst, N, device):
    """Sparse A² via sparse×dense: (N,N) dense, cost O(m·N)."""
    m2 = ei_bidi_src.shape[0]
    if m2 == 0:
        return torch.zeros(N, N, device=device)
    vals = torch.ones(m2, device=device)
    A_sp = torch.sparse_coo_tensor(
        torch.stack([ei_bidi_src, ei_bidi_dst]), vals, (N, N)
    ).coalesce()
    A_dense = A_sp.to_dense()
    return torch.sparse.mm(A_sp, A_dense)  # (N, N)


def _compute_Q_single(ei_local, e0, N, device):
    """Q = E ∪ W(G) for one graph.

    Args:
        ei_local : (2, E) upper-tri candidate pairs (LOCAL indices 0..N-1)
        e0       : (E,) binary ground-truth labels
        N        : number of nodes
        device

    Returns:
        q_idx     : (|Q|,) indices into ei_local / e0
        q_is_edge : (|Q|,) bool — True iff the pair is a true edge
    """
    pos_mask = (e0 == 1)
    e_src = ei_local[0, pos_mask]
    e_dst = ei_local[1, pos_mask]
    m = e_src.shape[0]

    if m == 0:
        # Empty graph: no wedges; Q = {} (no true edges, no wedge completions)
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, torch.zeros(0, dtype=torch.bool, device=device)

    # Build bidi edges from true edges
    ei_bidi_src = torch.cat([e_src, e_dst])
    ei_bidi_dst = torch.cat([e_dst, e_src])

    A2_true = _A2_from_edges(ei_bidi_src, ei_bidi_dst, N, device)
    tau_true = A2_true[ei_local[0], ei_local[1]]  # (E,) — from TRUE graph

    # Q = true_edges ∪ non-edges with at least one common true neighbor
    in_Q = pos_mask | (tau_true > 0.5)  # 0.5 guards float imprecision
    q_idx = in_Q.nonzero(as_tuple=True)[0]
    q_is_edge = pos_mask[q_idx]
    return q_idx, q_is_edge


def _compute_tau_single(ei_local, e_t, N, device):
    """τ̂_ij(t) = # common visible (state=1) neighbors for each candidate pair.

    Args:
        ei_local : (2, E) upper-tri candidate pairs (LOCAL indices)
        e_t      : (E,) current noisy state ∈ {0=non-edge, 1=edge, 2=M}
        N        : number of nodes
        device

    Returns:
        (E,) float — triangle counts from current noisy state
    """
    vis_mask = (e_t == 1)
    vis_src = ei_local[0, vis_mask]
    vis_dst = ei_local[1, vis_mask]
    m_vis = vis_src.shape[0]

    E = ei_local.shape[1]
    if m_vis == 0:
        return torch.zeros(E, device=device)

    A2 = _A2_from_edges(
        torch.cat([vis_src, vis_dst]),
        torch.cat([vis_dst, vis_src]),
        N, device)
    return A2[ei_local[0], ei_local[1]]  # (E,)


def _batch_node_offsets(batched_graph, device):
    """Cumulative node offsets for un-batching. Returns (B+1,) long tensor."""
    cum = torch.zeros(batched_graph.num_graphs + 1, dtype=torch.long, device=device)
    cum[1:] = batched_graph.nodes_per_graph.to(device).cumsum(0)
    return cum


def _compute_Q_batch(batched_graph, device, ht_weight_max=None):
    """Batch-level Q = E ∪ W(G) computation.

    Args:
        ht_weight_max : float or None — cap on 1/λ for wedge non-edges.
            PubMed has λ≈0.0008 → w=1250 uncapped → use ht_weight_max=50.
            None = no cap (backward compatible).

    Returns:
        q_global_idx : (|Q_batch|,) global indices into full_edge_index
        q_is_edge    : (|Q_batch|,) bool
        ht_weight    : (|Q_batch|,) HT weights (1 for edges, 1/λ for wedge non-edges)
        ht_lambda    : (B,) per-graph λ = |W \ E| / (#non-edges)
    """
    full_ei = batched_graph.full_edge_index
    e0 = batched_graph.full_edge_attr
    b = batched_graph.num_graphs
    edge_batch = batched_graph.batch[full_ei[0]]
    cum_nodes = _batch_node_offsets(batched_graph, device)

    all_q_idx, all_q_is_edge, all_ht_w = [], [], []
    ht_lambda = torch.zeros(b, device=device)

    for g in range(b):
        n_g = int(batched_graph.nodes_per_graph[g])
        node_off = int(cum_nodes[g])

        g_mask = (edge_batch == g)
        g_ei_global = full_ei[:, g_mask]
        ei_local = g_ei_global - node_off
        e0_g = e0[g_mask]
        g_edge_global = g_mask.nonzero(as_tuple=True)[0]

        q_idx_local, q_is_edge_g = _compute_Q_single(ei_local, e0_g, n_g, device)

        # Convert local → global edge index
        q_idx_global = g_edge_global[q_idx_local]

        # HT: λ = |W \ E| / (total_pairs - m)
        m_g = int(e0_g.sum())
        total_pairs_g = n_g * (n_g - 1) // 2
        n_ne_g = total_pairs_g - m_g
        n_wedge_ne = int((~q_is_edge_g).sum())

        lam = (n_wedge_ne / n_ne_g) if n_ne_g > 0 else 1.0
        lam = max(lam, 1e-6)
        ht_lambda[g] = lam

        w_wedge = 1.0 / lam
        if ht_weight_max is not None:
            w_wedge = min(w_wedge, float(ht_weight_max))
        ht_w = torch.where(
            q_is_edge_g,
            torch.ones(q_idx_local.shape[0], device=device),
            torch.full((q_idx_local.shape[0],), w_wedge, device=device),
        )

        all_q_idx.append(q_idx_global)
        all_q_is_edge.append(q_is_edge_g)
        all_ht_w.append(ht_w)

    if not all_q_idx:
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        return empty_l, torch.zeros(0, dtype=torch.bool, device=device), \
               torch.zeros(0, device=device), ht_lambda

    return (torch.cat(all_q_idx), torch.cat(all_q_is_edge),
            torch.cat(all_ht_w), ht_lambda)


def _compute_tau_batch(batched_graph, e_t_full, device):
    """τ̂_ij(t) for every pair in full_edge_index from current noisy state."""
    full_ei = batched_graph.full_edge_index
    b = batched_graph.num_graphs
    edge_batch = batched_graph.batch[full_ei[0]]
    cum_nodes = _batch_node_offsets(batched_graph, device)

    tau_full = torch.zeros(full_ei.shape[1], device=device)

    for g in range(b):
        n_g = int(batched_graph.nodes_per_graph[g])
        node_off = int(cum_nodes[g])

        g_mask = (edge_batch == g)
        ei_local = full_ei[:, g_mask] - node_off
        e_t_g = e_t_full[g_mask]
        g_idx = g_mask.nonzero(as_tuple=True)[0]

        tau_g = _compute_tau_single(ei_local, e_t_g, n_g, device)
        tau_full[g_idx] = tau_g

    return tau_full  # (E_full,)


def _compute_dynamic_Q_tau_batch(batched_graph, e_t_full, device):
    """Q_t and τ̂_ij(t) for sampling (dynamic: based on current e_t).

    Q_t = revealed pairs ∪ wedge pairs from currently visible (state=1) edges.

    Returns:
        q_global_idx : (|Q_t|,) global indices into full_edge_index
        tau_full     : (E_full,) tau values for ALL pairs (index q_global_idx to get Q_t subset)
    """
    full_ei = batched_graph.full_edge_index
    b = batched_graph.num_graphs
    edge_batch = batched_graph.batch[full_ei[0]]
    cum_nodes = _batch_node_offsets(batched_graph, device)

    tau_full = torch.zeros(full_ei.shape[1], device=device)
    all_q_idx = []

    for g in range(b):
        n_g = int(batched_graph.nodes_per_graph[g])
        node_off = int(cum_nodes[g])

        g_mask = (edge_batch == g)
        ei_local = full_ei[:, g_mask] - node_off
        e_t_g = e_t_full[g_mask]
        g_idx = g_mask.nonzero(as_tuple=True)[0]

        # τ̂_ij from visible edges (state=1)
        tau_g = _compute_tau_single(ei_local, e_t_g, n_g, device)
        tau_full[g_idx] = tau_g

        # Q_t = revealed ∪ wedge-completable from visible edges
        is_revealed_g = (e_t_g != _M)
        in_Q_g = is_revealed_g | (tau_g > 0.5)
        q_local = in_Q_g.nonzero(as_tuple=True)[0]
        all_q_idx.append(g_idx[q_local])

    q_global_idx = torch.cat(all_q_idx) if all_q_idx else \
        torch.zeros(0, dtype=torch.long, device=device)
    return q_global_idx, tau_full


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ERGMAnchoredDiffusion(DiffusionBase):
    """ERGM-Anchored Discrete DDPM with Horvitz-Thompson ELBO reweighting.

    Forward process: PA-DACA absorbing (floored cosine + β_ij degree masking).
    Candidate set  : Q = E ∪ W(G) — ERGM-3 gradient support, O(m·d̄) pairs.
    ELBO           : Σ_{(i,j)∈Q} w_ij · r_t^ij · CE(e_0^ij, p̂_θ^ij)
                     w=1 for edges, w=1/λ for wedge non-edges (HT-unbiased).
    Triangle feat  : τ̂_ij(t) fed to GraphDenoisingERGM for ERGM-3 conditioning.
    Prior L_T      : KL(q(e_T)|Chung-Lu) closed-form via α_min^{β_ij}.
    Node process   : standard binary marginal (unchanged from PA-DACA).
    """

    def __init__(self, num_node_classes, num_edge_classes,
                 initial_graph_sampler, denoise_fn,
                 timesteps=1000, loss_type='vb_kl', parametrization='x0',
                 final_prob_node=None, sample_time_method='uniform',
                 noise_schedule=cosine_beta_schedule, device='cuda',
                 alpha_min=0.022, daca_gamma=1.0, beta_max=3.0,
                 lambda_LT=1.0, kl_pos_weight=True, non_q_fraction=0.0,
                 lambda_deg=0.0, lambda_tri=0.0, ht_weight_max=None,
                 lambda_deg_node=0.0, lambda_tri_node=0.0, max_degree=100):

        # Absorbing state-space size depends on num_edge_classes:
        #   binary:  num_edge_classes=2  → {0=no-edge, 1=edge, 2=mask}  → _abs=3, _M=2
        #   typed:   num_edge_classes=K  → {0..K-1=bond types, K=mask}  → _abs=K+1
        self._abs_edge_classes  = num_edge_classes + 1
        self._mask_state        = num_edge_classes        # index of the mask token
        self._pred_edge_classes = num_edge_classes        # x0 prediction classes

        super().__init__(num_node_classes, self._abs_edge_classes,
                         initial_graph_sampler, denoise_fn, timesteps,
                         sample_time_method, device)

        self.loss_type       = loss_type
        self.parametrization = parametrization
        self.alpha_min       = alpha_min
        self.daca_gamma      = daca_gamma
        self.beta_max        = beta_max
        self.lambda_LT       = lambda_LT
        self.kl_pos_weight   = kl_pos_weight
        self.non_q_fraction  = float(non_q_fraction)  # fraction of non-Q non-edges to sample per step
        self.lambda_deg      = float(lambda_deg)
        self.lambda_tri      = float(lambda_tri)
        self.ht_weight_max   = float(ht_weight_max) if ht_weight_max is not None else None
        # Node-level auxiliary regression heads (in GraphDenoisingERGM).
        # lambda_deg_node: forces h_i to encode true degree d*_i through GNN layers.
        # lambda_tri_node: forces h_i to encode per-node triangle budget τ_node_i.
        # _log_max_deg: normalization constant shared by both heads.
        self.lambda_deg_node = float(lambda_deg_node)
        self.lambda_tri_node = float(lambda_tri_node)
        self._log_max_deg    = math.log1p(float(max_degree))
        assert 0.0 < alpha_min < 1.0

        # ── Node: standard binary marginal schedule ───────────────────────────
        n_alphas = noise_schedule(timesteps)
        n_alphas = torch.tensor(n_alphas.astype('float64'))
        log_a_n  = n_alphas.log().numpy()
        log_ca_n = np.cumsum(log_a_n)
        self.register_buffer('_log_alpha_n',  torch.tensor(log_a_n).float())
        self.register_buffer('_log_1ma_n',    log_1_min_a(torch.tensor(log_a_n)).float())
        self.register_buffer('_log_ca_n',     torch.tensor(log_ca_n).float())
        self.register_buffer('_log_1mca_n',   log_1_min_a(torch.tensor(log_ca_n)).float())
        if final_prob_node is None:
            final_prob_node = [0.5, 0.5]
        self.register_buffer('log_final_prob_node',
                             torch.tensor(final_prob_node).float().log().unsqueeze(0))

        # ── Edge: floored cosine absorbing schedule ───────────────────────────
        log_ab, log_ab_prev = _floored_cosine_cumprod(timesteps, alpha_min)
        self.register_buffer('log_cumprod_alpha_abs',
                             torch.tensor(log_ab).float())      # log(ᾱ_{t+1})
        self.register_buffer('log_cumprod_alpha_abs_prev',
                             torch.tensor(log_ab_prev).float()) # log(ᾱ_t), ᾱ_0=1

        # Importance-sampling history
        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count',   torch.zeros(timesteps))

        self._prop_diagnostics = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Schedule helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _beta_ij(self, batched_graph, edge_index):
        """β_ij = min(β_max, (d_i*·d_j* / d̄²_g)^(γ/2)).

        Uses sum(d_star)/N_g for d̄ — correct during sampling when
        full_edge_attr is all-zeros (empty-graph prior).
        """
        if (not hasattr(batched_graph, 'target_degree')
                or batched_graph.target_degree is None):
            return torch.ones(edge_index.shape[1], device=self.device)

        d_star  = batched_graph.target_degree.float().to(self.device)
        d_i     = d_star[edge_index[0]]
        d_j     = d_star[edge_index[1]]

        b       = batched_graph.num_graphs
        n_nodes = batched_graph.nodes_per_graph.float().to(self.device)
        d_sum   = scatter(d_star, batched_graph.batch.to(self.device),
                          dim=0, dim_size=b, reduce='sum')      # (B,) = 2|E|_g
        d_bar   = (d_sum / n_nodes.clamp(min=1.0)).clamp(min=1.0)

        batch_q = batched_graph.batch.to(self.device)[edge_index[0]]
        s_ij    = (d_i * d_j / d_bar[batch_q] ** 2).clamp(min=_EPS)
        return (s_ij ** (self.daca_gamma / 2.0)).clamp(max=self.beta_max)

    def _prior_rho_ij(self, batched_graph, full_ei):
        """Chung-Lu prior edge probability ρ_ij = d_i*·d_j* / 2m_g.

        Returns (E,) tensor, or None when target_degree is unavailable
        (callers must supply their own fallback in that case).
        Subclasses override this to implement ER or SBM priors.
        """
        if (not hasattr(batched_graph, 'target_degree')
                or batched_graph.target_degree is None):
            return None
        b        = batched_graph.num_graphs
        batch_fe = batched_graph.batch.to(self.device)[full_ei[0]]
        d_star   = batched_graph.target_degree.float().to(self.device)
        d_i      = d_star[full_ei[0]]
        d_j      = d_star[full_ei[1]]
        d_sum    = scatter(d_star, batched_graph.batch.to(self.device),
                           dim=0, dim_size=b, reduce='sum')
        return (d_i * d_j / d_sum[batch_fe].clamp(min=1.0)).clamp(_RHO_EPS, 1.0 - _RHO_EPS)

    def _alpha_bar_ij(self, t_edge, beta_ij):
        """ᾱ_t^{β_ij}.  (|E|,)"""
        log_ab = self.log_cumprod_alpha_abs[t_edge]
        return (log_ab * beta_ij).exp().clamp(_EPS, 1.0)

    def _alpha_bar_ij_prev(self, t_edge, beta_ij):
        """ᾱ_{t-1}^{β_ij} (=1 at t_index=0).  (|E|,)"""
        log_ab_p = self.log_cumprod_alpha_abs_prev[t_edge]
        return (log_ab_p * beta_ij).exp().clamp(_EPS, 1.0)

    def _unmask_prob(self, t_edge, beta_ij, t_prev_edge=None):
        """r_{t→t_prev}^{ij} = (ᾱ_{t_prev}^β − ᾱ_t^β) / (1 − ᾱ_t^β).

        t_prev_edge=None  → one-step-back (training ELBO).
        t_prev_edge given → multi-step skip (sampling skip-schedule).
        """
        ab_t = self._alpha_bar_ij(t_edge, beta_ij)
        if t_prev_edge is None:
            ab_tm1 = self._alpha_bar_ij_prev(t_edge, beta_ij)
        else:
            ab_tm1 = self._alpha_bar_ij(t_prev_edge, beta_ij)
        num = (ab_tm1 - ab_t).clamp(min=0.0)
        den = (1.0 - ab_t).clamp(min=_EPS)
        return (num / den).clamp(0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward process
    # ─────────────────────────────────────────────────────────────────────────

    def _q_sample_edge_abs(self, e0, t_edge, beta_ij):
        """e_t ~ q(e_t|e_0): retain e_0 w.p. ᾱ_t^β, else mask to M."""
        ab_t = self._alpha_bar_ij(t_edge, beta_ij)
        mask = (torch.rand(e0.shape[0], device=self.device) > ab_t)
        return torch.where(mask, torch.full_like(e0, self._mask_state), e0)

    def _q_sample_node(self, batched_graph, t_node):
        log_x0   = batched_graph.log_node_attr
        log_ca   = extract(self._log_ca_n,   t_node, log_x0.shape)
        log_1mca = extract(self._log_1mca_n, t_node, log_x0.shape)
        log_prob = log_add_exp(log_x0 + log_ca,
                               log_1mca + self.log_final_prob_node)
        return self.log_sample_categorical(log_prob, self.num_node_classes)

    # ─────────────────────────────────────────────────────────────────────────
    # Node posterior / KL (standard binary marginal — same as PA-DACA)
    # ─────────────────────────────────────────────────────────────────────────

    def _node_posterior(self, log_x0, log_xt, t_node, t_prev_node=None):
        tmin1 = (t_node - 1).clamp(min=0) if t_prev_node is None else t_prev_node.clamp(min=0)

        log_p1   = self.log_final_prob_node[:, 1]
        log_1mp1 = self.log_final_prob_node[:, 0]
        log_x0r  = log_x0[:, 1];  log_1mx0r = log_x0[:, 0]
        log_xtr  = log_xt[:, 1];  log_1mxtr = log_xt[:, 0]

        log_a_t    = extract(self._log_alpha_n,   t_node, log_x0r.shape)
        log_b_t    = extract(self._log_1ma_n,     t_node, log_x0r.shape)
        log_ca_tm1 = extract(self._log_ca_n,      tmin1,  log_x0r.shape)
        log_1mca_tm1 = extract(self._log_1mca_n,  tmin1,  log_x0r.shape)

        log0_gxt = log_add_exp(log_b_t + log_p1   + log_xtr,
                               log_1_min_a(log_b_t + log_p1) + log_1mxtr)
        log1_gxt = log_add_exp(log_add_exp(log_a_t, log_b_t + log_p1) + log_xtr,
                               log_b_t + log_1mp1 + log_1mxtr)
        log0_gx0 = log_add_exp(log_1mca_tm1 + log_1mp1, log_ca_tm1 + log_1mx0r)
        log1_gx0 = log_add_exp(log_ca_tm1 + log_x0r,    log_1mca_tm1 + log_p1)

        unnorm = torch.stack([log0_gxt + log0_gx0, log1_gxt + log1_gx0], dim=1)
        return unnorm - unnorm.logsumexp(1, keepdim=True)

    def _node_p_pred(self, batched_graph, t_node, log_pred_node, t_prev_node=None):
        return self._node_posterior(
            log_pred_node, batched_graph.log_node_attr_t, t_node, t_prev_node)

    def _compute_node_kl(self, batched_graph, t_node, log_pred_node, t):
        b = batched_graph.num_graphs
        log_true  = self._node_posterior(
            batched_graph.log_node_attr, batched_graph.log_node_attr_t, t_node)
        log_model = self._node_p_pred(batched_graph, t_node, log_pred_node)

        kl_n  = self.multinomial_kl(log_true, log_model)
        kl_n  = scatter(kl_n, batched_graph.batch, dim=0, dim_size=b, reduce='sum')
        nll_n = -log_categorical(batched_graph.log_node_attr, log_model)
        nll_n = scatter(nll_n, batched_graph.batch, dim=0, dim_size=b, reduce='sum')

        mask_t0 = (t == 0).float()
        return mask_t0 * nll_n + (1.0 - mask_t0) * kl_n

    # ─────────────────────────────────────────────────────────────────────────
    # Prior KL L_T (Chung-Lu, corrected denominator)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_LT(self, batched_graph):
        """L_T = Σ_{ij∈full} α_min^{β_ij} · CE(e_0^ij, ρ_ij).

        ρ_ij = d_i*·d_j* / (2|E|_g) = Chung-Lu pair probability.
        Denominator uses sum(d_star) = 2|E|_g, correct also during sampling.
        """
        b       = batched_graph.num_graphs
        full_ei = batched_graph.full_edge_index
        batch_fe = batched_graph.batch.to(self.device)[full_ei[0]]

        with torch.no_grad():
            beta_full      = self._beta_ij(batched_graph, full_ei)
            alpha_min_beta = self.alpha_min ** beta_full          # (E,)

            rho_ij = self._prior_rho_ij(batched_graph, full_ei)
            if rho_ij is None:
                n_pos   = scatter(batched_graph.full_edge_attr.float(),
                                  batch_fe, dim=0, dim_size=b, reduce='sum')
                total_c = getattr(batched_graph, 'total_cands_per_graph',
                                  batched_graph.edges_per_graph).float().to(self.device)
                rho_g   = (n_pos / total_c.clamp(min=1.0)).clamp(_RHO_EPS, 1.0 - _RHO_EPS)
                rho_ij  = rho_g[batch_fe]

        e0 = batched_graph.full_edge_attr.float()
        ce_prior = -(e0 * torch.log(rho_ij) + (1.0 - e0) * torch.log(1.0 - rho_ij))
        lt_pair  = alpha_min_beta * ce_prior
        return scatter(lt_pair, batch_fe, dim=0, dim_size=b, reduce='sum')

    # ─────────────────────────────────────────────────────────────────────────
    # Time sampling
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self._sample_time(b, device, 'uniform')
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt = torch.nan_to_num(Lt_sqrt, nan=0.0001, posinf=1.0, neginf=0.0001)
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            if torch.isnan(pt_all).any() or pt_all.sum() <= 0:
                return self._sample_time(b, device, 'uniform')
            t  = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(0, t)
            return t, pt
        else:  # uniform
            length = self.Lt_count.shape[0]
            t  = torch.randint(0, length, (b,), device=device).long()
            pt = torch.ones_like(t).float() / length
            return t, pt

    def _calc_num_entries(self, batched_graph):
        return (batched_graph.full_edge_attr.shape[0]
                + batched_graph.node_attr.shape[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Training loss
    # ─────────────────────────────────────────────────────────────────────────

    def _train_loss(self, batched_graph):
        return self._loss_impl(batched_graph, is_train=True)

    def _eval_loss(self, batched_graph):
        return self._loss_impl(batched_graph, is_train=False)

    def _loss_impl(self, batched_graph, is_train=True):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph)

        # ── x_0 one-hots ────────────────────────────────────────────────────
        batched_graph.log_node_attr = index_to_log_onehot(
            batched_graph.node_attr, self.num_node_classes)

        # ── Sample t ────────────────────────────────────────────────────────
        t, pt = self._sample_time(b, self.device, self.sample_time_method)
        t_node      = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge_full = t.repeat_interleave(batched_graph.edges_per_graph)

        # ── Forward process (edges: absorbing 3-class; nodes: binary marginal) ─
        beta_full  = self._beta_ij(batched_graph, batched_graph.full_edge_index)
        e0_full    = batched_graph.full_edge_attr                  # (E,) ∈{0,1}
        e_t_full   = self._q_sample_edge_abs(e0_full, t_edge_full, beta_full)
        batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_t_full, self._abs_edge_classes)
        batched_graph.log_node_attr_t = self._q_sample_node(batched_graph, t_node)

        # ── Build ERGM-3 candidate set Q = E ∪ W(G) with HT weights ─────────
        # Q computation uses the TRUE graph e0; τ̂ uses the NOISY state e_t.
        q_global_idx, q_is_edge, ht_weight, ht_lambda = \
            _compute_Q_batch(batched_graph, self.device, self.ht_weight_max)

        if q_global_idx.shape[0] == 0:
            # Degenerate: completely disconnected batch — skip step
            return torch.zeros(b, device=self.device)

        # ── Dynamic triangle feature τ̂_ij(t) ─────────────────────────────────
        # NOTE: τ̂ uses the NOISY graph e_t, so S pairs can have τ̂>0 via noisy
        # common neighbours — this is correct and mirrors sampling-time behaviour.
        tau_full = _compute_tau_batch(batched_graph, e_t_full, self.device)
        tau_q    = tau_full[q_global_idx]                         # (|Q|,)

        # ── SparseDiff non-Q extension ────────────────────────────────────────
        # Q covers only ~1-2% of pairs for sparse citation graphs.  The denoiser
        # never sees the other 98% during training, yet must score ALL pairs at
        # sampling time — causing a systematic coverage gap.  We fix this by
        # sampling a fraction of non-Q non-edges (SparseDiff HT estimator) with
        # importance weight 1/λ_2 so the ELBO remains an unbiased sum over all
        # candidate pairs.  True edges are always in Q (E ⊆ Q), so S ⊆ non-edges.
        s_idx  = torch.zeros(0, dtype=torch.long, device=self.device)
        s_ht_w = torch.zeros(0, device=self.device)
        if is_train and self.non_q_fraction > 0.0:
            in_Q_full  = torch.zeros(e0_full.shape[0], dtype=torch.bool, device=self.device)
            in_Q_full[q_global_idx] = True
            # S ⊆ non-Q non-edges (τ^{true}=0 by definition; τ̂ may be non-zero)
            non_q_ne_idx = torch.where(~in_Q_full & (e0_full == 0))[0]
            n_non_q_ne   = non_q_ne_idx.shape[0]
            if n_non_q_ne > 0:
                n_s    = min(int(self.non_q_fraction * n_non_q_ne), n_non_q_ne)
                perm   = torch.randperm(n_non_q_ne, device=self.device)[:n_s]
                s_idx  = non_q_ne_idx[perm]
                lam_2  = n_s / n_non_q_ne           # sampling probability
                s_ht_w = torch.full((n_s,), 1.0 / lam_2, device=self.device)

        # ── Gather Q∪S quantities for single denoiser call ───────────────────
        has_s = (s_idx.shape[0] > 0)
        nQ    = q_global_idx.shape[0]

        q_ei       = batched_graph.full_edge_index[:, q_global_idx]  # (2, |Q|)
        t_edge_q   = t_edge_full[q_global_idx]
        beta_q     = beta_full[q_global_idx]
        e0_q       = e0_full[q_global_idx]
        e_t_q      = e_t_full[q_global_idx]
        batch_q    = batched_graph.batch[q_ei[0]]

        if has_s:
            comb_idx   = torch.cat([q_global_idx, s_idx])
            comb_ei    = batched_graph.full_edge_index[:, comb_idx]
            comb_tau   = tau_full[comb_idx]
            comb_t_e   = t_edge_full[comb_idx]
            comb_beta  = beta_full[comb_idx]
            comb_e0    = e0_full[comb_idx]
            comb_e_t   = e_t_full[comb_idx]
            comb_ht_w  = torch.cat([ht_weight, s_ht_w])
            comb_batch = batched_graph.batch[comb_ei[0]]
        else:
            comb_idx   = q_global_idx
            comb_ei    = q_ei
            comb_tau   = tau_q
            comb_t_e   = t_edge_q
            comb_beta  = beta_q
            comb_e0    = e0_q
            comb_e_t   = e_t_q
            comb_ht_w  = ht_weight
            comb_batch = batch_q

        # ── Pre-compute node-head side-channels before denoiser call ─────────
        # tau_node_true (per-node true triangle budget) must be set on
        # batched_graph BEFORE the denoiser forward so GraphDenoisingERGM can
        # produce deg_pred / tri_pred inside its forward pass.
        # We need tau_true_q for this, but the full triangle block runs after
        # the denoiser.  Compute it here when any node-level loss is active,
        # and store; the post-denoiser block will reuse it for L_tri if needed.
        _tau_true_q_precomputed = None
        if self.lambda_deg_node > 0.0 or self.lambda_tri_node > 0.0:
            _tau_true_full_pre    = _compute_tau_batch(batched_graph, e0_full, self.device)
            _tau_true_q_precomputed = _tau_true_full_pre[q_global_idx]   # (|Q|,)
            _tau_node_true_pre = (
                scatter(_tau_true_q_precomputed, q_ei[0], dim=0,
                        dim_size=batched_graph.num_nodes, reduce='sum')
              + scatter(_tau_true_q_precomputed, q_ei[1], dim=0,
                        dim_size=batched_graph.num_nodes, reduce='sum'))
            batched_graph.tau_node_true = _tau_node_true_pre  # side-channel to forward

        # ── Run denoiser once over Q∪S ────────────────────────────────────────
        batched_graph.query_edge_index = comb_ei
        batched_graph.tau_feat         = comb_tau
        out_node, out_comb = self._denoise_fn(batched_graph, t_node, comb_t_e)
        batched_graph.query_edge_index = None
        batched_graph.tau_feat         = None

        # out_node is the pass-through log_node_attr (all-zero class for our datasets).
        # Node KL = KL(truth || truth) = 0 by construction — not used in loss.
        # out_node is still returned for sampling compatibility (_ergm_p_sample line ~849).
        log_pred_comb   = F.log_softmax(out_comb,          dim=-1)  # (|Q|+|S|, 2)
        log_pred_edge_q = log_pred_comb[:nQ]                         # (|Q|, 2) — Q only

        # ── HT-reweighted absorbing ELBO (Q∪S combined) ──────────────────────
        is_masked_comb = (comb_e_t == self._mask_state).float()
        r_t_comb       = self._unmask_prob(comb_t_e, comb_beta)

        ce_comb = -(comb_e0.float() * log_pred_comb[:, 1]
                    + (1.0 - comb_e0.float()) * log_pred_comb[:, 0])  # (|Q|+|S|,)

        elbo_pair = r_t_comb * ce_comb * is_masked_comb * comb_ht_w   # (|Q|+|S|,)

        # ── P2 fix: w+^M(t) masked-pool reweighting ──────────────────────────
        if self.kl_pos_weight:
            batch_fe  = batched_graph.batch[batched_graph.full_edge_index[0]]
            is_m_full = (e_t_full == self._mask_state).float()
            e0_ff     = e0_full.float()
            n_pos_m   = scatter(e0_ff * is_m_full, batch_fe,
                                dim=0, dim_size=b, reduce='sum')
            n_neg_m   = scatter((1.0 - e0_ff) * is_m_full, batch_fe,
                                dim=0, dim_size=b, reduce='sum')
            w_plus    = n_neg_m / n_pos_m.clamp(min=1.0)
            # S pairs are all non-edges (comb_e0=0) → kl_pos_w=1 for them
            kl_pos_w  = w_plus[comb_batch] * comb_e0.float() + (1.0 - comb_e0.float())
        else:
            kl_pos_w  = torch.ones(comb_e0.shape[0], device=self.device)

        kl_edge  = scatter(elbo_pair * kl_pos_w,
                           comb_batch, dim=0, dim_size=b, reduce='sum')

        # ── Reconstruction NLL at t=0  (L_0) — Q∪S consistent with KL ───────
        nll_edge = scatter(ce_comb * comb_ht_w * kl_pos_w,
                           comb_batch, dim=0, dim_size=b, reduce='sum')

        # ── t=0 switching ─────────────────────────────────────────────────────
        # Node KL removed: node_attr = zeros for all datasets → node_loss ≡ 0.
        mask_t0   = (t == 0).float()
        loss_step = mask_t0 * nll_edge + (1.0 - mask_t0) * kl_edge

        # ── Prior KL L_T ──────────────────────────────────────────────────────
        lt = self._compute_LT(batched_graph)

        # ── Noise-level weight α̅_t for auxiliary losses ───────────────────────
        # Mathematically: x̂₀ predictions are reliable when the chain is near
        # clean (t≈0, α̅_t≈1) and unreliable at high noise (t≈T, α̅_t≈α_min).
        # Uniform-β approximation: α̅_t ≈ α_min^(t/T), per-graph (B,).
        # This suppresses auxiliary constraints at high t, preventing gradient
        # conflict with the ELBO which dominates where predictions are poor.
        T = float(self.log_cumprod_alpha_abs.shape[0])
        with torch.no_grad():
            ab_t = self.alpha_min ** (t.float() / T)        # (B,) ∈ [α_min, 1]

        # ── Auxiliary: degree penalty L_deg (directly differentiable, no GAN) ─
        # E[deg_i] = Σ_{(i,j)∈Q} P̂(e_ij=1).  Q covers all edges and their
        # wedge pairs; non-Q non-edges have τ=0 and e0=0, so P̂≈0 there.
        # Using Q only (no HT amplification) gives lower-variance gradient.
        # Weighted by α̅_t: at high noise the model's x̂₀ cannot satisfy the
        # degree constraint (Q too sparse vs. full degree), causing gradient
        # conflict with ELBO.  α̅_t suppresses L_deg at those timesteps.
        l_deg = None
        if self.lambda_deg > 0.0:
            pred_p1_q = log_pred_comb[:nQ].exp()[:, 1]          # (|Q|,)
            exp_deg   = (
                scatter(pred_p1_q, q_ei[0], dim=0,
                        dim_size=batched_graph.num_nodes, reduce='sum')
                + scatter(pred_p1_q, q_ei[1], dim=0,
                          dim_size=batched_graph.num_nodes, reduce='sum')
            )
            if (hasattr(batched_graph, 'target_degree')
                    and batched_graph.target_degree is not None):
                d_star_loss = batched_graph.target_degree.float().to(self.device)
            else:
                d_star_loss = (
                    scatter(e0_full.float(), batched_graph.full_edge_index[0],
                            dim=0, dim_size=batched_graph.num_nodes, reduce='sum')
                    + scatter(e0_full.float(), batched_graph.full_edge_index[1],
                              dim=0, dim_size=batched_graph.num_nodes, reduce='sum')
                )
            sq_err  = scatter((exp_deg - d_star_loss.detach()).pow(2),
                              batched_graph.batch, dim=0, dim_size=b, reduce='sum')
            l_deg   = sq_err / batched_graph.nodes_per_graph.float().to(self.device)
            loss_step = loss_step + self.lambda_deg * (l_deg * ab_t)  # α̅_t weighted

        # ── True triangle budget: shared computation for L_tri and L_tri_node ──
        # Computed from clean x_0 so the target is the TRUE triangle structure.
        # If node-level losses pre-computed tau_true_q above, reuse it to avoid
        # a second _compute_tau_batch call.  If only L_tri is active (no node
        # losses), compute it fresh here.
        tau_true_q    = None
        tau_node_true = None
        T_hat = T_true = None   # kept as locals so diagnostics can access them
        if self.lambda_tri > 0.0 or self.lambda_tri_node > 0.0:
            if _tau_true_q_precomputed is not None:
                # Already computed before denoiser call — reuse, no extra cost.
                tau_true_q    = _tau_true_q_precomputed
                tau_node_true = batched_graph.tau_node_true  # already set
            else:
                # Node-level losses inactive: compute fresh for L_tri only.
                tau_true_full = _compute_tau_batch(batched_graph, e0_full, self.device)
                tau_true_q    = tau_true_full[q_global_idx]              # (|Q|,)
                tau_node_true = (
                    scatter(tau_true_q, q_ei[0], dim=0,
                            dim_size=batched_graph.num_nodes, reduce='sum')
                  + scatter(tau_true_q, q_ei[1], dim=0,
                            dim_size=batched_graph.num_nodes, reduce='sum'))

        # ── Auxiliary: triangle penalty L_tri ─────────────────────────────────
        # T̂ = Σ_{(i,j)∈Q} P̂_ij · τ_ij(e_0)/3.  Normalized by T* so the loss is
        # scale-invariant (OTC: T*=33K, PubMed: T*>>33K).
        # Now per-graph (B,), weighted by α̅_t for the same reason as L_deg:
        # at high noise T_hat converges to the prior expectation, not T*.
        l_tri = None
        if self.lambda_tri > 0.0:
            pred_p1_q = log_pred_comb[:nQ].exp()[:, 1]          # (|Q|,) — reuse
            T_hat  = scatter(pred_p1_q * tau_true_q / 3.0,
                             batch_q, dim=0, dim_size=b, reduce='sum')
            T_true = scatter(e0_q.float() * tau_true_q / 3.0,
                             batch_q, dim=0, dim_size=b, reduce='sum')
            denom  = T_true.detach().clamp(min=1.0)
            l_tri  = ((T_hat - T_true.detach()) / denom).pow(2)  # (B,), was scalar
            loss_step = loss_step + self.lambda_tri * (l_tri * ab_t).mean()

        # ── Node-level degree regression head L_deg_node ──────────────────────
        # MSE( deg_head(h_i), log(1+d*_i)/log(1+d_max) ).
        # Gradient flows through ALL GNN layers, giving h_i a direct signal to
        # encode degree alongside the edge prediction objective.
        # Weighted by α̅_t: at high t, h_i is dominated by the initial embedding
        # and GNN signal is weak — the head adds little information.
        l_deg_node = None
        if self.lambda_deg_node > 0.0 and getattr(batched_graph, 'deg_pred', None) is not None:
            d_star_n    = batched_graph.target_degree.float().to(self.device)
            deg_target  = torch.log1p(d_star_n) / self._log_max_deg  # (N,) ∈ [0,1]
            ab_n        = ab_t.repeat_interleave(
                batched_graph.nodes_per_graph.to(self.device))         # (N,)
            l_deg_node  = scatter(
                (batched_graph.deg_pred - deg_target.detach()).pow(2) * ab_n,
                batched_graph.batch, dim=0, dim_size=b, reduce='sum'   # (B,)
            ) / batched_graph.nodes_per_graph.float().to(self.device)
            loss_step   = loss_step + self.lambda_deg_node * l_deg_node

        # ── Node-level triangle budget regression head L_tri_node ─────────────
        # MSE( tri_head(h_i), log(1+τ_node_i)/log(1+d_max²) ).
        # τ_node_i = Σ_{j:(i,j)∈Q} τ_ij(e_0): per-node true triangle budget.
        # Gradient forces h_i to encode local triangle density, making
        # cat[h_i, h_j] in the edge decoder triangle-aware at both endpoints.
        l_tri_node = None
        if self.lambda_tri_node > 0.0 and getattr(batched_graph, 'tri_pred', None) is not None:
            # log(1+d_max²) is a conservative upper bound on log(1+τ_node_max).
            log_max_tau = 2.0 * self._log_max_deg
            tri_target  = (torch.log1p(tau_node_true.float())
                           / log_max_tau)                    # (N,) ∈ [0,1]
            ab_n        = ab_t.repeat_interleave(
                batched_graph.nodes_per_graph.to(self.device))         # (N,)
            l_tri_node  = scatter(
                (batched_graph.tri_pred - tri_target.detach()).pow(2) * ab_n,
                batched_graph.batch, dim=0, dim_size=b, reduce='sum'   # (B,)
            ) / batched_graph.nodes_per_graph.float().to(self.device)
            loss_step   = loss_step + self.lambda_tri_node * l_tri_node

        # ── Cleanup side-channel fields ────────────────────────────────────────
        # These are set before the denoiser forward pass so the model can compute
        # deg_pred / tri_pred.  Must be cleared so they don't persist into
        # sampling steps or the next training iteration.
        batched_graph.tau_node_true = None
        batched_graph.deg_pred      = None
        batched_graph.tri_pred      = None

        # ── Importance-sampling history update ────────────────────────────────
        if is_train:
            Lt2      = loss_step.pow(2)
            Lt2_prev = self.Lt_history.gather(0, t)
            self.Lt_history.scatter_(0, t, (0.1 * Lt2 + 0.9 * Lt2_prev).detach())
            self.Lt_count.scatter_add_(0, t, torch.ones_like(Lt2))

            with torch.no_grad():
                pred_p1   = log_pred_edge_q.exp()[:, 1]
                xt_dens   = (e_t_full == 1).float().mean().item()
                xt_mask   = (e_t_full == self._mask_state).float().mean().item()
                q_frac    = nQ / max(e0_full.shape[0], 1)
                s_frac    = s_idx.shape[0] / max(e0_full.shape[0], 1)
                diag = {
                    'p1/pred_density_mean':  pred_p1.mean().item(),
                    'p2/w_plus_masked_mean': (w_plus.mean().item()
                                              if self.kl_pos_weight else 1.0),
                    'p3/xt_edge_density':    xt_dens,
                    'p3/xt_mask_fraction':   xt_mask,
                    'ergm/ht_lambda_mean':   float(ht_lambda.mean()),
                    'ergm/ht_w_max':         float(ht_weight.max()) if ht_weight.numel() > 0 else 0.0,
                    'ergm/Q_frac':           q_frac,
                    'ergm/S_frac':           s_frac,
                    'ergm/tau_q_mean':       tau_q.mean().item(),
                    'aux/ab_t_mean':         ab_t.mean().item(),
                }
                if self.lambda_deg > 0.0 and l_deg is not None:
                    pq = log_pred_comb[:nQ].exp()[:, 1].detach()
                    ed = (scatter(pq, q_ei[0], dim=0,
                                  dim_size=batched_graph.num_nodes, reduce='sum')
                          + scatter(pq, q_ei[1], dim=0,
                                    dim_size=batched_graph.num_nodes, reduce='sum'))
                    ds = (batched_graph.target_degree.float().detach()
                          if hasattr(batched_graph, 'target_degree')
                             and batched_graph.target_degree is not None
                          else ed.detach())
                    diag['aux/exp_deg_mean']   = ed.mean().item()
                    diag['aux/d_star_mean']    = ds.mean().item()
                    diag['aux/l_deg_mean']     = l_deg.mean().item()
                if self.lambda_tri > 0.0 and T_hat is not None:
                    diag['aux/T_hat_mean']     = T_hat.detach().mean().item()
                    diag['aux/T_true_mean']    = T_true.detach().mean().item()
                    diag['aux/l_tri_mean']     = l_tri.mean().item()
                if self.lambda_deg_node > 0.0 and l_deg_node is not None:
                    diag['aux/l_deg_node_mean'] = l_deg_node.mean().item()
                if self.lambda_tri_node > 0.0 and l_tri_node is not None:
                    diag['aux/l_tri_node_mean'] = l_tri_node.mean().item()
                self._prop_diagnostics = diag

        vb_loss = loss_step / pt + self.lambda_LT * lt
        return -vb_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_data_for_sampling(self, batched_graph):
        """Initialize x_T from partial-absorbing Chung-Lu prior.

        p(e_ij,T = M)  = 1 − α_min^{β_ij}
        p(e_ij,T = k)  = α_min^{β_ij} · Bernoulli(ρ_ij)   k∈{0,1}
        where ρ_ij = d_i*·d_j* / sum(d_star)_g  (Chung-Lu).
        """
        batched_graph.log_node_attr = index_to_log_onehot(
            batched_graph.node_attr, self.num_node_classes)

        log_p_node = self.log_final_prob_node.expand_as(batched_graph.log_node_attr)
        batched_graph.log_node_attr_t = self.log_sample_categorical(
            log_p_node, self.num_node_classes)

        b       = batched_graph.num_graphs
        full_ei = batched_graph.full_edge_index
        batch_fe = batched_graph.batch.to(self.device)[full_ei[0]]
        E       = full_ei.shape[1]

        beta_full = self._beta_ij(batched_graph, full_ei)
        ab_T      = (self.alpha_min ** beta_full).clamp(_EPS, 1.0)

        rho_ij = self._prior_rho_ij(batched_graph, full_ei)
        if rho_ij is None:
            rho_ij = torch.full((E,), 0.01, device=self.device)

        u_vis  = torch.rand(E, device=self.device)
        u_edge = torch.rand(E, device=self.device)
        is_vis = (u_vis < ab_T)
        e_bern = (u_edge < rho_ij).long()
        e_T    = torch.where(is_vis, e_bern, torch.full_like(e_bern, self._mask_state))

        batched_graph.log_full_edge_attr = index_to_log_onehot(
            batched_graph.full_edge_attr, self._pred_edge_classes)
        batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_T, self._abs_edge_classes)
        return batched_graph

    def _ergm_p_sample(self, batched_graph, t_node, t_edge,
                        t_prev_node=None, t_prev_edge=None):
        """One absorbing reverse step over the full candidate set.

        Q during sampling = all N(N-1)/2 pairs (full dense set), NOT the
        dynamic E ∪ W(G_t) used in training.  Reason: the dynamic Q starts
        tiny at t=T (only ~α_min pairs revealed from the Chung-Lu prior) and
        hub pairs — which have β_ij=β_max and are almost always masked — never
        enter Q_t until a triangle forms, creating a cascade-stall cold-start.
        Training uses the restricted Q only for ELBO efficiency; sampling must
        score all pairs so that hub degree structure is recovered.
        τ̂_ij(t) is still computed from currently-visible edges only.
        """
        with torch.no_grad():
            full_ei  = batched_graph.full_edge_index
            e_t      = batched_graph.log_full_edge_attr_t.argmax(-1)  # (E,) ∈{0,1,2}
            E        = e_t.shape[0]

            # τ̂ from visible edges — provides triangle context as chain progresses
            tau_full = _compute_tau_batch(batched_graph, e_t, self.device)

            # Full candidate set: all N(N-1)/2 pairs
            q_dyn_idx = torch.arange(E, device=self.device)

            q_ei       = full_ei[:, q_dyn_idx]
            t_edge_q   = t_edge[q_dyn_idx]
            beta_full  = self._beta_ij(batched_graph, full_ei)
            beta_q     = beta_full[q_dyn_idx]
            e_t_q      = e_t[q_dyn_idx]
            tau_q      = tau_full[q_dyn_idx]

            # Run denoiser over Q_t
            batched_graph.query_edge_index = q_ei
            batched_graph.tau_feat         = tau_q
            out_node, out_edge_q = self._denoise_fn(batched_graph, t_node, t_edge_q)
            batched_graph.query_edge_index = None
            batched_graph.tau_feat         = None

            log_pred_node   = F.log_softmax(out_node,   dim=-1)
            log_pred_edge_q = F.log_softmax(out_edge_q, dim=-1)  # (|Q_t|, 2)

            # Unmask probability: final step (t_prev=None) reveals all; else skip-schedule
            if t_prev_edge is None:
                r_t_q = torch.ones(q_dyn_idx.shape[0], device=self.device)
            else:
                t_prev_q = t_prev_edge[q_dyn_idx]
                ab_tm1   = self._alpha_bar_ij(t_prev_q, beta_q)
                ab_t     = self._alpha_bar_ij(t_edge_q, beta_q)
                r_t_q    = ((ab_tm1 - ab_t).clamp(min=0.0)
                            / (1.0 - ab_t).clamp(min=_EPS)).clamp(0.0, 1.0)

            is_masked_q = (e_t_q == self._mask_state)
            u           = torch.rand(q_dyn_idx.shape[0], device=self.device)
            unmask_q    = is_masked_q & (u < r_t_q)

            # Sample e_0 from model (Gumbel-argmax over 2-class prediction)
            e_hat_q = self.log_sample_categorical(log_pred_edge_q, self._pred_edge_classes).argmax(-1)

            # New state for Q_t pairs
            e_tm1_q = torch.where(
                is_masked_q,
                torch.where(unmask_q, e_hat_q, torch.full_like(e_t_q, self._mask_state)),
                e_t_q)

            # Build full new state: copy old, update Q_t
            e_tm1_full = e_t.clone()
            e_tm1_full[q_dyn_idx] = e_tm1_q
            log_edge_tm1 = index_to_log_onehot(e_tm1_full, self._abs_edge_classes)

            # Node reverse step
            log_node_tm1 = self.log_sample_categorical(
                self._node_p_pred(batched_graph, t_node, log_pred_node, t_prev_node),
                self.num_node_classes)

        return log_node_tm1, log_edge_tm1

    def sample(self, num_samples, inference_steps=None):
        """Generate graphs via absorbing reverse chain over full candidate set.

        Algorithm:
        1. Sample (N, d*) from empirical distribution (EmpiricalEmptyGraphGenerator)
        2. Initialize e_T from Chung-Lu partial-absorbing prior p(e_T)
        3. For t = T, ..., 1:
           a. τ̂_ij(t) from A_t² (currently-visible edges only)
           b. Score ALL N(N-1)/2 pairs — not a dynamic subset — so hub pairs
              get degree/triangle signal from step 1 regardless of cascade state
           c. For each masked pair: sample e_{t-1,ij} via absorbing posterior
        4. Output: binary edge labels (state=1 → edge, states 0/M → non-edge)
        """
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        step_seq = self._build_discrete_skip_schedule(inference_steps)
        n_steps  = len(step_seq)

        print()
        for step_i, (t, t_prev) in enumerate(step_seq):
            print(f'ERGM sample step {step_i+1:4d}/{n_steps}  (t={t})', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)

            if t_prev >= 0:
                tp_node = torch.full((num_nodes,), t_prev, device=self.device, dtype=torch.long)
                tp_edge = torch.full((num_edges,), t_prev, device=self.device, dtype=torch.long)
            else:
                tp_node = tp_edge = None

            log_node_tm1, log_edge_tm1 = self._ergm_p_sample(
                batched_graph, t_node, t_edge,
                t_prev_node=tp_node, t_prev_edge=tp_edge)
            batched_graph.log_node_attr_t      = log_node_tm1
            batched_graph.log_full_edge_attr_t = log_edge_tm1

        print()

        # Convert 3-class → binary: state=1 → edge, states {0, M} → non-edge
        e_state  = batched_graph.log_full_edge_attr_t.argmax(-1)
        edge_attr = (e_state == 1).long()

        is_edge_idx = edge_attr.nonzero(as_tuple=True)[0]
        edge_index  = batched_graph.full_edge_index[:, is_edge_idx]
        batched_graph.edge_index = edge_index
        batched_graph.edge_attr  = edge_attr[is_edge_idx]
        batched_graph.node_attr  = batched_graph.log_node_attr_t.argmax(-1)

        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice,
                             dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1, 0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index']   = batched_graph._inc_dict['full_edge_index']

        return batched_graph
