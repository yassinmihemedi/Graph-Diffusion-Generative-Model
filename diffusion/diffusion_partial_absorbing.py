"""
PA-DACA: Partial-Absorbing Degree-Aware Causal Absorbing Diffusion

Edge state space: {0=non-edge, 1=edge, 2=M} -- 3 classes
  Forward:  q(e_t|e_0,d*) = ᾱ_t^β · δ(e_t,e_0) + (1-ᾱ_t^β) · δ(e_t,M)
  Schedule: ᾱ_t = α_min + (1-α_min)·f_cos(t/T)   [floored cosine, never→0]
  β_ij:     min(β_max, (d_i*·d_j*/d̄²)^(γ/2))     [degree-aware masking speed]
  Posterior (masked): r_t·δ(e_0) + (1-r_t)·δ(M),  r_t=(ᾱ_{t-1}^β-ᾱ_t^β)/(1-ᾱ_t^β)
  ELBO:     r_t · CE(e_0, p̂) for masked pairs  [KL exact for absorbing process]
  L_T:      Σ_{ij} α_min^{β_ij} · CE(e_0, ρ_ij)   [closed-form prior KL, ≠0]

Node state space: {0,1} -- standard binary marginal, unchanged.
"""
import torch
import torch.nn.functional as F
import numpy as np

from diffusion.diffusion_base import (
    DiffusionBase, scatter,
    index_to_log_onehot, log_categorical,
    extract, log_1_min_a, log_add_exp,
    cosine_beta_schedule,
)

_M = 2             # mask-state index  {0, 1, M=2}
_ABS_CLASSES = 3   # edge state-space size
_PRED_CLASSES = 2  # x_0 prediction is always binary


# ─────────────────────────────────────────────────────────────────────────────
# Schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _floored_cosine_cumprod(timesteps, alpha_min, s=0.008):
    """Returns (log_ᾱ_t, log_ᾱ_{t-1}) arrays, each shape (T,).

    Convention (matches DiffusionBase / BinomialDiffusionVanilla):
      log_ab[t_index]      = log(ᾱ_{t_index+1})  where ᾱ_1 ≈ 1, ᾱ_T = alpha_min
      log_ab_prev[t_index] = log(ᾱ_{t_index})     where ᾱ_0 = 1.0

    ᾱ_t = alpha_min + (1-alpha_min) · f_cos(t/T),   t = 0..T
    """
    T = timesteps
    steps = T + 1
    x = np.linspace(0, steps, steps)
    f = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    f = f / f[0]                                        # f[0]=1, f[T]≈0
    f_floor = alpha_min + (1.0 - alpha_min) * f        # (T+1,): 1→alpha_min
    # f_floor[0]=1.0, f_floor[T]=alpha_min
    log_ab      = np.log(f_floor[1:]).astype('float64')  # (T,) log(ᾱ_{1..T})
    log_ab_prev = np.log(f_floor[:-1]).astype('float64') # (T,) log(ᾱ_{0..T-1})
    # log_ab_prev[0] = log(f_floor[0]) = log(1) = 0  ← ᾱ_0 = 1
    return log_ab, log_ab_prev


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PartialAbsorbingDiffusion(DiffusionBase):
    """PA-DACA: 3-state absorbing edge diffusion + standard binary node diffusion."""

    def __init__(self, num_node_classes, num_edge_classes,
                 initial_graph_sampler, denoise_fn,
                 timesteps=1000, loss_type='vb_kl', parametrization='x0',
                 final_prob_node=None, sample_time_method='uniform',
                 noise_schedule=cosine_beta_schedule, device='cuda',
                 alpha_min=0.022, daca_gamma=1.0, beta_max=3.0,
                 lambda_LT=1.0, edge_fraction=1.0, kl_pos_weight=True):

        # Tell DiffusionBase the edge state space is 3-class (absorbing).
        # The denoiser outputs _PRED_CLASSES=2; we handle the mismatch explicitly.
        super().__init__(num_node_classes, _ABS_CLASSES,
                         initial_graph_sampler, denoise_fn, timesteps,
                         sample_time_method, device)

        self.loss_type       = loss_type
        self.parametrization = parametrization
        self.alpha_min       = alpha_min
        self.daca_gamma      = daca_gamma
        self.beta_max        = beta_max
        self.lambda_LT       = lambda_LT
        self.edge_fraction   = edge_fraction
        self.kl_pos_weight   = kl_pos_weight
        assert 0.0 < edge_fraction <= 1.0
        assert 0.0 < alpha_min < 1.0

        # ── Node: standard binary marginal schedule ───────────────────────────
        n_alphas = noise_schedule(timesteps)
        n_alphas = torch.tensor(n_alphas.astype('float64'))
        log_a_n  = n_alphas.log().numpy()
        log_ca_n = np.cumsum(log_a_n)
        self.register_buffer('_log_alpha_n',      torch.tensor(log_a_n).float())
        self.register_buffer('_log_1ma_n',        log_1_min_a(torch.tensor(log_a_n)).float())
        self.register_buffer('_log_ca_n',         torch.tensor(log_ca_n).float())
        self.register_buffer('_log_1mca_n',       log_1_min_a(torch.tensor(log_ca_n)).float())
        if final_prob_node is None:
            final_prob_node = [0.5, 0.5]
        self.register_buffer('log_final_prob_node',
                             torch.tensor(final_prob_node).float().log().unsqueeze(0))

        # ── Edge: floored cosine absorbing schedule ───────────────────────────
        log_ab, log_ab_prev = _floored_cosine_cumprod(timesteps, alpha_min)
        # log_cumprod_alpha_abs[t]      = log(ᾱ_{t+1})  (t_index t → timestep t+1)
        # log_cumprod_alpha_abs_prev[t] = log(ᾱ_t)      (t_index t → timestep t, ᾱ_0=1)
        self.register_buffer('log_cumprod_alpha_abs',      torch.tensor(log_ab).float())
        self.register_buffer('log_cumprod_alpha_abs_prev', torch.tensor(log_ab_prev).float())

        # Importance-sampling history
        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count',   torch.zeros(timesteps))

        self._prop_diagnostics = {}

    # ─────────────────────────────────────────────────────────────────────────
    # β_ij and schedule helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _beta_ij(self, batched_graph, edge_index):
        """β_ij = min(β_max, (d_i*·d_j* / d̄²_g)^(γ/2))  per candidate pair.

        Falls back to β_ij=1 when target_degree is absent (e.g. empty-graph
        sampler before the first real graph is conditioned on).
        """
        if not hasattr(batched_graph, 'target_degree') or batched_graph.target_degree is None:
            return torch.ones(edge_index.shape[1], device=self.device)

        d_star = batched_graph.target_degree.float()  # (N,)
        d_i    = d_star[edge_index[0]]                # (|Q|,)
        d_j    = d_star[edge_index[1]]                # (|Q|,)

        # Per-graph mean degree from FULL edge set (not sparse query)
        b = batched_graph.num_graphs
        batch_full_e = batched_graph.batch[batched_graph.full_edge_index[0]]
        n_pos_full   = scatter(batched_graph.full_edge_attr.float(),
                               batch_full_e, dim=0, dim_size=b, reduce='sum')  # (B,)
        n_nodes      = batched_graph.nodes_per_graph.float().to(self.device)   # (B,)
        d_bar        = (2.0 * n_pos_full / n_nodes.clamp(min=1.0)).clamp(min=1.0)
        d_bar_sq     = d_bar ** 2                                               # (B,)

        batch_q = batched_graph.batch[edge_index[0]]                           # (|Q|,)
        s_ij    = (d_i * d_j / d_bar_sq[batch_q]).clamp(min=1e-6)
        return (s_ij ** (self.daca_gamma / 2.0)).clamp(max=self.beta_max)     # (|Q|,)

    def _alpha_bar_ij(self, t_edge, beta_ij):
        """ᾱ_t^{β_ij}: forward retention probability.  Shape (|Q|,)."""
        log_ab = self.log_cumprod_alpha_abs[t_edge]        # (|Q|,)
        return (log_ab * beta_ij).exp().clamp(1e-9, 1.0)

    def _alpha_bar_ij_prev(self, t_edge, beta_ij):
        """ᾱ_{t-1}^{β_ij}  (= 1 at t_index=0).  Shape (|Q|,)."""
        log_ab_p = self.log_cumprod_alpha_abs_prev[t_edge] # (|Q|,) log(ᾱ_t_index)
        return (log_ab_p * beta_ij).exp().clamp(1e-9, 1.0)

    def _unmask_prob(self, t_edge, beta_ij):
        """r_t^{ij} = (ᾱ_{t-1}^β - ᾱ_t^β) / (1 - ᾱ_t^β).  Shape (|Q|,)."""
        ab_t   = self._alpha_bar_ij(t_edge, beta_ij)       # (|Q|,)
        ab_tm1 = self._alpha_bar_ij_prev(t_edge, beta_ij)  # (|Q|,)
        num    = (ab_tm1 - ab_t).clamp(min=0.0)
        den    = (1.0 - ab_t).clamp(min=1e-9)
        return (num / den).clamp(0.0, 1.0)                 # (|Q|,)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward process
    # ─────────────────────────────────────────────────────────────────────────

    def _q_sample_edge_abs(self, e0, t_edge, beta_ij):
        """Sample e_t ~ q(e_t|e_0): retain e_0 with prob ᾱ_t^β, else mask to M."""
        ab_t = self._alpha_bar_ij(t_edge, beta_ij)                     # (E,)
        mask = (torch.rand(e0.shape[0], device=self.device) > ab_t)    # True → mask
        return torch.where(mask, torch.full_like(e0, _M), e0)          # (E,) ∈{0,1,2}

    def _q_sample_node(self, batched_graph, t_node):
        """Sample x_t^{node} from standard binary marginal q(x_t|x_0)."""
        log_x0 = batched_graph.log_node_attr
        log_ca  = extract(self._log_ca_n,  t_node, log_x0.shape)
        log_1mca = extract(self._log_1mca_n, t_node, log_x0.shape)
        log_prob = log_add_exp(log_x0 + log_ca,
                               log_1mca + self.log_final_prob_node)     # (N,2)
        return self.log_sample_categorical(log_prob, self.num_node_classes)  # (N,2)

    # ─────────────────────────────────────────────────────────────────────────
    # Node KL and reverse
    # ─────────────────────────────────────────────────────────────────────────

    def _node_posterior(self, log_x0, log_xt, t_node, t_prev_node=None):
        """q(x_{t-1}^{node} | x_t^{node}, x_0^{node}) — standard binary marginal."""
        if t_prev_node is None:
            tmin1 = (t_node - 1).clamp(min=0)
        else:
            tmin1 = t_prev_node.clamp(min=0)

        log_p1    = self.log_final_prob_node[:, 1]
        log_1mp1  = self.log_final_prob_node[:, 0]

        log_x0r   = log_x0[:, 1]
        log_1mx0r = log_x0[:, 0]
        log_xtr   = log_xt[:, 1]
        log_1mxtr = log_xt[:, 0]

        log_a_t   = extract(self._log_alpha_n,  t_node, log_x0r.shape)
        log_b_t   = extract(self._log_1ma_n,    t_node, log_x0r.shape)
        log_ca_tm1 = extract(self._log_ca_n,    tmin1,  log_x0r.shape)
        log_1mca_tm1 = extract(self._log_1mca_n, tmin1, log_x0r.shape)

        log_xm1_eq0_gxt = log_add_exp(log_b_t + log_p1   + log_xtr,
                                       log_1_min_a(log_b_t + log_p1) + log_1mxtr)
        log_xm1_eq1_gxt = log_add_exp(log_add_exp(log_a_t, log_b_t + log_p1) + log_xtr,
                                       log_b_t + log_1mp1 + log_1mxtr)

        log_xm1_eq0_gx0 = log_add_exp(log_1mca_tm1 + log_1mp1,
                                       log_ca_tm1 + log_1mx0r)
        log_xm1_eq1_gx0 = log_add_exp(log_ca_tm1 + log_x0r,
                                       log_1mca_tm1 + log_p1)

        log0 = log_xm1_eq0_gxt + log_xm1_eq0_gx0
        log1 = log_xm1_eq1_gxt + log_xm1_eq1_gx0
        unnorm = torch.stack([log0, log1], dim=1)
        return unnorm - unnorm.logsumexp(1, keepdim=True)  # (N,2)

    def _compute_node_kl(self, batched_graph, t_node, log_pred_node, t):
        """Node ELBO: KL at t>0, reconstruction NLL at t=0."""
        b = batched_graph.num_graphs

        log_true_node = self._node_posterior(
            batched_graph.log_node_attr, batched_graph.log_node_attr_t, t_node)
        log_model_node = self._node_p_pred(batched_graph, t_node, log_pred_node)

        kl_n = self.multinomial_kl(log_true_node, log_model_node)
        kl_n = scatter(kl_n, batched_graph.batch, dim=0, dim_size=b, reduce='sum')

        nll_n = -log_categorical(batched_graph.log_node_attr, log_model_node)
        nll_n = scatter(nll_n, batched_graph.batch, dim=0, dim_size=b, reduce='sum')

        mask_t0 = (t == 0).float()
        return mask_t0 * nll_n + (1.0 - mask_t0) * kl_n   # (B,)

    def _node_p_pred(self, batched_graph, t_node, log_pred_node, t_prev_node=None):
        """p(x_{t-1}^{node}|x_t^{node}) via predicted x_0."""
        return self._node_posterior(
            log_pred_node, batched_graph.log_node_attr_t, t_node, t_prev_node)

    # ─────────────────────────────────────────────────────────────────────────
    # Prior KL  L_T
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_LT(self, batched_graph):
        """L_T = Σ_{ij} α_min^{β_ij} · CE(e_0, ρ_ij)   (B,) tensor.

        ρ_ij = Chung-Lu pair probability = min(1, d_i*·d_j* / (2·|E|_g)).
        The weight α_min^{β_ij} = ᾱ_T^{β_ij} is the fraction of pair (i,j)
        that remains visible at t=T, making this the expected KL of the scaffold.
        """
        b          = batched_graph.num_graphs
        full_ei    = batched_graph.full_edge_index
        batch_fe   = batched_graph.batch[full_ei[0]]

        with torch.no_grad():
            beta_full = self._beta_ij(batched_graph, full_ei)  # (E,)
            alpha_min_beta = self.alpha_min ** beta_full        # (E,)  ᾱ_T^β

            # Chung-Lu ρ_ij = d_i*·d_j* / (2·n_edges_g)
            if hasattr(batched_graph, 'target_degree') and batched_graph.target_degree is not None:
                d_star = batched_graph.target_degree.float()
                d_i    = d_star[full_ei[0]]
                d_j    = d_star[full_ei[1]]
                n_pos  = scatter(batched_graph.full_edge_attr.float(),
                                 batch_fe, dim=0, dim_size=b, reduce='sum')  # (B,)
                denom  = (2.0 * n_pos[batch_fe]).clamp(min=1.0)
                rho_ij = (d_i * d_j / denom).clamp(1e-6, 1.0 - 1e-6)
            else:
                # fallback: global density
                n_pos  = scatter(batched_graph.full_edge_attr.float(),
                                 batch_fe, dim=0, dim_size=b, reduce='sum')
                total_c = getattr(batched_graph, 'total_cands_per_graph',
                                  batched_graph.edges_per_graph).float().to(self.device)
                rho_g  = (n_pos / total_c.clamp(min=1.0)).clamp(1e-6, 1.0-1e-6)
                rho_ij = rho_g[batch_fe]

        e0 = batched_graph.full_edge_attr.float()
        ce_prior = -(e0 * torch.log(rho_ij) + (1.0 - e0) * torch.log(1.0 - rho_ij))
        lt_pair  = alpha_min_beta * ce_prior
        return scatter(lt_pair, batch_fe, dim=0, dim_size=b, reduce='sum')  # (B,)

    # ─────────────────────────────────────────────────────────────────────────
    # SparseDiff query builder (same logic as BinomialDiffusionVanilla)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_edge_query(self, batched_graph, t_edge_full):
        labels  = batched_graph.full_edge_attr
        device  = labels.device
        ef      = self.edge_fraction
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        has_total  = hasattr(batched_graph, 'total_cands_per_graph')
        pre_sparse = (has_total and
                      batched_graph.edges_per_graph.sum() <
                      batched_graph.total_cands_per_graph.sum())

        if pre_sparse:
            query_idx = torch.arange(labels.numel(), device=device)
            neg_mask  = (labels == 0)
            query_weight = torch.where(neg_mask,
                torch.full_like(labels, 1.0/ef, dtype=torch.float),
                torch.ones_like(labels, dtype=torch.float))
        else:
            n_q_neg  = max(1, int(round(ef * neg_idx.numel())))
            perm     = torch.randperm(neg_idx.numel(), device=device)[:n_q_neg]
            samp_neg = neg_idx[perm]
            query_idx    = torch.cat([pos_idx, samp_neg])
            query_weight = torch.cat([
                torch.ones(pos_idx.numel(), device=device),
                torch.full((samp_neg.numel(),), 1.0/ef, device=device),
            ])

        query_ei = batched_graph.full_edge_index[:, query_idx]
        t_edge_q = t_edge_full[query_idx]
        return query_idx, query_weight, query_ei, t_edge_q

    # ─────────────────────────────────────────────────────────────────────────
    # Time sampling (same as BinomialDiffusionVanilla)
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
        elif method == 'uniform':
            length = self.Lt_count.shape[0]
            t  = torch.randint(0, length, (b,), device=device).long()
            pt = torch.ones_like(t).float() / length
            return t, pt
        else:
            raise ValueError(f'Unknown method: {method}')

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0] + batched_graph.node_attr.shape[0]

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

        # ── x_0 log one-hots ─────────────────────────────────────────────────
        batched_graph.log_node_attr = index_to_log_onehot(
            batched_graph.node_attr, self.num_node_classes)
        # full_edge_attr ∈ {0,1} — binary x_0, NOT stored as 3-class

        # ── Sample t ────────────────────────────────────────────────────────
        t, pt = self._sample_time(b, self.device, self.sample_time_method)
        t_node        = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge_full   = t.repeat_interleave(batched_graph.edges_per_graph)

        # ── Forward process: sample x_t for edges (3-class) ─────────────────
        # Always compute β for FULL set (needed for x_t message passing + w+ count)
        beta_full = self._beta_ij(batched_graph, batched_graph.full_edge_index)
        e0_full   = batched_graph.full_edge_attr                                # (E,) ∈{0,1}
        e_t_full  = self._q_sample_edge_abs(e0_full, t_edge_full, beta_full)   # (E,) ∈{0,1,2}
        batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_t_full, _ABS_CLASSES)

        # ── Forward process: sample x_t for nodes (2-class) ─────────────────
        batched_graph.log_node_attr_t = self._q_sample_node(batched_graph, t_node)

        # ── SparseDiff: build query subset ───────────────────────────────────
        if is_train and self.edge_fraction < 1.0:
            query_idx, query_weight, query_ei, t_edge_q = \
                self._build_edge_query(batched_graph, t_edge_full)
        else:
            E         = e0_full.shape[0]
            query_idx = torch.arange(E, device=self.device)
            query_weight = torch.ones(E, device=self.device)
            query_ei  = batched_graph.full_edge_index
            t_edge_q  = t_edge_full

        batch_q = batched_graph.batch[query_ei[0]]
        e0_q    = e0_full[query_idx]                    # (|Q|,) ∈{0,1}
        e_t_q   = e_t_full[query_idx]                   # (|Q|,) ∈{0,1,2}
        beta_q  = beta_full[query_idx]                  # (|Q|,)

        # ── Run denoiser  →  2-class x_0 predictions ────────────────────────
        batched_graph.query_edge_index = query_ei
        out_node, out_edge_q = self._denoise_fn(batched_graph, t_node, t_edge_q)
        batched_graph.query_edge_index = None

        log_pred_node   = F.log_softmax(out_node,   dim=-1)   # (N,2)
        log_pred_edge_q = F.log_softmax(out_edge_q, dim=-1)   # (|Q|,2)

        # ── Absorbing edge ELBO ───────────────────────────────────────────────
        # Only masked pairs contribute to the KL.
        # KL(q(e_{t-1}|e_t=M,e_0) || p_θ(e_{t-1}|e_t=M)) = r_t · CE(e_0, p̂)
        is_masked = (e_t_q == _M).float()              # (|Q|,)  1 → masked
        r_t_q     = self._unmask_prob(t_edge_q, beta_q)  # (|Q|,)

        ce_q = -(e0_q.float() * log_pred_edge_q[:, 1]
                 + (1.0 - e0_q.float()) * log_pred_edge_q[:, 0])  # (|Q|,)

        elbo_pair = r_t_q * ce_q * is_masked * query_weight        # (|Q|,)

        # ── P2 fix: w+^M(t) from masked pool (counts over FULL set) ──────────
        # At each t, the masked pool is a biased sample of e_0 values.
        # Reweight by n_neg_masked / n_pos_masked within that pool.
        if self.kl_pos_weight:
            batch_fe  = batched_graph.batch[batched_graph.full_edge_index[0]]
            is_m_full = (e_t_full == _M).float()                   # (E,)
            e0_ff     = e0_full.float()
            n_pos_m   = scatter(e0_ff * is_m_full, batch_fe,
                                dim=0, dim_size=b, reduce='sum')   # (B,)
            n_neg_m   = scatter((1.0-e0_ff) * is_m_full, batch_fe,
                                dim=0, dim_size=b, reduce='sum')   # (B,)
            w_plus    = n_neg_m / n_pos_m.clamp(min=1.0)          # (B,) masked-(1-ρ)/ρ
            kl_pos_w  = w_plus[batch_q] * e0_q.float() + (1.0 - e0_q.float())
        else:
            kl_pos_w  = torch.ones(e0_q.shape[0], device=self.device)

        elbo_w = elbo_pair * kl_pos_w
        kl_edge = scatter(elbo_w, batch_q, dim=0, dim_size=b, reduce='sum')   # (B,)

        # ── Reconstruction NLL at t=0  (L_0) ────────────────────────────────
        nll_edge_q = ce_q * query_weight * kl_pos_w
        nll_edge   = scatter(nll_edge_q, batch_q, dim=0, dim_size=b, reduce='sum')

        # ── Node ELBO ────────────────────────────────────────────────────────
        # Includes t=0 switching automatically inside _compute_node_kl
        node_loss  = self._compute_node_kl(batched_graph, t_node, log_pred_node, t)

        # ── t=0 mask (use NLL instead of KL) ─────────────────────────────────
        mask_t0    = (t == 0).float()
        loss_step  = mask_t0 * (nll_edge + node_loss) + (1.0 - mask_t0) * (kl_edge + node_loss)

        # ── Prior KL  L_T ────────────────────────────────────────────────────
        lt = self._compute_LT(batched_graph)

        # ── Importance sampling history update ───────────────────────────────
        if is_train:
            Lt2      = loss_step.pow(2)
            Lt2_prev = self.Lt_history.gather(0, t)
            self.Lt_history.scatter_(0, t, (0.1*Lt2 + 0.9*Lt2_prev).detach())
            self.Lt_count.scatter_add_(0, t, torch.ones_like(Lt2))

        # ── Proposition diagnostics ───────────────────────────────────────────
        if is_train:
            with torch.no_grad():
                pred_p1 = log_pred_edge_q.exp()[:, 1]
                kl_e_s  = (elbo_pair * e0_q.float()).sum().item()
                kl_n_s  = (elbo_pair * (1.0-e0_q.float())).sum().item()
                kl_tot  = kl_e_s + kl_n_s + 1e-12
                xt_dens = (e_t_full == 1).float().mean().item()
                xt_mask = (e_t_full == _M).float().mean().item()
                self._prop_diagnostics = {
                    'p1/pred_density_mean': pred_p1.mean().item(),
                    'p1/pred_density_std':  pred_p1.std().item(),
                    'p2/kl_frac_edge_raw':  kl_e_s / kl_tot,
                    'p2/w_plus_masked_mean': (w_plus.mean().item()
                                              if self.kl_pos_weight else 1.0),
                    'p3/xt_edge_density':   xt_dens,
                    'p3/xt_mask_fraction':  xt_mask,
                    'p3/alpha_min':         self.alpha_min,
                }

        vb_loss = loss_step / pt + self.lambda_LT * lt
        return -vb_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_data_for_sampling(self, batched_graph):
        """Initialize x_T from partial-absorbing prior p(x_T).

        p(e_{ij,T}=M)   = 1 - α_min^{β_ij}
        p(e_{ij,T}=k)   = α_min^{β_ij} · Bernoulli(ρ_ij)   k∈{0,1}
        """
        batched_graph.log_node_attr = index_to_log_onehot(
            batched_graph.node_attr, self.num_node_classes)

        # Nodes: sample from marginal prior
        log_p_node = torch.ones_like(batched_graph.log_node_attr) * self.log_final_prob_node
        batched_graph.log_node_attr_t = self.log_sample_categorical(log_p_node, self.num_node_classes)

        # Edges: partial-absorbing prior
        b       = batched_graph.num_graphs
        full_ei = batched_graph.full_edge_index
        batch_fe = batched_graph.batch[full_ei[0]]
        E       = full_ei.shape[1]

        # β_ij at highest noise level
        beta_full = self._beta_ij(batched_graph, full_ei)
        ab_T      = (self.alpha_min ** beta_full).clamp(1e-9, 1.0)    # (E,)

        # Chung-Lu ρ_ij
        if hasattr(batched_graph, 'target_degree') and batched_graph.target_degree is not None:
            d_star  = batched_graph.target_degree.float()
            d_i     = d_star[full_ei[0]]
            d_j     = d_star[full_ei[1]]
            n_pos   = scatter(batched_graph.full_edge_attr.float(),
                              batch_fe, dim=0, dim_size=b, reduce='sum')
            denom   = (2.0 * n_pos[batch_fe]).clamp(min=1.0)
            rho_ij  = (d_i * d_j / denom).clamp(1e-6, 1.0 - 1e-6)
        else:
            n_pos   = scatter(batched_graph.full_edge_attr.float(),
                              batch_fe, dim=0, dim_size=b, reduce='sum')
            total_c = getattr(batched_graph, 'total_cands_per_graph',
                              batched_graph.edges_per_graph).float().to(self.device)
            rho_g   = (n_pos / total_c.clamp(min=1.0)).clamp(1e-6, 1.0-1e-6)
            rho_ij  = rho_g[batch_fe]

        # Sample: visible with prob ab_T → Bernoulli(ρ_ij), else M
        u_vis   = torch.rand(E, device=self.device)
        u_edge  = torch.rand(E, device=self.device)
        is_vis  = (u_vis < ab_T)
        e_bern  = (u_edge < rho_ij).long()                             # (E,) ∈{0,1}
        e_T     = torch.where(is_vis, e_bern, torch.full_like(e_bern, _M))  # (E,) ∈{0,1,2}

        batched_graph.log_full_edge_attr = index_to_log_onehot(
            batched_graph.full_edge_attr, _PRED_CLASSES)               # x_0 placeholder (binary)
        batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_T, _ABS_CLASSES)
        return batched_graph

    def p_sample(self, batched_graph, t_node, t_edge,
                 t_prev_node=None, t_prev_edge=None):
        """One absorbing reverse step.

        Edges:
          - e_t ∈ {0,1}: copy deterministically  (no KL contribution)
          - e_t = M:      with prob r_t → sample e_0 from p̂_θ; else stay M
        Nodes: standard binary marginal reverse.
        """
        with torch.no_grad():
            full_ei   = batched_graph.full_edge_index
            beta_full = self._beta_ij(batched_graph, full_ei)

            # Get x_0 predictions from denoiser
            out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge)
            log_pred_node = F.log_softmax(out_node, dim=-1)    # (N,2)
            log_pred_edge = F.log_softmax(out_edge, dim=-1)    # (E,2)

            # Current edge state
            e_t      = batched_graph.log_full_edge_attr_t.argmax(-1)   # (E,)
            is_masked = (e_t == _M)                                     # (E,) bool

            # r_t: probability of unmasking this step
            r_t = self._unmask_prob(t_edge, beta_full)         # (E,)

            # For masked pairs: draw Bernoulli(r_t) to decide whether to unmask
            u        = torch.rand(e_t.shape[0], device=self.device)
            unmask   = is_masked & (u < r_t)

            # Sample x_0 from model (only used when unmasking)
            e_hat = self.log_sample_categorical(log_pred_edge, _PRED_CLASSES).argmax(-1)  # (E,)

            # New edge state
            e_tm1 = torch.where(
                is_masked,
                torch.where(unmask, e_hat, torch.full_like(e_t, _M)),
                e_t)
            log_edge_tm1 = index_to_log_onehot(e_tm1, _ABS_CLASSES)

            # Node reverse step
            log_node_tm1 = self.log_sample_categorical(
                self._node_p_pred(batched_graph, t_node, log_pred_node, t_prev_node),
                self.num_node_classes)

        return log_node_tm1, log_edge_tm1

    def sample(self, num_samples, inference_steps=None):
        """Generate graphs via absorbing reverse chain.

        Overrides DiffusionBase.sample to:
        1. Use PA-DACA p_sample (absorbing reverse)
        2. Convert 3-class edge states {0,1,M} → binary {0,1} at the end
           (M state treated as non-edge)
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
            print(f'Sample step {step_i+1:4d}/{n_steps}  (t={t})', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)

            if t_prev >= 0:
                tp_node = torch.full((num_nodes,), t_prev, device=self.device, dtype=torch.long)
                tp_edge = torch.full((num_edges,), t_prev, device=self.device, dtype=torch.long)
            else:
                tp_node = tp_edge = None

            log_node_tm1, log_edge_tm1 = self.p_sample(
                batched_graph, t_node, t_edge,
                t_prev_node=tp_node, t_prev_edge=tp_edge)
            batched_graph.log_node_attr_t      = log_node_tm1
            batched_graph.log_full_edge_attr_t = log_edge_tm1

        print()

        # Convert 3-class absorbing states → binary edge labels
        # M (state 2) → non-edge (0);  state 1 → edge;  state 0 → non-edge
        e_state   = batched_graph.log_full_edge_attr_t.argmax(-1)   # (E,) ∈{0,1,2}
        edge_attr = (e_state == 1).long()                            # (E,) binary

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
