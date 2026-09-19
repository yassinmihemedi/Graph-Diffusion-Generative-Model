"""
ScalableGraphDiffusion — full implementation of Algorithms 1–6.

Forward process  (Alg. 1): Bernoulli edge-deletion + VP Gaussian feature noise.
f_θ forward pass (Alg. 2): handled by ScalableGNN in layers/layers.py.
Disc_φ           (Alg. 3): handled by ScalableDisc in layers/layers.py.
Training losses  (Alg. 4): L_X + L_F + L_E + λ_adv * L_G  /  L_D.
Training loop    (Alg. 5): driven by the existing GraphExperiment in experiment.py.
Sampling         (Alg. 6): full reverse-diffusion with optional disc guidance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

try:
    from torch_scatter import scatter as _scatter
except (ImportError, OSError):
    from torch_geometric.utils import scatter as _scatter

# reuse the simple scatter from diffusion_base for the sample() helper
from diffusion.diffusion_base import scatter as _scatter_base

_eps = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
#  Noise schedule helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_edge_schedule(timesteps, eps_min=0.01, eps_max=0.99):
    """Linear ε_t → ρ_t = 1-ε_t, ρ̄_t = cumproduct(ρ).

    At t=T, ρ̄_T ≈ 0  (almost all edges deleted).
    At t=0, ρ̄_0 = 1  (all edges survive — clean graph).
    """
    t = np.arange(1, timesteps + 1, dtype=np.float64)
    eps_t  = eps_min + (t / timesteps) * (eps_max - eps_min)
    rho_t  = 1.0 - eps_t
    rho_bar = np.cumprod(rho_t)
    return (torch.tensor(rho_t,   dtype=torch.float32),
            torch.tensor(rho_bar, dtype=torch.float32))


def _build_feature_schedule(timesteps, s=0.008):
    """Cosine ᾱ_t schedule (Nichol & Dhariwal 2021) for VP Gaussian features.

    Returns tensors of length T:
        alpha_bar   ᾱ_t
        beta_t      β_t  = 1 - ᾱ_t/ᾱ_{t-1}
        alpha_t     α_t  = 1 - β_t
        beta_tilde  β̃_t  = (1-ᾱ_{t-1})·β_t / (1-ᾱ_t)  [DDPM posterior var]
        alpha_bar_prev  ᾱ_{t-1}  (ᾱ_0 = 1 for t=1)
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    ab = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    ab = ab / ab[0]                              # normalise so ᾱ_0 = 1
    ab = ab[1:]                                  # ᾱ_1 … ᾱ_T  (length T)
    ab_prev = np.concatenate([[1.0], ab[:-1]])   # ᾱ_0 … ᾱ_{T-1}

    beta_t    = np.clip(1.0 - ab / ab_prev, 1e-6, 0.999)
    alpha_t   = 1.0 - beta_t
    # β̃_t = (1-ᾱ_{t-1})·β_t / (1-ᾱ_t); set to 0 at t=1 (deterministic)
    beta_tilde = (1.0 - ab_prev) * beta_t / np.maximum(1.0 - ab, 1e-10)
    beta_tilde[0] = 0.0

    return (torch.tensor(ab,         dtype=torch.float32),
            torch.tensor(beta_t,     dtype=torch.float32),
            torch.tensor(alpha_t,    dtype=torch.float32),
            torch.tensor(beta_tilde, dtype=torch.float32),
            torch.tensor(ab_prev,    dtype=torch.float32))


# ──────────────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────────────

class ScalableGraphDiffusion(nn.Module):
    """Scalable Graph Diffusion Generative Model (Algorithms 1–6).

    Plug-in replacement for BinomialDiffusionVanilla in the existing
    training pipeline.  ``forward(mode='log_prob')`` and ``sample()``
    have the same signature so GraphExperiment works unchanged.

    Key differences from BinomialDiffusion:
      - Continuous VP Gaussian noise on node/edge features (not categorical).
      - Bernoulli edge-deletion noise on graph structure (not multinomial).
      - GAN discriminator (ScalableDisc) instead of gradient-guidance-only disc.
      - Loss: L_X + L_F + L_E + λ_adv·L_G  /  L_D  (Algorithms 4–5).

    Args:
        num_node_classes  (int)   d_x — node feature dim (= one-hot width)
        num_edge_classes  (int)   d_f — edge feature dim (= one-hot width)
        initial_graph_sampler     from datasets.data (provides sample())
        denoise_fn                ScalableGNN instance  (f_θ)
        timesteps         (int)   T
        discriminator             ScalableDisc instance or None
        disc_lambda       (float) λ_adv  (0 = disabled)
        sample_rho        (float) edge density for initialising G_T at sampling
        device            (str)
        **kwargs          ignored (API compatibility with BinomialDiffusionVanilla)
    """

    def __init__(
        self,
        num_node_classes,
        num_edge_classes,
        initial_graph_sampler,
        denoise_fn,
        timesteps=1000,
        discriminator=None,
        disc_lambda=0.0,
        sample_rho=0.5,
        device='cuda',
        edge_fraction=1.0,
        inference_steps=None,
        # ── unused but accepted for API parity ──
        loss_type=None, parametrization=None,
        final_prob_node=None, final_prob_edge=None,
        sample_time_method=None, noise_schedule=None,
        lambda_ce=None, ce_pos_weight=None,
    ):
        super().__init__()
        self.d_x              = num_node_classes
        self.d_f              = num_edge_classes
        self.num_timesteps    = timesteps
        self.initial_graph_sampler = initial_graph_sampler
        self._denoise_fn      = denoise_fn
        self.discriminator    = discriminator
        self.disc_lambda      = disc_lambda
        self.sample_rho       = sample_rho if sample_rho is not None else 0.5
        self._device          = device

        # ── SparseDiff λ: fraction of pairs used per training step ────────────
        # During training, subsample edge_fraction of all n(n-1)/2 candidate
        # pairs for L_E; scale loss by 1/edge_fraction for unbiased gradient.
        # edge_fraction=1.0 (default) = standard full prediction scope.
        assert 0.0 < edge_fraction <= 1.0, 'edge_fraction must be in (0, 1]'
        self.edge_fraction = edge_fraction

        # ── inference_steps: DDPM skip schedule for fast sampling ─────────────
        # None = use all T steps. K < T = run only K reverse steps with proper
        # DDPM posterior computed for the actual (t, t_prev) pair.
        if inference_steps is not None:
            assert 1 <= inference_steps <= timesteps, \
                f'inference_steps must be in [1, {timesteps}]'
        self.inference_steps = inference_steps

        # ── Edge-deletion schedule (Bernoulli) ────────────────────────────────
        rho_t, rho_bar = _build_edge_schedule(timesteps)
        self.register_buffer('rho_t',     rho_t)
        self.register_buffer('rho_bar_t', rho_bar)

        # ── VP feature schedule (cosine) ──────────────────────────────────────
        ab, beta_t, alpha_t, beta_tilde, ab_prev = _build_feature_schedule(timesteps)
        self.register_buffer('alpha_bar',      ab)
        self.register_buffer('beta_t',         beta_t)
        self.register_buffer('alpha_t',        alpha_t)
        self.register_buffer('beta_tilde',     beta_tilde)
        self.register_buffer('alpha_bar_prev', ab_prev)

        # ── Importance-sampling history (mirrors BinomialDiffusionVanilla) ────
        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count',   torch.zeros(timesteps))
        self._loss_components = {}

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self._device)

    # ─────────────────────────────────────────────────────────────────────────
    #  nn.Module interface (mirrors DiffusionBase)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, *args, mode='log_prob', **kwargs):
        if mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        raise ValueError(f'Unknown mode: {mode}')

    def log_prob(self, batched_graph):
        if self.training:
            return self._train_loss(batched_graph)
        return self._eval_loss(batched_graph)

    def _calc_num_entries(self, batched_graph):
        """Total elements for bpd normalisation (matches BinomialDiffusionVanilla)."""
        return batched_graph.full_edge_attr.shape[0] + batched_graph.node_attr.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    #  Schedule indexing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ab(self, t_graph):
        """ᾱ_t  per graph.  t_graph: (B,) 1-indexed."""
        return self.alpha_bar[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    def _ab_prev(self, t_graph):
        """ᾱ_{t-1} per graph."""
        return self.alpha_bar_prev[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    def _rb(self, t_graph):
        """ρ̄_t per graph."""
        return self.rho_bar_t[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    def _bt(self, t_graph):
        """β_t per graph."""
        return self.beta_t[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    def _at(self, t_graph):
        """α_t per graph."""
        return self.alpha_t[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    def _btt(self, t_graph):
        """β̃_t per graph."""
        return self.beta_tilde[(t_graph - 1).clamp(0, self.num_timesteps - 1)]

    # ─────────────────────────────────────────────────────────────────────────
    #  Algorithm 1: forward process
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_process(self, batched_graph, t_graph):
        """Apply forward noise at diffusion step t (1-indexed).

        Returns a SimpleNamespace with all noised quantities plus the noise
        samples η_X, η_F needed for the DDPM posterior in the reverse step.
        """
        device = self.device
        B = batched_graph.num_graphs

        # Clean features: one-hot float  X_0 ∈ R^{N×d_x},  F_0 ∈ R^{E×d_f}
        X_0 = F.one_hot(batched_graph.node_attr, self.d_x).float()         # (N, d_x)
        F_0 = F.one_hot(batched_graph.full_edge_attr, self.d_f).float()    # (E, d_f)

        # ── Structural noise: Bernoulli edge deletion ─────────────────────────
        rho_bar = self._rb(t_graph)                                         # (B,)
        edge_batch = batched_graph.batch[batched_graph.full_edge_index[0]]  # (E,)
        rho_bar_e  = rho_bar[edge_batch]                                    # (E,)
        # Zero-step guard: if t=0 keep all edges
        surv_mask = torch.bernoulli(rho_bar_e.clamp(0.0, 1.0)).bool()      # (E,)
        surv_ei   = batched_graph.full_edge_index[:, surv_mask]             # (2,|E_t|)

        # ── Feature noise: VP Gaussian ────────────────────────────────────────
        ab       = self._ab(t_graph)                                        # (B,)
        ab_node  = ab[batched_graph.batch]                                  # (N,)
        eta_X    = torch.randn_like(X_0)
        sqrt_ab  = ab_node.sqrt().unsqueeze(-1)
        X_t      = sqrt_ab * X_0 + (1 - ab_node).sqrt().unsqueeze(-1) * eta_X

        # Edge features: only for surviving edges
        F_0_surv = F_0[surv_mask]                                           # (|E_t|, d_f)
        eta_F    = torch.randn_like(F_0_surv)
        if surv_ei.shape[1] > 0:
            ab_e    = ab[batched_graph.batch[surv_ei[0]]]                   # (|E_t|,)
            sqrt_ab_e = ab_e.sqrt().unsqueeze(-1)
            F_t = sqrt_ab_e * F_0_surv + (1 - ab_e).sqrt().unsqueeze(-1) * eta_F
        else:
            F_t  = torch.zeros(0, self.d_f, device=device)
            eta_F = torch.zeros(0, self.d_f, device=device)

        return SimpleNamespace(
            x_cont          = X_t,
            surv_edge_index = surv_ei,
            surv_mask       = surv_mask,       # (E_full,) bool mask over full_edge_index
            edge_feat_cont  = F_t,
            eta_X           = eta_X,
            eta_F           = eta_F,
            X_0             = X_0,
            F_0_surv        = F_0_surv,
            ab              = ab,
        )

    def _apply_state(self, batched_graph, state):
        """Attach forward-process tensors to batched_graph for the GNN."""
        batched_graph.x_cont          = state.x_cont
        batched_graph.surv_edge_index = state.surv_edge_index
        batched_graph.edge_feat_cont  = state.edge_feat_cont

    # ─────────────────────────────────────────────────────────────────────────
    #  Algorithm 4: compute training losses
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batched_graph, t_graph, state, enable_adv=True):
        """Return L_recon (B,), L_G (B,) per-graph tensors."""
        device   = self.device
        B        = batched_graph.num_graphs
        N        = batched_graph.num_nodes

        t_node = t_graph.repeat_interleave(batched_graph.nodes_per_graph)  # (N,)
        t_edge = t_graph.repeat_interleave(batched_graph.edges_per_graph)  # (E_full,)

        # f_θ forward pass
        X_hat, F_hat, p_hat, P_t, H = self._denoise_fn(
            batched_graph, t_node, t_edge)

        # ── L_X: node feature MSE (sum per graph) ────────────────────────────
        # Must use reduce='sum' (not 'mean') so that loglik_bpd's division by
        # num_entries yields a true per-element BPD.  reduce='mean' would
        # produce double-normalisation → BPD ≈ 0 for large graphs.
        l_x = ((X_hat - state.X_0) ** 2).sum(-1)                           # (N,)
        L_X = _scatter(l_x, batched_graph.batch, dim=0,
                        dim_size=B, reduce='sum')                           # (B,)

        # ── L_F: edge feature MSE (surviving edges only, sum per graph) ───────
        if state.surv_mask.any():
            surv_batch = batched_graph.batch[state.surv_edge_index[0]]
            l_f = ((F_hat - state.F_0_surv) ** 2).sum(-1)                  # (|E_t|,)
            L_F = _scatter(l_f, surv_batch, dim=0, dim_size=B, reduce='sum')
        else:
            L_F = torch.zeros(B, device=device)

        # ── L_E: weighted BCE with λ-fraction subsampling (SparseDiff Eq.1) ───
        # Training sees edge_fraction λ of all n(n-1)/2 candidate pairs per
        # step. Loss is scaled by 1/λ to maintain an unbiased gradient.
        # At λ=1.0 (default) the full prediction scope is used as before.
        E_full = P_t.shape[1]
        if self.edge_fraction < 1.0 and self.training:
            n_query = max(1, int(self.edge_fraction * E_full))
            perm    = torch.randperm(E_full, device=device)[:n_query]
            pi, pj  = P_t[0][perm], P_t[1][perm]
            p_hat_q = p_hat[perm]
            # Recompute balanced weights on the FULL set, then subsample.
            # This prevents weight collapse when the query sample is unbalanced.
            y_full  = batched_graph.full_edge_attr.float()                  # (E_full,)
            ep_full = batched_graph.batch[P_t[0]]
            n_pos_f = _scatter(y_full, ep_full, dim=0, dim_size=B, reduce='sum')
            n_tot_f = _scatter(torch.ones_like(y_full), ep_full, dim=0,
                               dim_size=B, reduce='sum')
            n_neg_f = n_tot_f - n_pos_f
            w_pos_f = (n_neg_f / n_tot_f.clamp(min=1.0))[ep_full]
            w_neg_f = (n_pos_f / n_tot_f.clamp(min=1.0))[ep_full]
            w_full  = w_pos_f * y_full + w_neg_f * (1.0 - y_full)
            y       = y_full[perm]
            w       = w_full[perm]
            ep_batch = ep_full[perm]
            L_E_scale = 1.0 / self.edge_fraction                            # unbiased
        else:
            pi, pj   = P_t[0], P_t[1]
            p_hat_q  = p_hat
            y        = batched_graph.full_edge_attr.float()                 # (E_full,)
            ep_batch = batched_graph.batch[pi]
            n_pos    = _scatter(y, ep_batch, dim=0, dim_size=B, reduce='sum')
            n_tot    = _scatter(torch.ones_like(y), ep_batch, dim=0,
                                dim_size=B, reduce='sum')
            n_neg    = n_tot - n_pos
            w_pos    = (n_neg / n_tot.clamp(min=1.0))[ep_batch]
            w_neg    = (n_pos / n_tot.clamp(min=1.0))[ep_batch]
            w        = w_pos * y + w_neg * (1.0 - y)
            L_E_scale = 1.0

        bce = -(y        * torch.log(p_hat_q.clamp(min=_eps))
                + (1-y)  * torch.log((1 - p_hat_q).clamp(min=_eps)))       # (|query|,)
        L_E = L_E_scale * _scatter(w * bce, ep_batch, dim=0,
                                   dim_size=B, reduce='sum')

        L_recon = L_X + L_F + L_E                                          # (B,)

        # ── L_G: Diffusion-GAN generator loss ────────────────────────────────
        # Following Wang et al. (2022): apply forward diffusion to the model's
        # predicted clean graph Ĝ_0 and let the time-conditioned discriminator
        # score q(G_t | Ĝ_0) vs q(G_t | G_0^real) at the SAME noise level t.
        # Gradients flow through X̂_0 → f_θ via the differentiable noise addition.
        L_G = torch.zeros(B, device=device)
        if enable_adv and self.discriminator is not None and self.disc_lambda > 0:
            G_fake_t = self._noise_predicted_clean(
                batched_graph, t_graph, state, X_hat, F_hat, p_hat, P_t)
            disc_kwargs = dict(
                x         = G_fake_t.x_cont,
                edge_index= G_fake_t.surv_edge_index,
                edge_feat = G_fake_t.edge_feat_cont,
                batch     = batched_graph.batch,
                num_nodes = N,
            )
            # Time-conditioned discriminator (ScalableDiscDiffGAN) needs t_graph;
            # plain ScalableDisc ignores the extra kwarg safely.
            try:
                score_fake = self.discriminator(**disc_kwargs, t_graph=t_graph)
            except TypeError:
                score_fake = self.discriminator(**disc_kwargs)
            L_G = -torch.log(score_fake.clamp(min=_eps))                   # (B,)

        return L_recon, L_G, X_hat, F_hat, p_hat, P_t

    def _noise_predicted_clean(self, batched_graph, t_graph, state,
                               X_hat, F_hat, p_hat, P_t):
        """Diffusion-GAN: apply forward diffusion at level t to the model's Ĝ_0.

        Returns Ĝ_t = q(G_t | Ĝ_0) so that the discriminator compares
            D(q(G_t | G_0^real), t)   ← real
            D(q(G_t | Ĝ_0),      t)   ← fake  (this function)
        at the SAME noise level t — the Diffusion-GAN design.

        Gradients flow through X_hat (differentiable VP noise) → f_θ (L_G).
        Edge structure is discretised via threshold but that path is detached;
        only the feature branch carries gradient.
        """
        device = self.device
        ab_t   = self._ab(t_graph)                                          # (B,) ᾱ_t

        # ── Feature noise applied to model's predicted clean features ─────────
        # X̂_t = √ᾱ_t · X̂_0 + √(1-ᾱ_t) · η   (differentiable w.r.t. X̂_0)
        ab_node   = ab_t[batched_graph.batch]                               # (N,)
        eta_X_new = torch.randn_like(X_hat)
        X_fake_t  = (ab_node.sqrt().unsqueeze(-1) * X_hat
                     + (1.0 - ab_node).sqrt().unsqueeze(-1) * eta_X_new)

        # ── Edge structure: threshold p̂ to get Ĝ_0 edge set, then delete ─────
        # Detach p_hat so only the feature branch carries gradient for L_G.
        # The structural BCE (L_E) already provides structural gradient.
        with torch.no_grad():
            fake_edge_mask = (p_hat.detach() >= 0.5)                       # (E_full,) bool
            rho_bar_t      = self._rb(t_graph)
            edge_batch     = batched_graph.batch[P_t[0]]
            rho_bar_e      = rho_bar_t[edge_batch]
            surv_mask_fake = (fake_edge_mask
                              & torch.bernoulli(rho_bar_e).bool())          # (E_full,)
            fake_surv_ei   = batched_graph.full_edge_index[:, surv_mask_fake]

        # ── Edge feature noise for surviving predicted edges ───────────────────
        if fake_surv_ei.shape[1] > 0 and F_hat.shape[0] > 0:
            # Map fake surviving edges back to F_hat (predicted edges only)
            # pred_mask: positions in P_t that were predicted as edges
            # surv_mask_fake: subset of those that also survived deletion
            # We only have F_hat for the ORIGINAL surviving edges (state.surv_mask).
            # For simplicity, initialise fake edge features with noise.
            ab_e = ab_t[batched_graph.batch[fake_surv_ei[0]]]
            eta_F_fake = torch.randn(fake_surv_ei.shape[1], self.d_f, device=device)
            F_fake_t   = (1.0 - ab_e).sqrt().unsqueeze(-1) * eta_F_fake
        else:
            F_fake_t = torch.zeros(0, self.d_f, device=device)

        return SimpleNamespace(
            x_cont          = X_fake_t,
            surv_edge_index = fake_surv_ei,
            edge_feat_cont  = F_fake_t,
        )

    def _build_fake_graph(self, batched_graph, t_graph, state, X_hat, F_hat, p_hat, P_t):
        """Construct Ĝ_{t-1}: model's reconstruction at noise level t-1.

        Gradients flow through X_hat / F_hat → f_θ so that L_G trains f_θ.
        """
        device  = self.device
        ab_tm1  = self._ab_prev(t_graph)                                    # (B,) ᾱ_{t-1}
        surv_ei = state.surv_edge_index

        # ── Predicted edge set Ê_{t-1} = E_t ∪ {recovered with p̂ ≥ 0.5} ───
        recover_mask = (~state.surv_mask) & (p_hat >= 0.5)
        pi, pj = P_t[0], P_t[1]
        new_ei = torch.stack([pi[recover_mask], pj[recover_mask]])          # (2,|new|)
        if new_ei.shape[1] > 0:
            fake_ei = torch.cat([surv_ei, new_ei], dim=1)
        else:
            fake_ei = surv_ei

        # ── Node features re-noised to level t-1 ─────────────────────────────
        ab_node = ab_tm1[batched_graph.batch]                               # (N,)
        X_fake  = (ab_node.sqrt().unsqueeze(-1) * X_hat
                   + (1 - ab_node).sqrt().unsqueeze(-1) * state.eta_X)

        # ── Edge features for Ê_{t-1} ─────────────────────────────────────────
        if surv_ei.shape[1] > 0:
            ab_e = ab_tm1[batched_graph.batch[surv_ei[0]]]
            F_fake_surv = (ab_e.sqrt().unsqueeze(-1) * F_hat
                           + (1 - ab_e).sqrt().unsqueeze(-1) * state.eta_F)
        else:
            F_fake_surv = torch.zeros(0, self.d_f, device=device)

        # Recovered edges: zero initialisation (no MLP_f during training)
        if new_ei.shape[1] > 0:
            F_fake = torch.cat(
                [F_fake_surv,
                 torch.zeros(new_ei.shape[1], self.d_f, device=device)], dim=0)
        else:
            F_fake = F_fake_surv

        return SimpleNamespace(
            x_cont          = X_fake,
            surv_edge_index = fake_ei,
            edge_feat_cont  = F_fake,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Algorithm 5: training / eval entry points
    # ─────────────────────────────────────────────────────────────────────────

    def _train_loss(self, batched_graph):
        B       = batched_graph.num_graphs
        t_graph = torch.randint(1, self.num_timesteps + 1, (B,), device=self.device)

        state = self._forward_process(batched_graph, t_graph)
        self._apply_state(batched_graph, state)

        L_recon, L_G, X_hat, F_hat, p_hat, P_t = self._compute_loss(
            batched_graph, t_graph, state, enable_adv=True)

        total = L_recon + self.disc_lambda * L_G                            # (B,)

        # Update importance-sampling history (mirrors BinomialDiffusionVanilla).
        # Under DDP each rank updates its own shard; sync so all ranks share a
        # consistent history (only when the process group is available).
        t_idx    = (t_graph - 1).clamp(0, self.num_timesteps - 1)
        Lt2      = total.detach().pow(2)
        Lt2_prev = self.Lt_history.gather(0, t_idx)
        self.Lt_history.scatter_(0, t_idx, (0.1 * Lt2 + 0.9 * Lt2_prev).detach())
        self.Lt_count.scatter_add_(0, t_idx, torch.ones_like(Lt2))
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self.Lt_history, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.Lt_count,   op=dist.ReduceOp.SUM)
        except Exception:
            pass

        self._loss_components = {
            'L_recon': L_recon.mean().item(),
            'L_G':     L_G.mean().item(),
            'total':   total.mean().item(),
        }
        return -total                                                        # (B,)

    def _eval_loss(self, batched_graph):
        B       = batched_graph.num_graphs
        t_graph = torch.randint(1, self.num_timesteps + 1, (B,), device=self.device)
        state   = self._forward_process(batched_graph, t_graph)
        self._apply_state(batched_graph, state)
        L_recon, L_G, *_ = self._compute_loss(
            batched_graph, t_graph, state, enable_adv=False)
        return -(L_recon + self.disc_lambda * L_G)

    # ─────────────────────────────────────────────────────────────────────────
    #  Discriminator training loss  L_D
    # ─────────────────────────────────────────────────────────────────────────

    def disc_loss(self, batched_graph):
        """L_D for Disc_φ using the Diffusion-GAN principle (Wang et al. 2022).

        Both real and fake are presented at the SAME noise level t:
          Real:  q(G_t | G_0^real)          — true forward process
          Fake:  q(G_t | Ĝ_0^fake)          — forward process from f_θ's prediction

        For the fake: run f_θ(G_t) → Ĝ_0, then noise Ĝ_0 back to level t.
        f_θ is called under no_grad so only Disc_φ parameters receive gradients.

        The time-conditioned discriminator D(G, t) adapts its criterion per
        noise level: coarse connectivity at high t, fine motifs at low t.
        """
        if self.discriminator is None:
            return torch.tensor(0.0, device=self.device)

        B      = batched_graph.num_graphs
        device = self.device
        N      = batched_graph.num_nodes

        # Sample noise level — uniform over all T steps
        t_graph = torch.randint(1, self.num_timesteps + 1, (B,), device=device)

        # ── Real: q(G_t | G_0^real) — true forward process at level t ─────────
        state_real = self._forward_process(batched_graph, t_graph)

        def _disc_score(ns, t_g):
            kwargs = dict(
                x         = ns.x_cont.detach(),
                edge_index= ns.surv_edge_index,
                edge_feat = ns.edge_feat_cont.detach(),
                batch     = batched_graph.batch,
                num_nodes = N,
            )
            try:
                return self.discriminator(**kwargs, t_graph=t_g)
            except TypeError:
                return self.discriminator(**kwargs)

        score_real = _disc_score(state_real, t_graph)                      # (B,)

        # ── Fake: q(G_t | Ĝ_0) — f_θ predicts Ĝ_0, then noise to level t ────
        self._apply_state(batched_graph, state_real)
        t_node = t_graph.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge = t_graph.repeat_interleave(batched_graph.edges_per_graph)
        with torch.no_grad():
            X_hat, F_hat, p_hat, P_t, _ = self._denoise_fn(
                batched_graph, t_node, t_edge)
        # Noise the model's prediction back to level t
        fake_t = self._noise_predicted_clean(
            batched_graph, t_graph, state_real, X_hat, F_hat, p_hat, P_t)
        score_fake = _disc_score(fake_t, t_graph)                          # (B,)

        L_D = (-torch.log(score_real.clamp(min=_eps)).mean()
               - torch.log((1.0 - score_fake).clamp(min=_eps)).mean())

        self._disc_components = {
            'disc_real': score_real.mean().item(),
            'disc_fake': score_fake.mean().item(),
        }
        return L_D

    # ─────────────────────────────────────────────────────────────────────────
    #  Algorithm 6: sampling (reverse diffusion)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_skip_schedule(self):
        """Return timestep sequence for reverse diffusion.

        If self.inference_steps < T: uniformly spaced steps so that
        high-t (noisy) and low-t (clean) regions each get coverage.
        Step sizes adapt to span [1, T] evenly.

        Returns list of (t_val, t_prev) pairs in reverse order: t_val
        runs T→1 (or subsampled), t_prev is the PREVIOUS (smaller) step.
        t_prev=0 marks the final deterministic step.
        """
        T = self.num_timesteps
        K = self.inference_steps if self.inference_steps is not None else T

        if K >= T:
            # Full schedule: t=T, T-1, ..., 1
            seq = list(range(T, 0, -1))
        else:
            # Uniform skip: pick K evenly-spaced indices in [1, T]
            # np.linspace guarantees endpoints are included
            import numpy as np
            seq = sorted(set(np.round(np.linspace(1, T, K)).astype(int)),
                         reverse=True)

        pairs = [(seq[i], seq[i+1] if i+1 < len(seq) else 0)
                 for i in range(len(seq))]
        return pairs

    def _ddpm_coefficients(self, t_val: int, t_prev: int):
        """Compute DDPM posterior coefficients for a (possibly skipped) step.

        When t_prev = t-1 this reduces to the standard DDPM posterior.
        When t_prev < t-1 (skip schedule) we use ᾱ_{t_prev} instead of ᾱ_{t-1},
        re-deriving β_eff and β̃_eff for the larger jump.

        Returns (ab_t, ab_tprev, beta_eff, alpha_eff, beta_tilde_eff, rho_bar_t, rho_bar_tprev)
        all as scalar tensors on self.device.
        """
        device = self.device
        ab_t     = self.alpha_bar[t_val - 1]
        ab_tprev = self.alpha_bar[t_prev - 1] if t_prev > 0 \
                   else torch.ones(1, device=device).squeeze()

        # β_eff = 1 - ᾱ_t/ᾱ_{t_prev}  (fraction of signal lost in this jump)
        beta_eff  = (1.0 - ab_t / ab_tprev.clamp(min=1e-8)).clamp(0.0, 1.0)
        alpha_eff = 1.0 - beta_eff

        # β̃_eff = (1 - ᾱ_{t_prev}) * β_eff / (1 - ᾱ_t)  [DDPM posterior var]
        if t_prev > 0:
            beta_tilde_eff = ((1.0 - ab_tprev) * beta_eff
                              / (1.0 - ab_t).clamp(min=1e-8))
        else:
            beta_tilde_eff = torch.zeros(1, device=device).squeeze()       # deterministic

        rho_bar_t     = self.rho_bar_t[t_val - 1]
        rho_bar_tprev = self.rho_bar_t[t_prev - 1] if t_prev > 0 \
                        else torch.ones(1, device=device).squeeze()

        return ab_t, ab_tprev, beta_eff, alpha_eff, beta_tilde_eff, \
               rho_bar_t, rho_bar_tprev

    @torch.no_grad()
    def sample(self, num_samples, inference_steps=None):
        """Generate graphs via reverse diffusion with optional DDPM skip schedule.

        inference_steps: runtime override for self.inference_steps.
          None → use self.inference_steps (set at construction time).
          K    → run K uniformly-spaced reverse steps regardless of training config.

        With effective inference_steps=None or =T: full T-step chain (Algorithm 6).
        With effective inference_steps=K < T: DDPM skip with exact posterior.

        Returns a PyG Batch with ``edge_index`` set so ``to_data_list()`` works.
        """
        # Runtime override wins over construction-time default
        if inference_steps is not None:
            _saved = self.inference_steps
            self.inference_steps = inference_steps
        device = self.device
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph = batched_graph.to(device)

        B       = batched_graph.num_graphs
        N       = batched_graph.num_nodes
        full_ei = batched_graph.full_edge_index                             # (2, E_full)
        E_full  = full_ei.shape[1]

        # ── Initialise G_T ────────────────────────────────────────────────────
        init_mask     = torch.bernoulli(
            torch.full((E_full,), self.sample_rho, device=device)).bool()
        cur_surv_mask = init_mask
        cur_surv_ei   = full_ei[:, cur_surv_mask]
        cur_x         = torch.randn(N, self.d_x, device=device)
        cur_ef        = (torch.randn(cur_surv_ei.shape[1], self.d_f, device=device)
                         if cur_surv_ei.shape[1] > 0
                         else torch.zeros(0, self.d_f, device=device))

        tau       = 0.5
        step_seq  = self._build_skip_schedule()
        n_steps   = len(step_seq)

        for step_i, (t_val, t_prev) in enumerate(step_seq):
            t_graph = torch.full((B,), t_val, dtype=torch.long, device=device)
            t_node  = t_graph.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge  = t_graph.repeat_interleave(batched_graph.edges_per_graph)

            batched_graph.x_cont          = cur_x
            batched_graph.surv_edge_index = cur_surv_ei
            batched_graph.edge_feat_cont  = cur_ef

            # ── Step 1: f_θ forward ───────────────────────────────────────────
            X_hat, F_hat, p_hat, P_t, H = self._denoise_fn(
                batched_graph, t_node, t_edge)

            # ── DDPM coefficients for this (t_val, t_prev) pair ───────────────
            ab_t, ab_tprev, beta_eff, alpha_eff, beta_tilde_eff, \
                rho_bar_t, rho_bar_tprev = self._ddpm_coefficients(t_val, t_prev)

            # ── Step 2: structural reverse ────────────────────────────────────
            # Fraction of edges deleted between t_prev and t:
            # (1 - ρ̄_t / ρ̄_{t_prev}).  For the full schedule t_prev=t-1 this
            # reduces to the original (1 - ρ̄_t) heuristic when ρ̄_{t-1}≈1.
            del_frac = (1.0 - rho_bar_t / rho_bar_tprev.clamp(min=1e-8)).clamp(0.0, 1.0)
            r = torch.where(
                cur_surv_mask,
                p_hat,                      # surviving edge: use model
                p_hat * del_frac,           # deleted-but-reachable: scale by interval
            )
            # clamp() does NOT sanitize NaN (NaN fails both comparisons and passes
            # through unchanged), and torch.bernoulli CUDA-asserts on any non-finite
            # probability. nan_to_num first, then clamp, makes r always valid input.
            r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            if t_prev > 0:
                new_surv_mask = torch.bernoulli(r).bool()
            else:
                new_surv_mask = (p_hat >= tau)                              # deterministic

            new_surv_ei = full_ei[:, new_surv_mask]

            # ── Step 3: node feature DDPM posterior ──────────────────────────
            # μ_X = (√ᾱ_{prev}·β_eff)/(1-ᾱ_t)·X̂_0
            #      + (√α_eff·(1-ᾱ_{prev}))/(1-ᾱ_t)·X_t
            denom_x = (1.0 - ab_t).clamp(min=1e-8)
            mu_X    = (ab_tprev.sqrt() * beta_eff / denom_x) * X_hat \
                    + (alpha_eff.sqrt() * (1.0 - ab_tprev) / denom_x) * cur_x
            noise_X = torch.randn_like(cur_x) if t_prev > 0 \
                      else torch.zeros_like(cur_x)
            new_x   = mu_X + beta_tilde_eff.sqrt() * noise_X

            # ── Step 4: edge feature DDPM posterior ──────────────────────────
            if new_surv_ei.shape[1] > 0:
                was_surv   = cur_surv_mask[new_surv_mask]
                surv_local = (torch.cumsum(cur_surv_mask.long(), 0) - 1).clamp(min=0)
                idx_in_cur = surv_local[new_surv_mask]
                noise_F    = (torch.randn(new_surv_ei.shape[1], self.d_f, device=device)
                              if t_prev > 0
                              else torch.zeros(new_surv_ei.shape[1], self.d_f, device=device))

                if was_surv.any() and F_hat.shape[0] > 0:
                    idx_s   = idx_in_cur.clamp(0, F_hat.shape[0] - 1)
                    mu_F    = (ab_tprev.sqrt() * beta_eff / denom_x) * F_hat[idx_s] \
                            + (alpha_eff.sqrt() * (1.0 - ab_tprev) / denom_x) * cur_ef[idx_s]
                    ef_surv = mu_F + beta_tilde_eff.sqrt() * noise_F
                else:
                    ef_surv = noise_F

                ef_recov = (1.0 - ab_tprev).sqrt() * noise_F
                new_ef   = torch.where(was_surv.unsqueeze(-1), ef_surv, ef_recov)
            else:
                new_ef = torch.zeros(0, self.d_f, device=device)

            # ── Step 5: discriminator guidance ────────────────────────────────
            if t_prev > 0 and self.discriminator is not None and self.disc_lambda > 0:
                new_x_g  = new_x.detach().requires_grad_(True)
                new_ef_g = (new_ef.detach().requires_grad_(True)
                            if new_ef.shape[0] > 0 else new_ef)
                with torch.enable_grad():
                    s     = self.discriminator(new_x_g, new_surv_ei, new_ef_g,
                                               batched_graph.batch, N)
                    torch.log(s.clamp(min=_eps)).sum().backward()
                grad_x = torch.nan_to_num(new_x_g.grad, nan=0.0).clamp(-1., 1.)
                new_x  = new_x_g.detach() + self.disc_lambda * grad_x
                if new_ef.shape[0] > 0 and new_ef_g.grad is not None:
                    grad_f = torch.nan_to_num(new_ef_g.grad, nan=0.0).clamp(-1., 1.)
                    new_ef = new_ef_g.detach() + self.disc_lambda * grad_f

            # ── Step 6: update state ──────────────────────────────────────────
            # Sanitize + bound the magnitude of the features fed into the NEXT
            # iteration. Over a 1000-step autoregressive rollout, the model only
            # ever trained on single forward-noised steps (never its own chained
            # output) can drift into extreme values it was never trained on
            # (exposure bias); without this guard the drift eventually produces
            # NaN/Inf that crashes torch.bernoulli several steps later.
            cur_surv_mask = new_surv_mask
            cur_surv_ei   = new_surv_ei
            cur_x  = torch.nan_to_num(new_x.detach(),  nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            new_ef = new_ef.detach() if new_ef.requires_grad else new_ef
            cur_ef = torch.nan_to_num(new_ef, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

            if t_val % 100 == 0 or t_val == 1:
                n_edges = cur_surv_ei.shape[1]
                print(f'  Sample t={t_val:4d}  |E_t|={n_edges}', end='\r')

        print()

        # ── Build final PyG Batch (edge_index ← E_0) ─────────────────────────
        batched_graph.edge_index = cur_surv_ei

        if cur_surv_ei.shape[1] > 0:
            edge_batch  = batched_graph.batch[cur_surv_ei[0]]
            edge_counts = _scatter_base(
                torch.ones_like(edge_batch), edge_batch, dim_size=B)
        else:
            edge_counts = torch.zeros(B, dtype=torch.long, device=device)

        edge_slice = F.pad(edge_counts, (1, 0), value=0)
        edge_slice = torch.cumsum(edge_slice, dim=0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index']   = batched_graph._inc_dict['full_edge_index']

        # Restore construction-time inference_steps if it was overridden
        if inference_steps is not None:
            self.inference_steps = _saved

        return batched_graph
