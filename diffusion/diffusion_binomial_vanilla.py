import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
# from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import scatter
from diffusion.diffusion_base import *
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


class BinomialDiffusionVanilla(DiffusionBase):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                 loss_type='vb_kl', parametrization='x0', final_prob_node=None, final_prob_edge=None, sample_time_method='importance',
                 noise_schedule=cosine_beta_schedule, device='cuda',
                 discriminator=None, disc_lambda=0.0,
                 lambda_ce=0.0, ce_pos_weight=True, kl_pos_weight=True,
                 sample_rho=None, edge_fraction=1.0,
                 # ── Partial-absorbing (PA-DACA) feature ──────────────────────
                 # alpha_min: when set, activates 3-class absorbing edge forward
                 # process. None = standard binary marginal mode (default).
                 alpha_min=None, daca_gamma=1.0, beta_max=3.0, lambda_LT=1.0):
        super(BinomialDiffusionVanilla, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn,
                                                    timesteps, sample_time_method, device)

        # SparseDiff λ: fraction of non-edge candidate pairs scored per training
        # step. All true edges are always included; sampled non-edges are
        # reweighted by 1/λ for an unbiased (Horvitz-Thompson) loss estimate.
        # 1.0 (default) = no subsampling, full N(N-1)/2 candidate scope.
        assert 0.0 < edge_fraction <= 1.0, '--edge_fraction must be in (0, 1]'
        self.edge_fraction = edge_fraction

        log_final_prob_node = torch.tensor(final_prob_node)[None, :].log()
        log_final_prob_edge = torch.tensor(final_prob_edge)[None, :].log()
        
        self.loss_type = loss_type
        self.parametrization = parametrization
        alphas = noise_schedule(timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_final_prob_node', log_final_prob_node.float())
        self.register_buffer('log_final_prob_edge', log_final_prob_edge.float())

        # sample_rho: if set, sampling initialises from Bernoulli(sample_rho) instead
        # of final_prob_edge. Useful for GraphDenoising / GIN to start the reverse chain
        # from an explicit rho-density random graph independently of the noise schedule.
        if sample_rho is not None:
            assert 0.0 < sample_rho < 1.0, '--sample_rho must be in (0, 1)'
            _slog = torch.tensor([1.0 - sample_rho, sample_rho])[None, :].log()
            self.register_buffer('sample_log_final_prob_edge', _slog.float())
        else:
            self.sample_log_final_prob_edge = None

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

        # discriminator guidance (optional)
        self.discriminator = discriminator
        self.disc_lambda   = disc_lambda

        # CE / KL auxiliary loss
        self.lambda_ce      = lambda_ce
        self.ce_pos_weight  = ce_pos_weight
        # P1+P2 fix: w+ = (1-ρ)/ρ reweighting in the VB-KL loss.
        # Removes the degenerate constant-predictor equilibrium (P1) and
        # restores SNR from O(ρ) to O(1) (P2). Default True.
        self.kl_pos_weight  = kl_pos_weight

        # ── Partial-absorbing (PA-DACA) feature ──────────────────────────────
        # When alpha_min is not None, the edge forward process uses a 3-class
        # absorbing state space {0=non-edge, 1=edge, 2=M} with floored cosine
        # schedule ᾱ_t = alpha_min + (1-alpha_min)·f_cos(t/T).
        # Addresses P1/P2/P3 simultaneously via:
        #   P1: β_ij correlation + w+^M(t) from masked pool
        #   P2: degree-aware masking concentrates edges in masked pool
        #   P3: partial-absorbing prior + L_T trained explicitly
        self.alpha_min  = alpha_min
        self.daca_gamma = daca_gamma
        self.beta_max   = beta_max
        self.lambda_LT  = lambda_LT

        if alpha_min is not None:
            assert 0.0 < alpha_min < 1.0, '--alpha_min must be in (0, 1)'
            from diffusion.diffusion_partial_absorbing import _floored_cosine_cumprod
            _log_ab, _log_ab_prev = _floored_cosine_cumprod(timesteps, alpha_min)
            self.register_buffer('log_cumprod_alpha_abs',
                                 torch.tensor(_log_ab).float())
            self.register_buffer('log_cumprod_alpha_abs_prev',
                                 torch.tensor(_log_ab_prev).float())
            self._abs_edge_classes = 3   # {0=non-edge, 1=edge, 2=M}

    def p_sample(self, batched_graph, t_node, t_edge,
                 t_prev_node=None, t_prev_edge=None):
        """One reverse-diffusion step, optionally steered by discriminator guidance.

        t_prev_node / t_prev_edge: explicit previous timestep for DDPM skip.
        None → t-1 (standard).
        """
        if self.alpha_min is not None:
            return self._absorbing_p_sample(batched_graph, t_node, t_edge,
                                            t_prev_node, t_prev_edge)

        with torch.no_grad():
            log_model_prob_node, log_model_prob_edge = self._p_pred(
                batched_graph, t_node=t_node, t_edge=t_edge,
                t_prev_node=t_prev_node, t_prev_edge=t_prev_edge)

        if self.discriminator is not None and self.disc_lambda > 0:
            try:
                with torch.enable_grad():
                    log_node_in = batched_graph.log_node_attr_t.detach().float().requires_grad_(True)
                    log_edge_in = batched_graph.log_full_edge_attr_t.detach().float().requires_grad_(True)
                    score = self.discriminator(batched_graph, log_node_in, log_edge_in)  # (B,)
                    torch.log(score.clamp(min=1e-8)).sum().backward()
                    node_grad = torch.nan_to_num(log_node_in.grad.detach(), nan=0.0).clamp(-1.0, 1.0)
                    edge_grad = torch.nan_to_num(log_edge_in.grad.detach(), nan=0.0).clamp(-1.0, 1.0)

                log_model_prob_node = log_model_prob_node + self.disc_lambda * node_grad
                log_model_prob_edge = log_model_prob_edge + self.disc_lambda * edge_grad
                # re-normalise in log-space
                log_model_prob_node = log_model_prob_node - log_model_prob_node.logsumexp(1, keepdim=True)
                log_model_prob_edge = log_model_prob_edge - log_model_prob_edge.logsumexp(1, keepdim=True)
            except Exception:
                pass  # guidance failed; fall back to unguided sample

        with torch.no_grad():
            log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)
            log_out_edge = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)
        return log_out_node, log_out_edge

    def disc_loss(self, batched_graph):
        """Discriminator training loss — Diffusion-GAN principle (Wang et al. 2022).

        Real and fake are presented at the SAME noise level t, so the
        discriminator must learn actual real-vs-generated structure rather than
        the (much easier, uninformative) "which of two adjacent noise levels"
        question the previous formulation collapsed to:

          real_t ~ q(x_t | x_0)     true forward process applied to the real graph
          x̂_0    = softmax(f_θ(x_t))  model's own x0-prediction (no_grad: D-only step)
          fake_t ~ q(x_t | x̂_0)     SAME mixture formula, mixing in x̂_0 instead of x_0

        x̂_0 is mixed in as a soft distribution (not hard-sampled) so the fake
        branch isn't artificially sharper than the real branch.
        """
        b = batched_graph.num_graphs
        t, _ = self._sample_time(b, self.device, 'uniform')

        t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
        t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

        batched_graph.log_node_attr      = index_to_log_onehot(batched_graph.node_attr,      self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        # real: q(x_t | x_0)
        log_node_t, log_edge_t = self.q_sample(batched_graph, t_node, t_edge)
        batched_graph.log_node_attr_t      = log_node_t
        batched_graph.log_full_edge_attr_t = log_edge_t
        score_real = self.discriminator(
            batched_graph, log_node_t.detach(), log_edge_t.detach(), sparse=True)

        # fake: x̂_0 = f_θ(x_t) under no_grad (D-only step), then q(x_t | x̂_0)
        with torch.no_grad():
            log_pred_node, log_pred_edge = self._predict_x0_or_xtmin1(
                batched_graph, t_node, t_edge)
            log_node_fake_pred, log_edge_fake_pred = self._q_pred(
                batched_graph, t_node, t_edge,
                log_node_start=log_pred_node, log_edge_start=log_pred_edge)
            log_node_fake_t = self.log_sample_categorical(log_node_fake_pred, self.num_node_classes)
            log_edge_fake_t = self.log_sample_categorical(log_edge_fake_pred, self.num_edge_classes)

        batched_graph.log_node_attr_t      = log_node_fake_t
        batched_graph.log_full_edge_attr_t = log_edge_fake_t
        score_fake = self.discriminator(
            batched_graph, log_node_fake_t.detach(), log_edge_fake_t.detach(), sparse=True)

        L_D = (-torch.log(score_real.clamp(min=1e-8)).mean()
               - torch.log((1. - score_fake).clamp(min=1e-8)).mean())
        self._disc_components = {
            'disc_real': score_real.mean().item(),
            'disc_fake': score_fake.mean().item(),
        }
        return L_D

    # ─────────────────────────────────────────────────────────────────────────
    # Auxiliary CE and KL losses
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_CE(self, batched_graph, t_node, t_edge):
        """Cross-entropy between model's direct x̂₀ prediction and true x₀.

        Only valid for parametrization='x0' (model outputs x₀ directly).

        Edge CE (class-balance weighted binary cross-entropy)
        ──────────────────────────────────────────────────────
          For edge e in graph g with true label x₀[e] ∈ {0,1}:

            CE_e = −[ x₀[e] · w₊[g] · log pθ(1|xₜ)[e]
                    + (1−x₀[e])    · log pθ(0|xₜ)[e] ]

            w₊[g] = |E_neg[g]| / max(|E_pos[g]|, 1)

          This mirrors PyTorch's BCEWithLogitsLoss(pos_weight=…) and
          compensates for the severe class imbalance in sparse graphs.

        Node feature CE (categorical cross-entropy)
        ────────────────────────────────────────────
          CE_n = −Σ_c x₀_onehot[n,c] · log pθ(c|xₜ)[n]

        Returns
        -------
        ce_edge : (B,) tensor — summed weighted edge CE per graph
        ce_node : (B,) tensor — summed node-feature CE per graph
        """
        assert self.parametrization == 'x0', \
            "_compute_CE requires parametrization='x0'"

        log_pred_node, log_pred_edge = self._predict_x0_or_xtmin1(
            batched_graph, t_node, t_edge)                       # (N,Cn), (E,2)

        batch_edge = batched_graph.batch[batched_graph.full_edge_index[0]]  # (E,)
        x0_label   = batched_graph.log_full_edge_attr.argmax(dim=1).float() # (E,) ∈{0,1}

        # ── Per-graph positive-class weight ──────────────────────────────────
        if self.ce_pos_weight:
            n_pos       = scatter(x0_label, batch_edge, dim=0,
                                  dim_size=batched_graph.num_graphs, reduce='sum')  # (B,)
            total_cands = getattr(batched_graph, 'total_cands_per_graph',
                                  batched_graph.edges_per_graph).float().to(self.device)
            n_neg       = total_cands - n_pos
            # Exact (1-ρ)/ρ — no cap; clip_norm controls gradient magnitude.
            w_pos   = n_neg / n_pos.clamp(min=1.0)                            # (B,)
            w_edge  = w_pos[batch_edge]                                        # (E,)
        else:
            w_edge = torch.ones(x0_label.shape[0], device=self.device)

        # ── Weighted binary CE per edge ───────────────────────────────────────
        # CE_e = −[ x₀·w₊·log pθ(1) + (1−x₀)·log pθ(0) ]
        ce_e = -(x0_label * w_edge          * log_pred_edge[:, 1]
                 + (1.0 - x0_label)         * log_pred_edge[:, 0])           # (E,)
        ce_edge = scatter(ce_e, batch_edge, dim=0,
                          dim_size=batched_graph.num_graphs, reduce='sum')    # (B,)

        # ── Categorical CE per node ───────────────────────────────────────────
        # CE_n = −Σ_c x₀_onehot[n,c]·log pθ(c|xₜ)[n]
        ce_n    = -log_categorical(batched_graph.log_node_attr, log_pred_node) # (N,)
        ce_node = scatter(ce_n, batched_graph.batch, dim=0,
                          dim_size=batched_graph.num_graphs, reduce='sum')    # (B,)

        return ce_edge, ce_node

    def _compute_KL_edge_node(self, batched_graph, t_edge, t_node):
        """KL divergence between true posterior and model, split into edge / node.

        KL_edge = Σ_e KL( q(x_{t-1}^e | xₜ^e, x₀^e)  ‖  pθ(x_{t-1}^e | xₜ) )
        KL_node = Σ_n KL( q(x_{t-1}^n | xₜ^n, x₀^n)  ‖  pθ(x_{t-1}^n | xₜ) )

        At t=0 the KL is undefined (x_{t-1} would be x_{-1}), so it is
        replaced by the exact reconstruction NLL:
            NLL_edge = −Σ_e log pθ(x₀[e] | x₁)
            NLL_node = −Σ_n log pθ(x₀[n] | x₁)

        The caller selects between KL and NLL via a binary mask (t == 0).

        Returns (B,) tensors: kl_edge, kl_node, nll_edge, nll_node
        """
        batch_edge = batched_graph.batch[batched_graph.full_edge_index[0]]

        log_true_node = self._q_posterior(
            log_x_start=batched_graph.log_node_attr,
            log_x_t=batched_graph.log_node_attr_t,
            t=t_node, log_final_prob=self.log_final_prob_node)               # (N, Cn)
        log_true_edge = self._q_posterior(
            log_x_start=batched_graph.log_full_edge_attr,
            log_x_t=batched_graph.log_full_edge_attr_t,
            t=t_edge, log_final_prob=self.log_final_prob_edge)               # (E, 2)

        log_model_node, log_model_edge = self._p_pred(
            batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        # KL per element → sum per graph
        kl_node = scatter(self.multinomial_kl(log_true_node, log_model_node),
                          batched_graph.batch, dim=0,
                          dim_size=batched_graph.num_graphs, reduce='sum')   # (B,)
        kl_edge = scatter(self.multinomial_kl(log_true_edge, log_model_edge),
                          batch_edge, dim=0,
                          dim_size=batched_graph.num_graphs, reduce='sum')   # (B,)

        # Reconstruction NLL (used when t=0)
        nll_node = scatter(
            -log_categorical(batched_graph.log_node_attr, log_model_node),
            batched_graph.batch, dim=0,
            dim_size=batched_graph.num_graphs, reduce='sum')                 # (B,)
        nll_edge = scatter(
            -log_categorical(batched_graph.log_full_edge_attr, log_model_edge),
            batch_edge, dim=0,
            dim_size=batched_graph.num_graphs, reduce='sum')                 # (B,)

        return kl_edge, kl_node, nll_edge, nll_node

    def _q_posterior(self, log_x_start, log_x_t, t, log_final_prob, t_prev=None):
        """Compute q(x_{t_prev} | x_t, x_0) for parametrization='x0'.

        t_prev=None  →  t_prev = t-1  (standard DDPM).
        t_prev given →  use that explicit previous timestep (DDPM skip schedule).
        This lets the discrete model sample with fewer than T reverse steps
        by computing the exact posterior for any (t, t_prev) pair.
        """
        assert log_x_start.shape[1] == 2, f'num_class > 2 not supported'

        if t_prev is None:
            tmin1 = t - 1
        else:
            tmin1 = t_prev
        # Remove negative values, will not be used anyway for final decoder
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)
        log_p1 = log_final_prob[:,1]
        log_1_min_p1 = log_final_prob[:,0]

        log_x_start_real = log_x_start[:,1]
        log_1_min_x_start_real = log_x_start[:,0]

        log_x_t_real = log_x_t[:,1]
        log_1_min_x_t_real = log_x_t[:,0]

        log_alpha_t = extract(self.log_alpha, t, log_x_start_real.shape)
        log_beta_t = extract(self.log_1_min_alpha, t, log_x_start_real.shape)
        
        log_cumprod_alpha_tmin1 = extract(self.log_cumprod_alpha, tmin1, log_x_start_real.shape)

        log_1_min_cumprod_alpha_tmin1 = extract(self.log_1_min_cumprod_alpha, tmin1, log_x_start_real.shape)


        log_xtmin1_eq_0_given_x_t = log_add_exp(log_beta_t+log_p1+log_x_t_real, log_1_min_a(log_beta_t+log_p1)+log_1_min_x_t_real)
        log_xtmin1_eq_1_given_x_t = log_add_exp(log_add_exp(log_alpha_t, log_beta_t+log_p1) + log_x_t_real,
                    log_beta_t + log_1_min_p1 + log_1_min_x_t_real)

        log_xtmin1_eq_0_given_x_start = log_add_exp(log_1_min_cumprod_alpha_tmin1+log_1_min_p1, log_cumprod_alpha_tmin1 + log_1_min_x_start_real)
        log_xtmin1_eq_1_given_x_start = log_add_exp(log_cumprod_alpha_tmin1 + log_x_start_real, log_1_min_cumprod_alpha_tmin1+log_p1)

        log_xt_eq_0_given_xt_x_start = log_xtmin1_eq_0_given_x_t + log_xtmin1_eq_0_given_x_start
        log_xt_eq_1_given_xt_x_start = log_xtmin1_eq_1_given_x_t + log_xtmin1_eq_1_given_x_start

        unnorm_log_probs = torch.stack([log_xt_eq_0_given_xt_x_start, log_xt_eq_1_given_xt_x_start], dim=1)
        log_EV_xtmin_given_xt_given_xstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True)
        return log_EV_xtmin_given_xt_given_xstart


    def _predict_x0_or_xtmin1(self, batched_graph, t_node, t_edge):
        out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge)

        assert out_node.size(1) == self.num_node_classes
        assert out_edge.size(1) == self.num_edge_classes

        log_pred_node = F.log_softmax(out_node, dim=1)
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_node, log_pred_edge

    def _ce_prior(self, batched_graph):
        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()

        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)

        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)


        ce_prior_node = -log_categorical(log_qxT_prob_node, log_final_prob_node)
        ce_prior_node = scatter(ce_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        ce_prior_edge = -log_categorical(log_qxT_prob_edge, log_final_prob_edge)
        ce_prior_edge = scatter(ce_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        ce_prior = ce_prior_node + ce_prior_edge

        return ce_prior

    def _kl_prior(self, batched_graph):
        if self.alpha_min is not None:
            # Absorbing prior KL: L_T = Σ α_min^β · CE(e_0, ρ_ij) (closed-form, ≠0)
            return self._compute_LT(batched_graph)

        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()

        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)

        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)


        kl_prior_node = self.multinomial_kl(log_qxT_prob_node, log_final_prob_node)
        kl_prior_node = scatter(kl_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        kl_prior_edge = self.multinomial_kl(log_qxT_prob_edge, log_final_prob_edge)
        kl_prior_edge = scatter(kl_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        kl_prior = kl_prior_node + kl_prior_edge

        return kl_prior
   
    def _compute_MC_KL(self, batched_graph, t_edge, t_node):
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)
        log_true_prob_node, log_true_prob_edge = self._q_pred_one_timestep(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        cross_ent_node = -log_categorical(batched_graph.log_node_attr_tmin1, log_model_prob_node)
        cross_ent_edge = -log_categorical(batched_graph.log_full_edge_attr_tmin1, log_model_prob_edge)


        ent_node = log_categorical(batched_graph.log_node_attr_t, log_true_prob_node).detach()
        ent_edge = log_categorical(batched_graph.log_full_edge_attr_t, log_true_prob_edge).detach()

        loss_node = cross_ent_node + ent_node
        loss_edge = cross_ent_edge + ent_edge

        loss_node = scatter(loss_node, batched_graph.batch, dim=-1, reduce='sum')
        loss_edge = scatter(loss_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum') 
        loss = loss_node + loss_edge

        return loss

    def _build_edge_query(self, batched_graph, t_edge):
        """SparseDiff-style candidate-pair subsampling for training.

        Two modes depending on whether preprocess() already made full_edge_index
        sparse (edges_per_graph < total_cands_per_graph):

        Pre-sparse mode (preprocess sparsified): full_edge_index already contains
        all true edges + λ-sampled non-edges.  Include all of it with HT weight
        1/λ for non-edges.  Avoids double-sampling (which would give prob λ²).

        Dense mode (full_edge_index = full N²/2): subsample on the fly: all true
        edges (weight 1) + λ-fraction of non-edges (weight 1/λ).

        Returns: query_idx (|Q|,), query_weight (|Q|,), query_edge_index (2,|Q|),
        t_edge_q (|Q|,).
        """
        labels = batched_graph.full_edge_attr
        device = labels.device
        ef = self.edge_fraction

        has_total_cands = hasattr(batched_graph, 'total_cands_per_graph')
        pre_sparse = (has_total_cands and
                      (batched_graph.edges_per_graph.sum() <
                       batched_graph.total_cands_per_graph.sum()))

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        if pre_sparse:
            # full_edge_index is already the sparse query — pass through as-is
            query_idx = torch.arange(labels.numel(), device=device)
            neg_mask = (labels == 0)
            query_weight = torch.where(
                neg_mask,
                torch.full_like(labels, 1.0 / ef, dtype=torch.float),
                torch.ones_like(labels, dtype=torch.float))
        else:
            n_query_neg = max(1, int(round(ef * neg_idx.numel())))
            perm = torch.randperm(neg_idx.numel(), device=device)[:n_query_neg]
            sampled_neg_idx = neg_idx[perm]
            query_idx = torch.cat([pos_idx, sampled_neg_idx])
            query_weight = torch.cat([
                torch.ones(pos_idx.numel(), device=device),
                torch.full((sampled_neg_idx.numel(),), 1.0 / ef, device=device),
            ])

        query_edge_index = batched_graph.full_edge_index[:, query_idx]
        t_edge_q = t_edge[query_idx]
        return query_idx, query_weight, query_edge_index, t_edge_q

    def _compute_RB_KL(self, batched_graph, t, t_edge, t_node):
        if self.alpha_min is not None:
            return self._compute_absorbing_ELBO(batched_graph, t, t_edge, t_node)

        log_true_prob_node = self._q_posterior(log_x_start=batched_graph.log_node_attr,
                                            log_x_t=batched_graph.log_node_attr_t, t=t_node, log_final_prob=self.log_final_prob_node)

        # SparseDiff subsampling: training only, never at eval/sampling time —
        # eval-loss (bpd) logging stays on the full set for a low-variance signal.
        if self.training and self.edge_fraction < 1.0:
            query_idx, query_weight, query_edge_index, t_edge_q = \
                self._build_edge_query(batched_graph, t_edge)
            batch_edge = batched_graph.batch[query_edge_index[0]]
            log_full_edge_attr_q   = batched_graph.log_full_edge_attr[query_idx]
            log_full_edge_attr_t_q = batched_graph.log_full_edge_attr_t[query_idx]
            log_true_prob_edge = self._q_posterior(
                log_x_start=log_full_edge_attr_q, log_x_t=log_full_edge_attr_t_q,
                t=t_edge_q, log_final_prob=self.log_final_prob_edge)
            log_model_prob_node, log_model_prob_edge = self._p_pred(
                batched_graph=batched_graph, t_node=t_node, t_edge=t_edge_q,
                query_edge_index=query_edge_index, log_edge_t_override=log_full_edge_attr_t_q)
        else:
            batch_edge = batched_graph.batch[batched_graph.full_edge_index[0]]
            query_weight = 1.0
            log_full_edge_attr_q = batched_graph.log_full_edge_attr
            log_true_prob_edge = self._q_posterior(log_x_start=batched_graph.log_full_edge_attr,
                                                log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge)
            log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        # ── P1 + P2 fix: positive-class reweighting w+ = (1-ρ)/ρ ────────────
        # P1: without reweighting the unweighted KL has a degenerate local minimum
        # at p* = ρ (constant predictor) regardless of θ.  w+ removes it.
        # P2: unweighted gradient is dominated by non-edge pairs (e.g. 97.8% for
        # Polblogs), suppressing the edge-class signal by a factor of ρ.  w+
        # equalises the aggregate edge vs. non-edge gradient contribution.
        #
        # Critically: n_pos/n_neg are counted over the FULL candidate set, not the
        # query subset.  With edge_fraction=0.2 the query holds 100% of true edges
        # but only 20% of non-edges; a query-only count would give the wrong ratio.
        #
        # No artificial cap: we use the exact theoretical value (1-ρ)/ρ and rely
        # on clip_norm to bound the gradient step.  A cap at 100 would leave
        # CiteSeer (true w+≈610) and Pubmed (true w+≈4385) severely under-corrected.
        if self.kl_pos_weight:
            x0_label_q    = log_full_edge_attr_q.argmax(dim=-1).float()       # (|Q|,)
            x0_label_full = batched_graph.log_full_edge_attr.argmax(dim=-1).float()
            batch_edge_full = batched_graph.batch[batched_graph.full_edge_index[0]]
            n_pos_full = scatter(x0_label_full, batch_edge_full, dim=0,
                                 dim_size=batched_graph.num_graphs, reduce='sum')  # (B,)
            total_cands = getattr(batched_graph, 'total_cands_per_graph',
                                  batched_graph.edges_per_graph).float().to(self.device)
            n_neg_full = total_cands - n_pos_full
            w_pos_full = n_neg_full / n_pos_full.clamp(min=1.0)               # (B,) = (1-ρ)/ρ
            kl_pos_w = w_pos_full[batch_edge] * x0_label_q + (1.0 - x0_label_q)
        else:
            x0_label_q = log_full_edge_attr_q.argmax(dim=-1).float()
            kl_pos_w = torch.ones(x0_label_q.shape[0], device=self.device)

        kl_node = self.multinomial_kl(log_true_prob_node, log_model_prob_node)
        kl_node = scatter(kl_node, batched_graph.batch, dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)

        # Compute raw (unweighted) KL once — reused for both the weighted loss and diagnostics.
        kl_per_pair_raw = self.multinomial_kl(log_true_prob_edge, log_model_prob_edge) * query_weight

        # ── Proposition diagnostics (training only, no grad) ──────────────────
        if self.training:
            with torch.no_grad():
                # P1: predicted edge probability distribution — constant predictor
                #     shows as mean≈ρ with std≈0.
                pred_prob_edge = log_model_prob_edge.exp()[:, 1]            # (|Q|,)
                # P2: raw KL contribution split — edges vs. non-edges before w+.
                #     In an unweighted sparse graph kl_frac_edge ≈ ρ (signal suppressed).
                #     After w+ the weighted contributions equalise toward 0.5.
                kl_e_sum = (kl_per_pair_raw * x0_label_q).sum().item()
                kl_n_sum = (kl_per_pair_raw * (1. - x0_label_q)).sum().item()
                kl_total  = kl_e_sum + kl_n_sum + 1e-12
                w_pos_val = (w_pos_full.mean().item()
                             if self.kl_pos_weight else 1.0)                # (B,) mean
                # P3: empirical density of noisy sample x_t.
                #     Marginal transition → xt_density ≈ ρ for all t.
                #     Absorbing (m1=0) → xt_density decays to 0 as t→T.
                xt_density = batched_graph.log_full_edge_attr_t.exp()[:, 1].mean().item()
                self._prop_diagnostics = {
                    'p1/pred_density_mean': pred_prob_edge.mean().item(),
                    'p1/pred_density_std':  pred_prob_edge.std().item(),
                    'p2/kl_frac_edge_raw':  kl_e_sum / kl_total,
                    'p2/w_pos_mean':        w_pos_val,
                    'p3/xt_density':        xt_density,
                    'p3/stationary_target': self.log_final_prob_edge.exp()[0, 1].item(),
                }

        kl_edge = kl_per_pair_raw * kl_pos_w
        kl_edge = scatter(kl_edge, batch_edge, dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)
        kl = kl_node + kl_edge

        decoder_nll_node = -log_categorical(batched_graph.log_node_attr, log_model_prob_node)
        decoder_nll_node = scatter(decoder_nll_node, batched_graph.batch, dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)
        decoder_nll_edge = -log_categorical(log_full_edge_attr_q, log_model_prob_edge) * query_weight * kl_pos_w
        decoder_nll_edge = scatter(decoder_nll_edge, batch_edge, dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)
        decoder_nll = decoder_nll_node + decoder_nll_edge

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def _q_sample_and_set_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        if self.alpha_min is not None:
            # Absorbing forward: edges → 3-class {0,1,M}, nodes → standard marginal
            beta_full = self._beta_ij(batched_graph, batched_graph.full_edge_index)
            e_t = self._q_sample_edge_abs(batched_graph.full_edge_attr, t_edge, beta_full)
            batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_t, self._abs_edge_classes)
            # Nodes: use standard marginal q_sample (reads log_node_attr, returns node only)
            log_node_t, _ = self.q_sample(batched_graph, t_node, t_edge)
            batched_graph.log_node_attr_t = log_node_t
            return

        log_node_attr_t, log_full_edge_attr_t = self.q_sample(batched_graph, t_node, t_edge)
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t
        

    def _q_sample_and_set_xtmin1_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        # sample xt-1
        tmin1_node = t_node - 1
        tmin1_edge = t_edge - 1
        tmin1_node_clamped = torch.where(tmin1_node < 0, torch.zeros_like(tmin1_node), tmin1_node)
        tmin1_edge_clamped = torch.where(tmin1_edge < 0, torch.zeros_like(tmin1_edge), tmin1_edge)
        
        log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.q_sample(batched_graph, tmin1_node_clamped, tmin1_edge_clamped)
        batched_graph.log_node_attr_tmin1 = log_node_attr_tmin1
        batched_graph.log_full_edge_attr_tmin1 = log_full_edge_attr_tmin1

        batched_graph.log_node_attr_tmin1[tmin1_node<0] = batched_graph.log_node_attr[tmin1_node<0]
        batched_graph.log_full_edge_attr_tmin1[tmin1_edge<0] = batched_graph.log_full_edge_attr[tmin1_edge<0]

        # sample xt given xt-1
        log_node_attr_t, log_full_edge_attr_t = self._q_sample_one_timestep(batched_graph, t_node, t_edge)
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t


    def _q_sample_one_timestep(self, batched_graph, t_node, t_edge):
        log_prob_node, log_prob_edge = self._q_pred_one_timestep(batched_graph, t_node, t_edge)

        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge  

        # xt ~ q(xt|xtmin1)

    def _q_pred_one_timestep(self, batched_graph, t_node, t_edge):
        log_alpha_t_node = extract(self.log_alpha, t_node, batched_graph.log_node_attr.shape)
        log_1_min_alpha_t_node = extract(self.log_1_min_alpha, t_node, batched_graph.log_node_attr.shape)

        log_prob_nodes = log_add_exp(
            batched_graph.log_node_attr_tmin1 + log_alpha_t_node,
            log_1_min_alpha_t_node + self.log_final_prob_node
        )

        log_alpha_t_edge = extract(self.log_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_alpha_t_edge = extract(self.log_1_min_alpha, t_edge, batched_graph.log_full_edge_attr.shape)

        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr_tmin1 + log_alpha_t_edge,
            log_1_min_alpha_t_edge + self.log_final_prob_edge
        )

        return log_prob_nodes, log_prob_edges 

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0] + batched_graph.node_attr.shape[0]

    def _sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self._sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            # NaN/Inf can enter Lt_history if the loss spikes; treat those steps uniformly.
            Lt_sqrt = torch.nan_to_num(Lt_sqrt, nan=0.0001, posinf=1.0, neginf=0.0001)
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            if torch.isnan(pt_all).any() or pt_all.sum() <= 0:
                return self._sample_time(b, device, method='uniform')

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            length = self.Lt_count.shape[0]
            t = torch.randint(0, length, (b,), device=device).long()

            pt = torch.ones_like(t).float() / length
            return t, pt
        else:
            raise ValueError 
    

    def _q_pred(self, batched_graph, t_node, t_edge, log_node_start=None, log_edge_start=None):
        """Compute q(x_t | x_start) = Cat(ᾱ_t · x_start + (1-ᾱ_t) · m).

        log_node_start / log_edge_start: optional explicit log-distributions to
        mix in place of x_0 (e.g. the model's own soft x̂_0 prediction, used by
        disc_loss's Diffusion-GAN fake-sample construction). Defaults to the
        true x_0 stored on batched_graph (standard forward-process behaviour).
        """
        if log_node_start is None:
            log_node_start = batched_graph.log_node_attr
        if log_edge_start is None:
            log_edge_start = batched_graph.log_full_edge_attr

        # nodes prob
        log_cumprod_alpha_t_node = extract(self.log_cumprod_alpha, t_node, log_node_start.shape)
        log_1_min_cumprod_alpha_node = extract(self.log_1_min_cumprod_alpha, t_node, log_node_start.shape)
        log_prob_nodes = log_add_exp(
            log_node_start + log_cumprod_alpha_t_node,
            log_1_min_cumprod_alpha_node + self.log_final_prob_node
        )

        # edges prob
        log_cumprod_alpha_t_edge = extract(self.log_cumprod_alpha, t_edge, log_edge_start.shape)
        log_1_min_cumprod_alpha_edge = extract(self.log_1_min_cumprod_alpha, t_edge, log_edge_start.shape)
        log_prob_edges = log_add_exp(
            log_edge_start + log_cumprod_alpha_t_edge,
            log_1_min_cumprod_alpha_edge + self.log_final_prob_edge
        )
        return log_prob_nodes, log_prob_edges


    def _p_pred(self, batched_graph, t_node, t_edge,
                t_prev_node=None, t_prev_edge=None,
                query_edge_index=None, log_edge_t_override=None):
        """Predict p(x_{t_prev} | x_t).

        t_prev_node / t_prev_edge: explicit previous timestep tensors for
        DDPM skip schedules.  None → t-1 (standard behaviour).

        query_edge_index / log_edge_t_override: SparseDiff subsampling support.
        When given, restricts the denoiser's edge-classification head (and the
        posterior computed from it) to a subsampled candidate set rather than
        the full N(N-1)/2 pairs — message passing itself is unaffected, since
        GIN still reads the complete current graph state from batched_graph.
        t_edge must already be sliced to match (len(query_edge_index)).
        """
        if self.parametrization == 'x0':
            if query_edge_index is not None:
                batched_graph.query_edge_index = query_edge_index
            log_node_recon, log_full_edge_recon = self._predict_x0_or_xtmin1(
                batched_graph, t_node=t_node, t_edge=t_edge)
            if query_edge_index is not None:
                batched_graph.query_edge_index = None
            log_edge_t = (log_edge_t_override if log_edge_t_override is not None
                          else batched_graph.log_full_edge_attr_t)
            log_model_pred_node = self._q_posterior(
                log_x_start=log_node_recon,
                log_x_t=batched_graph.log_node_attr_t,
                t=t_node, log_final_prob=self.log_final_prob_node,
                t_prev=t_prev_node)
            log_model_pred_edge = self._q_posterior(
                log_x_start=log_full_edge_recon,
                log_x_t=log_edge_t,
                t=t_edge, log_final_prob=self.log_final_prob_edge,
                t_prev=t_prev_edge)
        elif self.parametrization == 'xt':
            log_model_pred_node, log_model_pred_edge = self._predict_x0_or_xtmin1(
                batched_graph, t_node=t_node, t_edge=t_edge)
        return log_model_pred_node, log_model_pred_edge


    def _prepare_data_for_sampling(self, batched_graph):
        if self.alpha_min is not None:
            return self._prepare_absorbing_sampling(batched_graph)

        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        log_prob_node = torch.ones_like(batched_graph.log_node_attr, device=self.device) * self.log_final_prob_node
        batched_graph.log_node_attr_t = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        # If sample_rho is set, initialise the reverse chain from Bernoulli(sample_rho)
        # rather than from final_prob_edge. This gives GraphDenoising / GIN an explicit
        # rho-density random graph as xT, independently of the noise schedule.
        edge_init_logprob = (self.sample_log_final_prob_edge
                             if self.sample_log_final_prob_edge is not None
                             else self.log_final_prob_edge)
        log_prob_edge = (torch.ones_like(batched_graph.log_full_edge_attr, device=self.device)
                         * edge_init_logprob)
        batched_graph.log_full_edge_attr_t = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return batched_graph

    # ─────────────────────────────────────────────────────────────────────────
    # PA-DACA absorbing helpers  (active only when alpha_min is not None)
    # ─────────────────────────────────────────────────────────────────────────

    def _beta_ij(self, batched_graph, edge_index):
        """β_ij = min(β_max, (d_i*·d_j*/d̄²_g)^(γ/2)).

        Uses target_degree* stored on batched_graph (permanent ground-truth
        degree, never masked). Falls back to β_ij=1 when absent.
        """
        if not hasattr(batched_graph, 'target_degree') or batched_graph.target_degree is None:
            return torch.ones(edge_index.shape[1], device=self.device)
        d_star     = batched_graph.target_degree.float()                         # (N,)
        d_i        = d_star[edge_index[0]]
        d_j        = d_star[edge_index[1]]
        b          = batched_graph.num_graphs
        n_nodes    = batched_graph.nodes_per_graph.float().to(self.device)       # (B,)
        # sum(target_degree) = 2|E| → mean = 2|E|/N = true average degree.
        # Using target_degree (not full_edge_attr) makes this correct during
        # sampling when full_edge_attr is all-zeros (empty graph prior).
        d_bar      = (scatter(d_star, batched_graph.batch, dim=0, dim_size=b,
                              reduce='sum') / n_nodes.clamp(min=1.0)).clamp(min=1.0)
        d_bar_sq   = d_bar ** 2                                                  # (B,)
        batch_q    = batched_graph.batch[edge_index[0]]
        s_ij       = (d_i * d_j / d_bar_sq[batch_q]).clamp(min=1e-6)
        return (s_ij ** (self.daca_gamma / 2.0)).clamp(max=self.beta_max)       # (E,)

    def _alpha_bar_ij(self, t_edge, beta_ij):
        """ᾱ_t^{β_ij}: forward retention probability at time t."""
        return (self.log_cumprod_alpha_abs[t_edge] * beta_ij).exp().clamp(1e-9, 1.0)

    def _alpha_bar_ij_prev(self, t_edge, beta_ij):
        """ᾱ_{t-1}^{β_ij}: previous-step retention (= 1.0 at t_index=0)."""
        return (self.log_cumprod_alpha_abs_prev[t_edge] * beta_ij).exp().clamp(1e-9, 1.0)

    def _unmask_prob(self, t_edge, beta_ij, t_prev_edge=None):
        """r(t→t_prev)^{ij} = (ᾱ_{t_prev}^β - ᾱ_t^β) / (1 - ᾱ_t^β).

        t_prev_edge=None  → single-step-back: ᾱ_{t-1}^β (standard ELBO / training).
        t_prev_edge given → ᾱ_{t_prev+1}^β via schedule lookup (skip-schedule sampling).
        """
        ab_t = self._alpha_bar_ij(t_edge, beta_ij)
        if t_prev_edge is None:
            ab_tm1 = self._alpha_bar_ij_prev(t_edge, beta_ij)  # one-step-back (default)
        else:
            ab_tm1 = self._alpha_bar_ij(t_prev_edge, beta_ij)  # multi-step skip
        return ((ab_tm1 - ab_t).clamp(min=0.0) / (1.0 - ab_t).clamp(min=1e-9)).clamp(0.0, 1.0)

    def _q_sample_edge_abs(self, e0, t_edge, beta_ij):
        """Absorbing forward q(e_t|e_0): retain e_0 with prob ᾱ_t^β, else M=2."""
        ab_t = self._alpha_bar_ij(t_edge, beta_ij)
        mask = (torch.rand(e0.shape[0], device=self.device) > ab_t)            # True → mask
        return torch.where(mask, torch.full_like(e0, 2), e0)                   # (E,) ∈{0,1,2}

    def _compute_LT(self, batched_graph):
        """L_T = Σ_{ij} α_min^{β_ij} · CE(e_0, ρ_ij)  —  (B,) per-graph.

        The prior KL for the partial-absorbing process. Unlike the marginal
        prior (which goes to 0 as ᾱ_T→0), this is non-zero and acts as a
        structural regulariser matching the Chung-Lu scaffold.
        ρ_ij = Chung-Lu pair probability = d_i*·d_j* / (2·|E|_g).
        """
        b        = batched_graph.num_graphs
        full_ei  = batched_graph.full_edge_index
        batch_fe = batched_graph.batch[full_ei[0]]

        with torch.no_grad():
            beta_full       = self._beta_ij(batched_graph, full_ei)
            alpha_min_beta  = self.alpha_min ** beta_full   # (E,) = ᾱ_T^β
            if hasattr(batched_graph, 'target_degree') and batched_graph.target_degree is not None:
                d_star = batched_graph.target_degree.float()
                d_i    = d_star[full_ei[0]]
                d_j    = d_star[full_ei[1]]
                n_pos  = scatter(batched_graph.full_edge_attr.float(),
                                 batch_fe, dim=0, dim_size=b, reduce='sum')
                rho_ij = (d_i * d_j / (2.0 * n_pos[batch_fe]).clamp(min=1.0)).clamp(1e-6, 1.0-1e-6)
            else:
                total_c = getattr(batched_graph, 'total_cands_per_graph',
                                  batched_graph.edges_per_graph).float().to(self.device)
                n_pos   = scatter(batched_graph.full_edge_attr.float(),
                                  batch_fe, dim=0, dim_size=b, reduce='sum')
                rho_ij  = (n_pos / total_c.clamp(min=1.0))[batch_fe].clamp(1e-6, 1.0-1e-6)

        e0 = batched_graph.full_edge_attr.float()
        ce = -(e0 * torch.log(rho_ij) + (1.0 - e0) * torch.log(1.0 - rho_ij))
        return scatter(alpha_min_beta * ce, batch_fe, dim=0, dim_size=b, reduce='sum')

    def _compute_absorbing_ELBO(self, batched_graph, t, t_edge_full, t_node):
        """Absorbing-mode ELBO for edges + standard node KL.

        Edge term:  Σ_{(i,j): e_t=M}  r_t^{ij} · CE(e_0, p̂_θ) · w+^M(t)
        Node term:  standard binary marginal KL (unchanged from non-absorbing)

        Exactly resolves all three failure modes:
          P1: β_ij correlates masking speed with edge identity → no constant-predictor optimum
          P2: w+^M(t) computed from the masked pool → SNR = 1 within that pool
          P3: called via _kl_prior → L_T explicitly minimised
        """
        b = batched_graph.num_graphs

        # ── SparseDiff: build query subset ────────────────────────────────────
        if self.training and self.edge_fraction < 1.0:
            query_idx, query_weight, query_ei, t_edge_q = \
                self._build_edge_query(batched_graph, t_edge_full)
        else:
            E            = batched_graph.full_edge_attr.shape[0]
            query_idx    = torch.arange(E, device=self.device)
            query_weight = torch.ones(E, device=self.device)
            query_ei     = batched_graph.full_edge_index
            t_edge_q     = t_edge_full

        batch_q = batched_graph.batch[query_ei[0]]

        # ── Run denoiser: 2-class x_0 predictions (query edges, all nodes) ───
        batched_graph.query_edge_index = query_ei
        log_pred_node, log_pred_edge_q = self._predict_x0_or_xtmin1(
            batched_graph, t_node, t_edge_q)
        batched_graph.query_edge_index = None

        # ── Absorbing edge ELBO ───────────────────────────────────────────────
        e_t_full  = batched_graph.log_full_edge_attr_t.argmax(-1)           # (E,) ∈{0,1,2}
        e0_q      = batched_graph.full_edge_attr[query_idx]                 # (|Q|,) ∈{0,1}
        e_t_q     = e_t_full[query_idx]                                     # (|Q|,)
        beta_q    = self._beta_ij(batched_graph, query_ei)                  # (|Q|,)
        is_masked = (e_t_q == 2).float()                                    # 1 where M
        r_t_q     = self._unmask_prob(t_edge_q, beta_q)                     # (|Q|,)

        # CE(e_0, p̂): binary cross-entropy using 2-class x_0 prediction
        ce_q = -(e0_q.float() * log_pred_edge_q[:, 1]
                 + (1.0 - e0_q.float()) * log_pred_edge_q[:, 0])            # (|Q|,)

        elbo_pair = r_t_q * ce_q * is_masked * query_weight                 # (|Q|,)

        # ── P2 fix: w+^M(t) from masked pool on FULL edge set ────────────────
        # Counts positives/negatives within the masked pool, not the full set.
        # This ensures SNR=1 within the pool regardless of global ρ.
        if self.kl_pos_weight:
            is_m_full = (e_t_full == 2).float()                             # (E,)
            e0_ff     = batched_graph.full_edge_attr.float()
            batch_fe  = batched_graph.batch[batched_graph.full_edge_index[0]]
            n_pos_m   = scatter(e0_ff * is_m_full,
                                batch_fe, dim=0, dim_size=b, reduce='sum')  # (B,)
            n_neg_m   = scatter((1.0-e0_ff) * is_m_full,
                                batch_fe, dim=0, dim_size=b, reduce='sum')  # (B,)
            w_plus    = n_neg_m / n_pos_m.clamp(min=1.0)                   # (B,) masked-(1-ρ)/ρ
            kl_pos_w  = w_plus[batch_q] * e0_q.float() + (1.0 - e0_q.float())
        else:
            w_plus   = torch.ones(b, device=self.device)
            kl_pos_w = torch.ones(e0_q.shape[0], device=self.device)

        kl_edge = scatter(elbo_pair * kl_pos_w, batch_q, dim=0, dim_size=b, reduce='sum')

        # ── Node KL: standard binary marginal (uses existing _q_posterior) ───
        log_true_node  = self._q_posterior(batched_graph.log_node_attr,
                                            batched_graph.log_node_attr_t,
                                            t=t_node, log_final_prob=self.log_final_prob_node)
        log_model_node = self._q_posterior(log_pred_node,
                                            batched_graph.log_node_attr_t,
                                            t=t_node, log_final_prob=self.log_final_prob_node)
        kl_node = self.multinomial_kl(log_true_node, log_model_node)
        kl_node = scatter(kl_node, batched_graph.batch, dim=0, dim_size=b, reduce='sum')

        # ── Reconstruction NLL at t=0 (L_0) ─────────────────────────────────
        nll_edge_q = ce_q * query_weight * kl_pos_w
        nll_edge   = scatter(nll_edge_q, batch_q, dim=0, dim_size=b, reduce='sum')
        nll_node   = scatter(-log_categorical(batched_graph.log_node_attr, log_model_node),
                             batched_graph.batch, dim=0, dim_size=b, reduce='sum')
        decoder_nll = nll_edge + nll_node

        mask_t0 = (t == 0).float()
        loss    = mask_t0 * decoder_nll + (1.0 - mask_t0) * (kl_edge + kl_node)

        # ── Diagnostics ──────────────────────────────────────────────────────
        if self.training:
            with torch.no_grad():
                self._prop_diagnostics = {
                    'p1/pred_density_mean': log_pred_edge_q.exp()[:, 1].mean().item(),
                    'p1/pred_density_std':  log_pred_edge_q.exp()[:, 1].std().item(),
                    'p2/kl_frac_edge_raw':  ((elbo_pair * e0_q.float()).sum() /
                                              (elbo_pair.sum() + 1e-12)).item(),
                    'p2/w_plus_masked_mean': w_plus.mean().item(),
                    'p3/xt_edge_density':   (e_t_full == 1).float().mean().item(),
                    'p3/xt_mask_fraction':  (e_t_full == 2).float().mean().item(),
                    'p3/alpha_min':         self.alpha_min,
                }
        return loss

    def _absorbing_p_sample(self, batched_graph, t_node, t_edge,
                             t_prev_node=None, t_prev_edge=None):
        """One reverse step for 3-class absorbing edges + standard node marginal.

        Edges:
          e_t ∈ {0,1}: copy deterministically (posterior is a delta)
          e_t = M:     with prob r_t → sample e_0 from p̂_θ; else stay M
        Nodes: standard binary marginal reverse via _q_posterior.
        """
        with torch.no_grad():
            full_ei   = batched_graph.full_edge_index
            beta_full = self._beta_ij(batched_graph, full_ei)

            # x_0 predictions (2-class) from denoiser
            log_pred_node, log_pred_edge = self._predict_x0_or_xtmin1(
                batched_graph, t_node, t_edge)

            # Absorbing edge reverse
            e_t       = batched_graph.log_full_edge_attr_t.argmax(-1)     # (E,) ∈{0,1,2}
            is_masked = (e_t == 2)
            # t_prev_edge=None → final step of sampling → reveal ALL remaining masked.
            # t_prev_edge given → multi-step skip: use ᾱ_{t_prev+1} in numerator
            #   so skip jumps are correctly large (not just one-step probabilities).
            if t_prev_edge is None:
                r_t = torch.ones(e_t.shape[0], device=self.device)
            else:
                r_t = self._unmask_prob(t_edge, beta_full, t_prev_edge=t_prev_edge)

            u       = torch.rand(e_t.shape[0], device=self.device)
            unmask  = is_masked & (u < r_t)
            # Sample e_0 from 2-class model prediction
            e_hat   = self.log_sample_categorical(log_pred_edge, self.num_edge_classes).argmax(-1)
            e_tm1   = torch.where(is_masked,
                                  torch.where(unmask, e_hat, torch.full_like(e_t, 2)),
                                  e_t)
            log_edge_tm1 = index_to_log_onehot(e_tm1, self._abs_edge_classes)

            # Node reverse: standard marginal q_posterior
            log_node_tm1 = self.log_sample_categorical(
                self._q_posterior(log_pred_node, batched_graph.log_node_attr_t,
                                  t=t_node, log_final_prob=self.log_final_prob_node,
                                  t_prev=t_prev_node),
                self.num_node_classes)

        return log_node_tm1, log_edge_tm1

    def _prepare_absorbing_sampling(self, batched_graph):
        """Initialize x_T from partial-absorbing prior p(x_T).

        p(e_{ij,T}=M)   = 1 - α_min^{β_ij}
        p(e_{ij,T}∈{0,1}) = α_min^{β_ij} · Bernoulli(ρ_ij)

        This is the P3 fix: train- and sample-time x_T distributions are
        identical, so L_T terms trained during forward-pass are used at sampling.
        """
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        log_p_node = torch.ones_like(batched_graph.log_node_attr) * self.log_final_prob_node
        batched_graph.log_node_attr_t = self.log_sample_categorical(log_p_node, self.num_node_classes)

        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        b        = batched_graph.num_graphs
        full_ei  = batched_graph.full_edge_index
        batch_fe = batched_graph.batch[full_ei[0]]
        E        = full_ei.shape[1]

        beta_full = self._beta_ij(batched_graph, full_ei)
        ab_T      = (self.alpha_min ** beta_full).clamp(1e-9, 1.0)             # (E,)

        # Chung-Lu ρ_ij = d_i*·d_j* / (2|E|_g) for the scaffold distribution.
        # During sampling full_edge_attr is all-zeros (empty graph), so we must
        # derive 2|E| from target_degree (sum of degrees = 2|E|) not full_edge_attr.
        if hasattr(batched_graph, 'target_degree') and batched_graph.target_degree is not None:
            d_star  = batched_graph.target_degree.float()
            d_i     = d_star[full_ei[0]]
            d_j     = d_star[full_ei[1]]
            # d_sum[g] = sum(d_star for nodes in g) = 2|E|_g
            d_sum   = scatter(d_star, batched_graph.batch, dim=0, dim_size=b, reduce='sum')
            rho_ij  = (d_i * d_j / d_sum[batch_fe].clamp(min=1.0)).clamp(1e-6, 1.0-1e-6)
        else:
            # Fallback: use global density from total candidates (no degree info)
            total_c = getattr(batched_graph, 'total_cands_per_graph',
                              batched_graph.edges_per_graph).float().to(self.device)
            n_pos   = scatter(batched_graph.full_edge_attr.float(),
                              batch_fe, dim=0, dim_size=b, reduce='sum')
            rho_ij  = (n_pos / total_c.clamp(min=1.0))[batch_fe].clamp(1e-6, 1.0-1e-6)

        # Sample: visible with prob ab_T → Bernoulli(ρ_ij); else M
        u_vis  = torch.rand(E, device=self.device)
        u_edge = torch.rand(E, device=self.device)
        e_bern = (u_edge < rho_ij).long()                                      # (E,) ∈{0,1}
        e_T    = torch.where(u_vis < ab_T, e_bern, torch.full_like(e_bern, 2))
        batched_graph.log_full_edge_attr_t = index_to_log_onehot(e_T, self._abs_edge_classes)
        return batched_graph

    def sample(self, num_samples, inference_steps=None):
        """Override DiffusionBase.sample to post-process 3-class absorbing states."""
        if self.alpha_min is None:
            return super().sample(num_samples, inference_steps)

        # Absorbing mode: run chain then convert M → non-edge
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
                batched_graph, t_node, t_edge, tp_node, tp_edge)
            batched_graph.log_node_attr_t      = log_node_tm1
            batched_graph.log_full_edge_attr_t = log_edge_tm1
        print()

        # Convert 3-class absorbing states → binary edge labels (M → non-edge)
        e_state   = batched_graph.log_full_edge_attr_t.argmax(-1)             # (E,) ∈{0,1,2}
        edge_attr = (e_state == 1).long()                                     # (E,) binary

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

    def _eval_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        if self.loss_type == 'vb_kl':
            t, pt = self._sample_time(b, self.device, self.sample_time_method)
            
            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node)
            kl_prior = self._kl_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior
            return -loss
        
        elif self.loss_type == 'vb_kl_ce':
            # ELBO (separate edge / node KL) + λ_CE · direct reconstruction CE
            assert self.parametrization == 'x0', \
                "loss_type='vb_kl_ce' requires parametrization='x0'"

            t, pt = self._sample_time(b, self.device, self.sample_time_method)
            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            # ELBO ──────────────────────────────────────────────────────────
            kl_edge, kl_node, nll_edge, nll_node = self._compute_KL_edge_node(
                batched_graph, t_edge, t_node)
            mask  = (t == 0).float()                                   # (B,)
            elbo  = (mask  * (nll_edge + nll_node)
                     + (1. - mask) * (kl_edge + kl_node))             # (B,)
            kl_prior = self._kl_prior(batched_graph)

            # CE auxiliary ──────────────────────────────────────────────────
            ce_edge, ce_node = self._compute_CE(batched_graph, t_node, t_edge)

            # L = (ELBO + λ_CE · CE) / pt  +  L_prior
            vb_loss = elbo / pt + kl_prior + self.lambda_ce * (ce_edge + ce_node) / pt

            self._loss_components = {
                'kl_edge':  (kl_edge  * (1. - mask)).mean().item(),
                'kl_node':  (kl_node  * (1. - mask)).mean().item(),
                'nll_edge': (nll_edge * mask).mean().item(),
                'nll_node': (nll_node * mask).mean().item(),
                'kl_prior': kl_prior.mean().item(),
                'ce_edge':  ce_edge.mean().item(),
                'ce_node':  ce_node.mean().item(),
                'elbo':     elbo.mean().item(),
                'total':    vb_loss.mean().item(),
            }
            return -vb_loss

        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass #TODO not in the scope of the current submission

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior
            return -vb_loss

        else:
            raise ValueError()


    def _train_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph) 
        if self.loss_type == 'vb_kl':             
            # not sure it is ok to allow the parameterization to be xt, which is also sensible in math, for now it must be x0
            assert self.parametrization == 'x0'
            # sample t for each graph
            t, pt = self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self._kl_prior(batched_graph=batched_graph)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return -vb_loss
        
        
        elif self.loss_type == 'vb_kl_ce':
            # ELBO (separate edge / node KL) + λ_CE · direct reconstruction CE
            assert self.parametrization == 'x0', \
                "loss_type='vb_kl_ce' requires parametrization='x0'"

            t, pt = self._sample_time(b, self.device, self.sample_time_method)
            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            # ELBO ──────────────────────────────────────────────────────────
            kl_edge, kl_node, nll_edge, nll_node = self._compute_KL_edge_node(
                batched_graph, t_edge, t_node)
            mask  = (t == 0).float()                                   # (B,)
            elbo  = (mask  * (nll_edge + nll_node)
                     + (1. - mask) * (kl_edge + kl_node))             # (B,)

            # Update importance-sampling history using the ELBO signal
            Lt2      = elbo.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            self.Lt_history.scatter_(dim=0, index=t,
                                     src=(0.1 * Lt2 + 0.9 * Lt2_prev).detach())
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self._kl_prior(batched_graph)

            # CE auxiliary ──────────────────────────────────────────────────
            ce_edge, ce_node = self._compute_CE(batched_graph, t_node, t_edge)

            # L = (ELBO + λ_CE · CE) / pt  +  L_prior
            vb_loss = elbo / pt + kl_prior + self.lambda_ce * (ce_edge + ce_node) / pt

            # Store per-batch component means for external monitoring / VS Code watch
            self._loss_components = {
                'kl_edge':   (kl_edge  * (1. - mask)).mean().item(),
                'kl_node':   (kl_node  * (1. - mask)).mean().item(),
                'nll_edge':  (nll_edge * mask).mean().item(),
                'nll_node':  (nll_node * mask).mean().item(),
                'kl_prior':  kl_prior.mean().item(),
                'ce_edge':   ce_edge.mean().item(),
                'ce_node':   ce_node.mean().item(),
                'elbo':      elbo.mean().item(),
                'total':     vb_loss.mean().item(),
            }
            return -vb_loss

        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass # TODO

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)


            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior

            return -vb_loss 


        else:
            raise ValueError()
    