"""
Erdős-Rényi prior ablation for ERGMAnchoredDiffusion.

Ablation design:
  - β_ij = 1  (uniform masking — no degree-aware modulation)
  - ρ_ij = ρ_G = 2m / (N(N-1))  (global edge density, same for all pairs)

This isolates the contribution of the Chung-Lu degree-aware prior by keeping
all other components (ERGM ELBO, HT reweighting, τ̂_ij, noise schedule) identical.

Density at t=T:  E[visible density] = α_min^1 · ρ_G  (same α_min as CL)
which equals the CL density if β̄_CL ≈ 1.  Both priors preserve the same
expected scaffold density for a fair comparison.

Usage:
    --arch GraphDenoisingERGM_ER
"""

import torch
from .diffusion_base import scatter
from .diffusion_ergm import ERGMAnchoredDiffusion, _RHO_EPS, _EPS


class ERGMAnchoredDiffusion_ER(ERGMAnchoredDiffusion):
    """ER-prior ablation: uniform masking + global-density Bernoulli at t=T."""

    # ── Override: β_ij = 1 everywhere (no degree modulation) ──────────────────

    def _beta_ij(self, batched_graph, edge_index):
        return torch.ones(edge_index.shape[1], device=self.device)

    # ── Override: ρ_ij = ρ_G = 2m / (N(N-1))  ────────────────────────────────

    def _prior_rho_ij(self, batched_graph, full_ei):
        """Global ER density: same ρ_G for every pair in a graph."""
        b        = batched_graph.num_graphs
        batch_fe = batched_graph.batch.to(self.device)[full_ei[0]]

        if (hasattr(batched_graph, 'target_degree')
                and batched_graph.target_degree is not None):
            # ρ_G = Σ d*_i / (N*(N-1))  ← handshaking: Σ d* = 2m
            d_star   = batched_graph.target_degree.float().to(self.device)
            n_nodes  = batched_graph.nodes_per_graph.float().to(self.device)
            d_sum    = scatter(d_star, batched_graph.batch.to(self.device),
                               dim=0, dim_size=b, reduce='sum')   # (B,) = 2m_g
            n_pairs  = (n_nodes * (n_nodes - 1.0)).clamp(min=1.0)  # (B,)
            rho_g    = (d_sum / n_pairs).clamp(_RHO_EPS, 1.0 - _RHO_EPS)
        else:
            # Fallback: count edges from full_edge_attr
            e0      = batched_graph.full_edge_attr.float()
            n_pos   = scatter(e0, batch_fe, dim=0, dim_size=b, reduce='sum')
            n_cands = scatter(torch.ones_like(e0), batch_fe,
                              dim=0, dim_size=b, reduce='sum').clamp(min=1.0)
            rho_g   = (n_pos / n_cands).clamp(_RHO_EPS, 1.0 - _RHO_EPS)

        return rho_g[batch_fe]
