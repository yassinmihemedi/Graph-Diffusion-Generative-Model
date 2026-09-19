"""
Stochastic Block Model prior ablation for ERGMAnchoredDiffusion.

Ablation design:
  - β_ij = 1  (uniform masking — same as ER ablation)
  - ρ_ij = p_{z_i, z_j}  (block-to-block empirical probability from spectral clustering)

The SBM prior captures community structure: pairs within the same block get a
higher ρ than cross-block pairs.  The global p_kl matrix is fitted once from
all training graphs (spectral clustering → empirical block edge frequencies).

Density at t=T:  E[visible density] = α_min · (1/|E_full| Σ_{ij} p_{z_i,z_j})
                                     = α_min · ρ_G  (by construction of p_kl)
so the expected visible density matches ER and CL for the same α_min. ✓

Block alignment across graphs: blocks are sorted by ascending mean node degree,
so block 0 is always the lowest-degree block.  This makes the global p_kl
accumulation consistent.  If spectral clustering fails for a particular graph
(disconnected, too small), that graph falls back to Chung-Lu ρ_ij.

Usage:
    --arch GraphDenoisingERGM_SBM  --sbm_n_clusters 2
    (requires EmpiricalEmptyGraphGenerator to be built with sbm_n_clusters=2)
"""

import torch
import numpy as np
from .diffusion_base import scatter
from .diffusion_ergm import ERGMAnchoredDiffusion, _RHO_EPS, _EPS


class ERGMAnchoredDiffusion_SBM(ERGMAnchoredDiffusion):
    """SBM-prior ablation: uniform masking + block-structured Bernoulli at t=T."""

    def __init__(self, *args, sbm_n_clusters=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sbm_n_clusters = sbm_n_clusters

        # Load global p_kl from the sampler (populated by EmpiricalEmptyGraphGenerator
        # when sbm_n_clusters > 0 was passed during data loading).
        sampler = self.initial_graph_sampler
        if hasattr(sampler, 'sbm_p_kl') and sampler.sbm_p_kl is not None:
            self.register_buffer(
                '_sbm_p_kl',
                torch.tensor(sampler.sbm_p_kl, dtype=torch.float32)
            )
        else:
            # Fallback: uniform p_kl (reduces to ER prior)
            K = sbm_n_clusters
            self.register_buffer('_sbm_p_kl', torch.full((K, K), 0.5))

    # ── Override: β_ij = 1 everywhere ─────────────────────────────────────────

    def _beta_ij(self, batched_graph, edge_index):
        return torch.ones(edge_index.shape[1], device=self.device)

    # ── Override: ρ_ij = p_{z_i, z_j} from global SBM fit ────────────────────

    def _prior_rho_ij(self, batched_graph, full_ei):
        """SBM block-pair probability: ρ_ij = p_{z_i, z_j}."""
        if not hasattr(batched_graph, 'node_block') or batched_graph.node_block is None:
            # No block assignments available — fall back to Chung-Lu
            return super()._prior_rho_ij(batched_graph, full_ei)

        z   = batched_graph.node_block.long().to(self.device)  # (N_total,)
        z_i = z[full_ei[0]]
        z_j = z[full_ei[1]]

        p_kl   = self._sbm_p_kl.to(self.device)  # (K, K)
        # clamp indices to valid range in case a graph has fewer clusters
        K      = p_kl.shape[0]
        z_i    = z_i.clamp(0, K - 1)
        z_j    = z_j.clamp(0, K - 1)
        rho_ij = p_kl[z_i, z_j]
        return rho_ij.clamp(_RHO_EPS, 1.0 - _RHO_EPS)
