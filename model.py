import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.diffusion_base import cosine_beta_schedule, linear_beta_schedule, Tt1_beta_schedule, minimum_density_schedule
from diffusion.diffusion_binomial_vanilla import BinomialDiffusionVanilla
from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from diffusion.diffusion_partial_absorbing import PartialAbsorbingDiffusion, _ABS_CLASSES
from diffusion.diffusion_ergm import ERGMAnchoredDiffusion
from diffusion.diffusion_ergm_er import ERGMAnchoredDiffusion_ER
from diffusion.diffusion_ergm_sbm import ERGMAnchoredDiffusion_SBM
from layers.layers import (
    TGNN, GIN, GIN_Film, GCN, TGNN_degree_guided,
    GraphDiscriminator, GraphDiscriminatorFilm,
    GraphDenoising, GraphDenoisingERGM, GDiscPowerful,
    ScalableGNN, ScalableDisc, ScalableDiscDiffGAN,
)
from diffusion.diffusion_scalable import ScalableGraphDiffusion
from functools import partial
from diffusion.utils import str_to_bool

def add_model_args(parser):
    # Model params
    parser.add_argument('--loss_type', type=str, default='vb_kl')
    # ScalableGNN-specific
    parser.add_argument('--scalable_disc_dim', type=int, default=64,
                        help='Hidden dim for ScalableDisc (used with --arch ScalableGNN).')
    parser.add_argument('--scalable_disc_layers', type=int, default=3,
                        help='Number of MP layers in ScalableDisc.')
    # Discriminator type for ScalableGNN
    parser.add_argument('--disc_type', type=str, default='diffgan',
                        choices=['diffgan', 'vanilla'],
                        help='diffgan (default): time-conditioned Diffusion-GAN discriminator '
                             '(Wang et al. 2022). Both real and fake noised to same level t. '
                             'vanilla: original ScalableDisc without time conditioning.')
    # SparseDiff λ — for ScalableGNN and BinomialDiffusionVanilla (--arch GIN)
    parser.add_argument('--edge_fraction', type=float, default=1.0,
                        help='Fraction λ of non-edge candidate pairs scored per training '
                             'step (SparseDiff). All true edges are always included; '
                             'sampled non-edges are reweighted by 1/λ for an unbiased loss. '
                             '1.0 = full prediction scope (default); 0.2 = 5x fewer pairs scored.')
    # Fewer-step inference — for both ScalableGNN and discrete models
    parser.add_argument('--inference_steps', type=int, default=None,
                        help='Number of reverse-diffusion steps at inference. '
                             'None = use all T (--diffusion_steps) steps. '
                             'K < T = uniformly-spaced DDPM skip schedule, '
                             'enabling fast sampling to test H2.')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=64)
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--num_heads', type=int, nargs="*", default=[8, 8, 8, 8, 1])
    parser.add_argument('--final_prob_node', type=float, nargs="*", default=None)
    parser.add_argument('--final_prob_edge', type=float, nargs="*", default=[1-1e-12, 1e-12])
    parser.add_argument('--parametrization', type=str, default='x0')
    parser.add_argument('--sample_time_method', type=str, default='importance')
    parser.add_argument('--arch', type=str, default='GIN',
                        help='TGNN | GIN | GIN_Film | GCN | TGNN_degree_guided | GraphDenoising')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='cosine | linear | Tt1')
    parser.add_argument('--rho_min', type=float, default=None,
                        help='Minimum edge density (0,1) enforced at every noise step. '
                             'When set, wraps the base schedule with minimum_density_schedule.')
    parser.add_argument('--norm', type=str, default='None', help='None | BN' )
    parser.add_argument('--disc_lambda', type=float, default=0.0,
                        help='Discriminator guidance strength (0 = disabled).')
    parser.add_argument('--disc_dim', type=int, default=64,
                        help='Hidden dim for the graph discriminator.')
    parser.add_argument('--disc_layers', type=int, default=3,
                        help='Number of message-passing layers in the discriminator.')
    parser.add_argument('--disc_film', type=str_to_bool, default=False,
                        help='Use GraphDiscriminatorFilm (MiniAt_Film edge attention) '
                             'instead of plain GraphDiscriminator (default False).')
    parser.add_argument('--disc_gdisc', type=str_to_bool, default=False,
                        help='Use GDiscPowerful (GDisc architecture: GCN + edge-to-node '
                             'attention + MiniAt_Film). Overrides --disc_film when True.')
    parser.add_argument('--sample_rho', type=float, default=None,
                        help='Edge density used to initialise the reverse chain '
                             '(overrides final_prob_edge at sampling time only). '
                             'E.g. 0.05 starts from an Erdős–Rényi graph with 5%% density. '
                             'For GraphDenoising / GIN, set this to match your graph sparsity.')
    parser.add_argument('--lambda_ce', type=float, default=0.0,
                        help='Weight λ_CE for the direct-reconstruction CE auxiliary loss. '
                             'Active only with --loss_type vb_kl_ce. 0 = disabled.')
    parser.add_argument('--ce_pos_weight', type=str_to_bool, default=True,
                        help='Balance sparse-edge CE with per-graph pos_weight (default True).')
    parser.add_argument('--kl_pos_weight', type=str_to_bool, default=True,
                        help='Apply P1+P2 fix: positive-class reweighting w+=(1-rho)/rho in '
                             'the VB-KL loss. Removes degenerate equilibrium (P1) and restores '
                             'structural SNR from O(rho) to O(1) (P2). Default True.')
    # ── PA-DACA (Partial-Absorbing DACA) ──────────────────────────────────────
    parser.add_argument('--alpha_min', type=float, default=None,
                        help='Retention floor for PA-DACA floored cosine schedule. '
                             'Default: auto-set to train_density so the scaffold density '
                             'matches the real graph. Ignored for non-PA-DACA arches.')
    parser.add_argument('--daca_gamma', type=float, default=1.0,
                        help='Sharpness exponent γ for β_ij masking speed. '
                             'β_ij = min(β_max, (d_i*·d_j*/d̄²)^(γ/2)). Default 1.0.')
    parser.add_argument('--beta_max', type=float, default=3.0,
                        help='Cap on per-pair masking exponent β_ij. Default 3.0.')
    parser.add_argument('--lambda_LT', type=float, default=1.0,
                        help='Weight on prior KL term L_T in PA-DACA objective. Default 1.0.')
    # ── ERGM-Anchored DDPM (ERGMAnchoredDiffusion + GraphDenoisingERGM) ───────
    # --alpha_min, --daca_gamma, --beta_max, --lambda_LT are shared with PA-DACA.
    # No --edge_fraction: Q = E ∪ W(G) provides structural sparsification instead.
    parser.add_argument('--non_q_fraction', type=float, default=0.0,
                        help='SparseDiff: fraction of non-Q non-edges sampled per training step '
                             'with HT weight 1/λ to fix train/test coverage gap. 0 = disabled. '
                             'Start with 0.01 (|S|≈|Q| for Cora/CiteSeer). Default 0.0.')
    parser.add_argument('--ht_weight_max', type=float, default=None,
                        help='Cap on the Horvitz-Thompson weight 1/λ for wedge non-edges in Q. '
                             'Critical for very sparse graphs: PubMed has λ≈0.0008 → w=1250 uncapped. '
                             'Recommended: 50 for PubMed, None (no cap) for Cora/CiteSeer/Polblogs.')
    parser.add_argument('--lambda_deg', type=float, default=0.0,
                        help='Weight on auxiliary degree-matching loss L_deg = MSE(Σ_j P̂_ij, d*_i). '
                             'Directly penalizes wrong expected degree per node; fixes AvgDegree/PLE. '
                             'Recommended: 0.1 for sparse graphs. Default 0.0 (disabled).')
    parser.add_argument('--lambda_tri', type=float, default=0.0,
                        help='Weight on auxiliary triangle-matching loss L_tri = relative MSE of '
                             'expected triangle count vs true T*. Recommended: 0.05. Default 0.0.')
    parser.add_argument('--lambda_deg_node', type=float, default=0.0,
                        help='Weight on node-level degree regression head L_deg_node. '
                             'Forces GNN node embeddings h_i to encode d*_i via a direct gradient '
                             'path through all GNN layers. Both weighted by alpha_bar_t. '
                             'Recommended: 0.1. Default 0.0 (disabled).')
    parser.add_argument('--lambda_tri_node', type=float, default=0.0,
                        help='Weight on node-level triangle budget regression head L_tri_node. '
                             'Forces h_i to encode per-node triangle potential tau_node_i = Σ_j tau_ij(e_0). '
                             'Makes cat[h_i, h_j] in edge decoder triangle-aware at both endpoints. '
                             'Recommended: 0.05. Default 0.0 (disabled).')
    parser.add_argument('--sbm_n_clusters', type=int, default=None,
                        help='Number of SBM blocks for GraphDenoisingERGM_SBM prior ablation. '
                             'Set to 2 for community, 3-5 for ego. Triggers spectral clustering '
                             'on training graphs during data loading. Default: None (disabled).')

def get_model_id(args):
    return 'multinomial_diffusion'

def get_model(args, initial_graph_sampler):
    # For GIN/GCN/GraphDenoising with rho_min: align the stationary distribution with the
    # density floor so the reverse chain starts from that distribution.
    if getattr(args, 'rho_min', None) is not None and getattr(args, 'arch', '') in (
            'GIN', 'GIN_Film', 'GCN', 'GraphDenoising'):
        rho = float(args.rho_min)
        args.final_prob_edge = [1.0 - rho, rho]

    if args.final_prob_node is not None:
        assert abs(sum(args.final_prob_node) - 1) < 1e-6
        assert len(args.final_prob_node) == args.num_node_classes
    # Absorbing arches (PA-DACA, ERGM) use a 3-class state space internally but
    # args.num_edge_classes is still 2 (binary x_0 labels from the dataset).
    _absorbing_arches = ('GraphDenoisingPA', 'GraphDenoisingERGM')
    if getattr(args, 'arch', '') not in _absorbing_arches:
        assert abs(sum(args.final_prob_edge) - 1) < 1e-6
        assert len(args.final_prob_edge) == args.num_edge_classes

    if args.arch == 'TGNN':
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn = TGNN
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'GIN':
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn = GIN
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'GIN_Film':
        # GIN message passing + MiniAt_Film edge prediction + FiLM conditioning
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn = GIN_Film
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'GCN':
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn = GCN
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'TGNN_degree_guided':
        assert args.parametrization in ('xt_prescribed_st')
        dynamics_fn = TGNN_degree_guided
        diffusion_fn = BinomialDiffusionActive
    elif args.arch == 'GraphDenoising':
        # Scalable denoiser: sparse soft-weighted GIN MP + MiniAt_Film decoder.
        # Pair with --sample_rho to start sampling from an rho-density random graph.
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn  = GraphDenoising
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'GraphDenoisingPA':
        # PA-DACA: Partial-Absorbing Degree-Aware Causal Absorbing Diffusion.
        # Same GraphDenoising backbone, but:
        #   - 3-class absorbing edge state space {0, 1, M}
        #   - floored cosine schedule with α_min retention floor
        #   - per-pair β_ij masking speed from target_degree
        #   - w+^M(t) reweighting from masked pool (P1+P2 fixes)
        #   - L_T prior KL closed-form term (P3 fix)
        # Decoder outputs 2-class x_0 prediction (pred_edge_classes=2).
        assert args.parametrization == 'x0', 'PA-DACA requires parametrization=x0'
        dynamics_fn  = GraphDenoising
        diffusion_fn = PartialAbsorbingDiffusion
    elif args.arch in ('GraphDenoisingERGM', 'GraphDenoisingERGM_ER', 'GraphDenoisingERGM_SBM'):
        # ERGM-Anchored Discrete DDPM (and ER/SBM prior ablations):
        #   - Candidate set Q = E ∪ W(G) (ERGM-3 gradient support)
        #   - HT reweighting: w=1 for edges, 1/λ for wedge non-edges
        #   - Dynamic τ̂_ij(t) triangle feature for ERGM-3 conditioning
        #   - Same PA-DACA absorbing forward process + floored cosine schedule
        #   - GraphDenoisingERGM denoiser with τ̂_ij in edge embedding
        # Ablations (_ER, _SBM) use β_ij=1 and different ρ_ij at t=T.
        assert args.parametrization == 'x0', 'ERGM-DDPM requires parametrization=x0'
        dynamics_fn  = GraphDenoisingERGM
        if args.arch == 'GraphDenoisingERGM_ER':
            diffusion_fn = ERGMAnchoredDiffusion_ER
        elif args.arch == 'GraphDenoisingERGM_SBM':
            diffusion_fn = ERGMAnchoredDiffusion_SBM
        else:
            diffusion_fn = ERGMAnchoredDiffusion
    elif args.arch == 'ScalableGNN':
        # Scalable Graph Diffusion (Algorithms 1-6 pseudocode):
        #   - Bernoulli edge-deletion + VP Gaussian feature noise
        #   - MLP message-passing + multi-head edge attention (f_θ)
        #   - Optional ScalableDisc discriminator (--disc_lambda > 0)
        # Recommended flags:
        #   --parametrization x0 --noise_schedule cosine
        #   --sample_rho 0.5 --disc_lambda 0.1 (optional)
        dynamics_fn  = ScalableGNN
        diffusion_fn = ScalableGraphDiffusion
    else:
        raise NotImplementedError()
        
    # absorbing mode: GraphDenoisingPA / GraphDenoisingERGM (separate classes)
    # OR GraphDenoising + alpha_min (inline flag)
    _alpha_min_set = getattr(args, 'alpha_min', None) is not None
    _ergm_arches   = ('GraphDenoisingPA', 'GraphDenoisingERGM',
                      'GraphDenoisingERGM_ER', 'GraphDenoisingERGM_SBM')
    _is_absorbing  = (args.arch in _ergm_arches) or (
                      args.arch == 'GraphDenoising' and _alpha_min_set)
    dynamics = dynamics_fn(
            max_degree=args.max_degree,
            num_node_classes=2 if args.num_node_classes is None else args.num_node_classes,
            # Absorbing mode: edge state space is 3-class {0,1,M}, x_0 head is 2-class.
            num_edge_classes=(_ABS_CLASSES if _is_absorbing else args.num_edge_classes),
            pred_edge_classes=(2 if _is_absorbing else None),
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            num_heads=args.num_heads,
            dropout=args.dp_rate,
            norm=args.norm,
            gru=True,
            degree=args.degree,
            augmented_features=args.augmented_feature_dict,
            return_node_class=args.has_node_feature,
            rwse_k=getattr(args, 'rwse_k', 0),
            # Only consumed by GraphDenoisingERGM; GraphDenoising/GraphDenoisingPA
            # absorb it harmlessly via **kwargs. See _detect_legacy_ergm_embed in
            # eval_utils/evaluate.py for how this gets set at load time.
            legacy_ergm_embed=getattr(args, 'legacy_ergm_embed', False),
    )

    if args.noise_schedule == 'cosine':
        base_schedule = cosine_beta_schedule
    elif args.noise_schedule == 'linear':
        base_schedule = linear_beta_schedule
    elif args.noise_schedule == 'Tt1':
        base_schedule = Tt1_beta_schedule
    else:
        raise NotImplementedError()

    if args.rho_min is not None:
        assert 0.0 < args.rho_min < 1.0, '--rho_min must be in (0, 1)'
        noise_schedule = partial(minimum_density_schedule, base_schedule=base_schedule, rho_min=args.rho_min)
    else:
        noise_schedule = base_schedule

    # ── ScalableGNN: build and return independently ───────────────────────────
    if args.arch == 'ScalableGNN':
        dynamics = dynamics_fn(
            num_node_classes=2 if args.num_node_classes is None else args.num_node_classes,
            num_edge_classes=args.num_edge_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            num_heads=args.num_heads,
            dropout=args.dp_rate,
        )
        disc_lambda = getattr(args, 'disc_lambda', 0.0)
        discriminator = None
        if disc_lambda > 0:
            d_x        = 2 if args.num_node_classes is None else args.num_node_classes
            disc_dim   = getattr(args, 'scalable_disc_dim', 64)
            disc_layers= getattr(args, 'scalable_disc_layers', 3)
            disc_type  = getattr(args, 'disc_type', 'diffgan')
            if disc_type == 'diffgan':
                discriminator = ScalableDiscDiffGAN(
                    d_x=d_x, d_f=args.num_edge_classes,
                    dim=disc_dim, num_layers=disc_layers,
                    num_steps=args.diffusion_steps,
                )
            else:
                discriminator = ScalableDisc(
                    d_x=d_x, d_f=args.num_edge_classes,
                    dim=disc_dim, num_layers=disc_layers,
                )
        return ScalableGraphDiffusion(
            num_node_classes=2 if args.num_node_classes is None else args.num_node_classes,
            num_edge_classes=args.num_edge_classes,
            initial_graph_sampler=initial_graph_sampler,
            denoise_fn=dynamics,
            timesteps=args.diffusion_steps,
            discriminator=discriminator,
            disc_lambda=disc_lambda,
            sample_rho=getattr(args, 'sample_rho', 0.5) or 0.5,
            device=args.device,
            edge_fraction=getattr(args, 'edge_fraction', 1.0),
            inference_steps=getattr(args, 'inference_steps', None),
        )

    # ── All other arches: existing path below ────────────────────────────────
    discriminator = None
    disc_lambda = getattr(args, 'disc_lambda', 0.0)
    if disc_lambda > 0 and diffusion_fn is BinomialDiffusionVanilla:
        if getattr(args, 'disc_gdisc', False):
            # GDiscPowerful: GCN + edge-to-node attention + MiniAt_Film readout.
            # Recommended for GraphDenoising and GIN arches.
            disc_cls = GDiscPowerful
        elif getattr(args, 'disc_film', False):
            disc_cls = GraphDiscriminatorFilm
        else:
            disc_cls = GraphDiscriminator
        discriminator = disc_cls(
            num_node_classes=args.num_node_classes,
            num_edge_classes=args.num_edge_classes,
            dim=getattr(args, 'disc_dim', 64),
            num_layers=getattr(args, 'disc_layers', 3),
            att_heads=4,
        )

    # ── ERGM-Anchored DDPM (and ER/SBM ablations): build and return ──────────
    _ergm_diffusion_classes = (ERGMAnchoredDiffusion,
                               ERGMAnchoredDiffusion_ER,
                               ERGMAnchoredDiffusion_SBM)
    if diffusion_fn in _ergm_diffusion_classes:
        alpha_min = getattr(args, 'alpha_min', None)
        if alpha_min is None:
            # Target α_min = 100·ρ so the reverse chain starts from a dense enough
            # scaffold for MP and τ̂_ij to be informative.  For dense graphs where
            # 100·ρ ≥ 1 try 50·ρ; if still ≥ 1 cap at 0.5 (50% retention at t=T).
            rho = float(getattr(args, 'train_density', 0.022))
            _a = 100.0 * rho
            if _a >= 1.0:
                _a = 50.0 * rho
            if _a >= 1.0:
                _a = 0.5
            alpha_min = _a
        alpha_min = float(alpha_min)
        _ergm_kwargs = dict(
            num_node_classes=(2 if args.num_node_classes is None else args.num_node_classes),
            num_edge_classes=args.num_edge_classes,
            initial_graph_sampler=initial_graph_sampler,
            denoise_fn=dynamics,
            timesteps=args.diffusion_steps,
            loss_type=args.loss_type,
            parametrization=args.parametrization,
            final_prob_node=args.final_prob_node,
            sample_time_method=args.sample_time_method,
            noise_schedule=noise_schedule,
            device=args.device,
            alpha_min=alpha_min,
            daca_gamma=getattr(args, 'daca_gamma', 1.0),
            beta_max=getattr(args, 'beta_max', 3.0),
            lambda_LT=getattr(args, 'lambda_LT', 1.0),
            kl_pos_weight=getattr(args, 'kl_pos_weight', True),
            non_q_fraction=getattr(args, 'non_q_fraction', 0.0),
            lambda_deg=getattr(args, 'lambda_deg', 0.0),
            lambda_tri=getattr(args, 'lambda_tri', 0.0),
            ht_weight_max=getattr(args, 'ht_weight_max', None),
            lambda_deg_node=getattr(args, 'lambda_deg_node', 0.0),
            lambda_tri_node=getattr(args, 'lambda_tri_node', 0.0),
            max_degree=args.max_degree,
        )
        if diffusion_fn is ERGMAnchoredDiffusion_SBM:
            _ergm_kwargs['sbm_n_clusters'] = getattr(args, 'sbm_n_clusters', 2)
        return diffusion_fn(**_ergm_kwargs)

    # ── PA-DACA: build and return independently ───────────────────────────────
    if diffusion_fn is PartialAbsorbingDiffusion:
        # alpha_min defaults to train_density if available, else 0.022 (~Polblogs rho)
        alpha_min = getattr(args, 'alpha_min', None)
        if alpha_min is None:
            alpha_min = getattr(args, 'train_density', 0.022)
        alpha_min = float(alpha_min)
        return PartialAbsorbingDiffusion(
            num_node_classes=(2 if args.num_node_classes is None else args.num_node_classes),
            num_edge_classes=args.num_edge_classes,
            initial_graph_sampler=initial_graph_sampler,
            denoise_fn=dynamics,
            timesteps=args.diffusion_steps,
            loss_type=args.loss_type,
            parametrization=args.parametrization,
            final_prob_node=args.final_prob_node,
            sample_time_method=args.sample_time_method,
            noise_schedule=noise_schedule,
            device=args.device,
            alpha_min=alpha_min,
            daca_gamma=getattr(args, 'daca_gamma', 1.0),
            beta_max=getattr(args, 'beta_max', 3.0),
            lambda_LT=getattr(args, 'lambda_LT', 1.0),
            edge_fraction=getattr(args, 'edge_fraction', 1.0),
            kl_pos_weight=getattr(args, 'kl_pos_weight', True),
        )

    # Base kwargs accepted by all remaining diffusion classes
    base_kwargs = dict(
        timesteps=args.diffusion_steps,
        loss_type=args.loss_type,
        final_prob_node=args.final_prob_node,
        final_prob_edge=args.final_prob_edge,
        parametrization=args.parametrization,
        sample_time_method=args.sample_time_method,
        noise_schedule=noise_schedule,
        device=args.device,
    )

    # Extra kwargs only supported by BinomialDiffusionVanilla (not Active)
    if diffusion_fn is BinomialDiffusionVanilla:
        # Resolve alpha_min: explicit arg > train_density > None (standard marginal)
        _a_min = getattr(args, 'alpha_min', None)
        if _a_min is None and _is_absorbing:
            # GraphDenoising with alpha_min not explicitly set: use train_density
            _a_min = getattr(args, 'train_density', None)
        base_kwargs.update(
            discriminator=discriminator,
            disc_lambda=disc_lambda,
            lambda_ce=getattr(args, 'lambda_ce', 0.0),
            ce_pos_weight=getattr(args, 'ce_pos_weight', True),
            kl_pos_weight=getattr(args, 'kl_pos_weight', True),
            sample_rho=getattr(args, 'sample_rho', None),
            edge_fraction=getattr(args, 'edge_fraction', 1.0),
            # PA-DACA absorbing feature
            alpha_min=_a_min,
            daca_gamma=getattr(args, 'daca_gamma', 1.0),
            beta_max=getattr(args, 'beta_max', 3.0),
            lambda_LT=getattr(args, 'lambda_LT', 1.0),
        )

    base_dist = diffusion_fn(
        args.num_node_classes, args.num_edge_classes,
        initial_graph_sampler, dynamics,
        **base_kwargs)

    return base_dist
