"""
eval_utils/evaluate.py — Stand-alone evaluation for trained ERGM-Anchored DDPM.

Usage:
    python eval_utils/evaluate.py \
        --run_dir ./wandb/polblogs/multinomial_diffusion/multistep/polblogs_ergm_20250101_120000 \
        --checkpoint_epoch 4999 \
        --num_samples 16 \
        --batch_size 4 \
        --split test \
        --device cuda:0

The run_dir is the full path saved by DiffusionExperiment:
    {log_home}/{data_id}/{model_id}/{optim_id}/{run_name}
e.g. ./wandb/polblogs/multinomial_diffusion/multistep/polblogs_ergm_20250101_120000

Checkpoints are saved as checkpoint_{epoch}.pt (0-indexed epoch) when
(epoch+1) % check_every == 0.  For check_every=1000 and 5000 training
epochs, the last checkpoint is checkpoint_4999.pt → pass --checkpoint_epoch 4999.

Pass --list_checkpoints to print available checkpoint files and exit.
"""

import os
import sys
import pickle
import json
import argparse
import time

import torch
import numpy as np
import networkx as nx
import torch_geometric as pyg

# Works when invoked from the project root or from eval_utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data import get_data
from model import get_model


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate a trained ERGM-Anchored DDPM checkpoint.')
    p.add_argument('--run_dir', type=str, required=True,
        help='Full path to the run directory containing args.pickle and check/.')
    p.add_argument('--checkpoint_epoch', type=int, default=None,
        help='0-indexed training epoch to load: check/checkpoint_{epoch}.pt. '
             'Omit to load check/checkpoint.pt (the best-model save, if present).')
    p.add_argument('--list_checkpoints', action='store_true',
        help='Print available checkpoint files in run_dir/check/ and exit.')
    p.add_argument('--num_samples', type=int, default=16,
        help='Total number of graphs to generate (default 16).')
    p.add_argument('--batch_size', type=int, default=4,
        help='Graphs per model.sample() call. Reduce to 1 for pubmed/bitcoin (default 4).')
    p.add_argument('--split', type=str, default='test', choices=['eval', 'test'],
        help='"test" (default) or "eval" (validation set) evaluator.')
    p.add_argument('--device', type=str, default=None,
        help='Torch device string. Default: cuda:0 if available, else cpu.')
    p.add_argument('--seed', type=int, default=0,
        help='RNG seed for reproducible sampling (default 0).')
    p.add_argument('--inference_steps', type=int, default=None,
        help='Absolute number of reverse steps for DDIM-style skip schedule. '
             'None = use all T diffusion steps (default). '
             'Use --steps_fraction for a fraction of T (e.g. 0.5 for T/2).')
    p.add_argument('--steps_fraction', type=float, default=None,
        help='Inference steps as a fraction of the training T, e.g. 0.5 for T/2, '
             '0.333 for T/3. Takes priority over --inference_steps when both are set. '
             'Computed from args.diffusion_steps in the checkpoint pickle.')
    p.add_argument('--out', type=str, default=None,
        help='Write metrics as JSON to this path (optional).')
    p.add_argument('--save_graphs', type=str, default=None,
        help='Write generated nx graphs as pickle to this path (optional).')
    return p.parse_args()


# ─── Checkpoint helpers ────────────────────────────────────────────────────────

def _list_checkpoints(run_dir):
    check_dir = os.path.join(run_dir, 'check')
    if not os.path.isdir(check_dir):
        print(f'No check/ directory found at {run_dir}')
        return
    files = sorted(
        f for f in os.listdir(check_dir)
        if f.startswith('checkpoint') and f.endswith('.pt'))
    if not files:
        print(f'No checkpoints found in {check_dir}')
    else:
        print(f'Checkpoints in {check_dir}:')
        for f in files:
            path = os.path.join(check_dir, f)
            sz = os.path.getsize(path) / 1e6
            # Peek at epoch metadata without loading the full weights
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            epoch = ckpt.get('current_epoch', '?')
            print(f'  {f}  ({sz:.0f} MB) ')


def _load_checkpoint(run_dir, checkpoint_epoch):
    check_dir = os.path.join(run_dir, 'check')
    if checkpoint_epoch is not None:
        ckpt_name = f'checkpoint_{checkpoint_epoch}.pt'
    else:
        ckpt_name = 'checkpoint.pt'
    ckpt_path = os.path.join(check_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        available = sorted(
            f for f in os.listdir(check_dir)
            if f.startswith('checkpoint') and f.endswith('.pt'))
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            f'Available files: {available}')
    return ckpt_path, torch.load(ckpt_path, map_location='cpu', weights_only=False)


# Keys that GraphDenoisingERGM gained in a later revision.
# Old checkpoints lack these; they are safe to skip at eval time because
# both heads are gated by `tau_node_true is not None`, which is always
# False during sampling (only set inside _loss_impl during training).
_OPTIONAL_HEAD_KEYS = frozenset([
    '_denoise_fn.deg_head.weight',
    '_denoise_fn.deg_head.bias',
    '_denoise_fn.tri_head.weight',
    '_denoise_fn.tri_head.bias',
])


def _detect_legacy_ergm_embed(state_dict):
    """True if `state_dict` predates the node_embed_t / 3-layer node_embed
    revision of GraphDenoisingERGM (see layers/layers.py).

    Unlike deg_head/tri_head (dead at eval time, safe to skip via
    _OPTIONAL_HEAD_KEYS), node_embed / node_embed_t / tau_proj feed every
    forward pass. Loading a legacy checkpoint into the new-shaped modules
    would silently corrupt sampling instead of raising, so the model must be
    *constructed* in the matching shape before load_state_dict runs — see
    GraphDenoisingERGM(legacy_ergm_embed=...) and get_model() in model.py.

    Detection: the old node_embed was a bare nn.Linear(1, dim), so its
    weight key is '...node_embed.weight' with no '.0.' submodule index; the
    new node_embed is an nn.Sequential, keyed '...node_embed.0.weight'.
    """
    return ('_denoise_fn.node_embed.weight' in state_dict
            and '_denoise_fn.node_embed.0.weight' not in state_dict)


def _load_state_dict_compat(model, state_dict, ckpt_path):
    """Load state dict with backward compatibility for pre-head checkpoints.

    Strategy: always call load_state_dict(strict=False) to get the full
    IncompatibleKeys report, then decide:
      - Only _OPTIONAL_HEAD_KEYS missing, nothing unexpected → OK, warn and continue.
      - Anything else missing or unexpected → real architecture mismatch, raise.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing    = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)

    if not missing and not unexpected:
        return  # perfect match

    non_optional_missing = missing - _OPTIONAL_HEAD_KEYS

    if non_optional_missing or unexpected:
        raise RuntimeError(
            f'State dict mismatch loading {ckpt_path}.\n'
            f'  Missing (non-optional): {sorted(non_optional_missing)}\n'
            f'  Unexpected keys in checkpoint: {sorted(unexpected)}\n'
            f'This indicates a model architecture mismatch, not a version '
            f'compatibility issue.  Verify --run_dir points to the correct run.')

    # Only optional head keys are missing — old checkpoint, safe to proceed.
    print(
        f'\n[compat] Old checkpoint — lacks node-level head keys '
        f'(deg_head, tri_head).\n'
        f'  Missing: {sorted(missing)}\n'
        f'  These heads are only called during training (gated by '
        f'tau_node_true=None at sample time) and will not be invoked.\n'
        f'  Proceeding with randomly-initialised head weights — safe for eval.\n')


# ─── Inference step resolution ────────────────────────────────────────────────

def _resolve_inference_steps(eval_args, diffusion_steps):
    """Return the concrete inference_steps integer (or None = full T).

    Priority:
      1. --steps_fraction  → int(diffusion_steps * fraction), clamped to [1, T]
      2. --inference_steps → used as-is
      3. Neither set       → None (full T steps)
    """
    if eval_args.steps_fraction is not None:
        f = eval_args.steps_fraction
        if not (0 < f <= 1.0):
            raise ValueError(
                f'--steps_fraction must be in (0, 1]. Got {f}.')
        steps = max(1, int(diffusion_steps * f))
        label = f'T×{f} ({diffusion_steps}×{f}={steps})'
        print(f'Inference steps: {steps}  [{label}]')
        return steps
    if eval_args.inference_steps is not None:
        steps = eval_args.inference_steps
        print(f'Inference steps: {steps}  [{steps}/{diffusion_steps} of T]')
        return steps
    print(f'Inference steps: {diffusion_steps}  [full T — no skip schedule]')
    return None


# ─── Edge Overlap metric ───────────────────────────────────────────────────────

def _compute_edge_overlap(generated_nx_graphs, evaluator):
    """Compute mean Edge Overlap (Jaccard of edge sets) vs. reference graph(s).

    EO(G_gen, G_ref) = |E_gen ∩ E_ref| / |E_gen ∪ E_ref|

    Only well-defined for NetworkEvaluator (single reference graph with fixed node
    IDs shared by every generated graph, as in polblogs / single-graph datasets).
    For GenericGraphEvaluator (multi-graph, no node correspondence across graphs),
    EO is returned as NaN.
    """
    if not hasattr(evaluator, 'reference'):
        return {'edge_overlap': float('nan')}

    edges_ref = set(frozenset(e) for e in evaluator.reference.edges())
    eos = []
    for g_gen in generated_nx_graphs:
        edges_gen = set(frozenset(e) for e in g_gen.edges())
        union = len(edges_gen | edges_ref)
        if union > 0:
            eos.append(len(edges_gen & edges_ref) / union)

    if not eos:
        return {'edge_overlap': float('nan')}
    return {'edge_overlap': float(np.mean(eos))}


# ─── Graph post-processing ────────────────────────────────────────────────────

def _to_lcc(pyg_data):
    """Convert a PyG graph to its largest connected component as a nx.Graph.

    Returns (topo_graph, typed_graph) where:
      topo_graph  — topology-only nx.Graph (for structural MMD)
      typed_graph — same LCC with integer 'node_attr' on each node
                    (for molecular metrics, ZINC only); equals topo_graph
                    when no node_attr is present.

    Returns (None, None) if the graph has no edges.
    """
    _node_attrs = ['node_attr'] if hasattr(pyg_data, 'node_attr') else []
    g_typed = pyg.utils.to_networkx(
        pyg_data, node_attrs=_node_attrs, to_undirected=True)
    if g_typed.number_of_edges() == 0:
        return None, None
    comps = list(nx.connected_components(g_typed))
    lcc_nodes = max(comps, key=len)
    lcc_typed = g_typed.subgraph(lcc_nodes).copy()

    # Topology-only copy (no node attributes) for structural MMD
    lcc_topo = nx.Graph()
    lcc_topo.add_nodes_from(range(lcc_typed.number_of_nodes()))
    # Re-index so nodes are 0..N-1 (subgraph keeps original integer IDs)
    node_map = {v: i for i, v in enumerate(sorted(lcc_typed.nodes()))}
    for u, v in lcc_typed.edges():
        lcc_topo.add_edge(node_map[u], node_map[v])

    # Re-index typed graph too for consistency
    lcc_typed_reindexed = nx.relabel_nodes(lcc_typed, node_map)
    # Coerce node_attr to plain int (may be a 0-dim tensor from to_networkx)
    for nd in lcc_typed_reindexed.nodes():
        raw = lcc_typed_reindexed.nodes[nd].get('node_attr', 0)
        lcc_typed_reindexed.nodes[nd]['node_attr'] = int(raw)

    return lcc_topo, lcc_typed_reindexed


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    eval_args = _parse_args()

    torch.manual_seed(eval_args.seed)
    np.random.seed(eval_args.seed)

    # ── List checkpoints and exit ──────────────────────────────────────────────
    if eval_args.list_checkpoints:
        _list_checkpoints(eval_args.run_dir)
        return

    # ── Device ────────────────────────────────────────────────────────────────
    if eval_args.device is not None:
        device = eval_args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── Load training args from pickle ────────────────────────────────────────
    args_path = os.path.join(eval_args.run_dir, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f'args.pickle not found at {args_path}. '
            f'Is run_dir pointing to the correct run directory?')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    # Override training-time settings that are irrelevant or harmful at eval time.
    args.device = device
    args.num_workers = 0       # avoid multiprocessing issues on login/compute nodes
    args.pin_memory = False    # pin_memory requires CUDA DataLoader workers
    args.batch_size = 1        # eval data loader batch size; does not affect sampling

    # ── Rebuild data components ────────────────────────────────────────────────
    # get_data() reconstructs the initial_graph_sampler (needed by model.sample())
    # and the evaluators (needed for metric computation).
    # It returns 13 values; unpack all of them.
    print(f'Loading dataset: {args.dataset} …')
    (train_loader, eval_loader, test_loader,
     num_node_feat, num_node_classes, num_edge_classes,
     max_degree, augmented_feature_dict,
     initial_graph_sampler,
     eval_evaluator, test_evaluator,
     monitoring_statistics, train_density) = get_data(args)

    # The pickled args already has all model-construction fields correctly set:
    # train.py calls get_data(), modifies args (num_node_classes, final_prob_node,
    # train_density, etc.), and THEN calls save_args().  So the pickle reflects the
    # fully-prepared state that get_model() expects.
    #
    # Critical: do NOT override args.num_node_classes or args.num_edge_classes from
    # get_data()'s return values.  For polblogs (no node features), get_data()
    # returns num_node_classes=None, but train.py then sets it to 2 before saving.
    # Overwriting with None breaks the assertion in get_model():
    #   assert len(args.final_prob_node) == args.num_node_classes  → 2 == None
    #
    # Only patch fields that may be absent in very old pickles.
    if not hasattr(args, 'train_density') or args.train_density is None:
        args.train_density = train_density
    if not hasattr(args, 'has_node_feature'):
        args.has_node_feature = False
    if not hasattr(args, 'final_prob_node') or args.final_prob_node is None:
        args.final_prob_node = [1 - 1e-12, 1e-12]
        args.num_node_classes = 2
        args.has_node_feature = False

    # ── Build and load model ───────────────────────────────────────────────────
    # Checkpoint must be loaded BEFORE get_model() so args.legacy_ergm_embed can
    # be set from its actual key shapes — GraphDenoisingERGM's node_embed /
    # node_embed_t / tau_proj submodules are architecture, not dead weights, so
    # they must be *constructed* in the matching shape (see
    # _detect_legacy_ergm_embed above), not patched after the fact.
    ckpt_path, ckpt = _load_checkpoint(eval_args.run_dir, eval_args.checkpoint_epoch)
    args.legacy_ergm_embed = _detect_legacy_ergm_embed(ckpt['model'])
    if args.legacy_ergm_embed:
        print('[compat] Detected legacy GraphDenoisingERGM checkpoint '
              '(single-Linear node_embed, no node_embed_t, 2-layer tau_proj). '
              'Building model in the matching legacy shape.')

    model = get_model(args, initial_graph_sampler=initial_graph_sampler)
    _load_state_dict_compat(model, ckpt['model'], ckpt_path)
    saved_epoch = ckpt.get('current_epoch', 'unknown')
    print(f'Loaded checkpoint: {ckpt_path}')
    #print(f'  Saved at epoch (0-indexed): {saved_epoch}')

    model = model.to(device)
    model.eval()

    # ── Resolve inference steps (after pickle is loaded for diffusion_steps) ───
    inference_steps = _resolve_inference_steps(
        eval_args, getattr(args, 'diffusion_steps', None))

    # ── Generate graphs ────────────────────────────────────────────────────────
    total       = eval_args.num_samples
    batch_size  = eval_args.batch_size
    n_batches   = (total + batch_size - 1) // batch_size
    generated   = []
    skipped     = 0
    t_start     = time.time()

    print(f'\nGenerating {total} graphs  '
          f'({n_batches} batch(es) of ≤{batch_size}) …')

    typed_generated = []  # typed graphs (node_attr preserved) for molecular metrics

    with torch.no_grad():
        for b_idx in range(n_batches):
            n_this = min(batch_size, total - b_idx * batch_size)
            t_b = time.time()
            print(f'  Batch {b_idx+1}/{n_batches}: sampling {n_this} graph(s) …',
                  end=' ', flush=True)

            batch_out = model.sample(n_this, inference_steps=inference_steps)
            pyg_list  = batch_out.to_data_list()

            for pyg_data in pyg_list:
                g_topo, g_typed = _to_lcc(pyg_data)
                if g_topo is None:
                    skipped += 1
                    continue
                generated.append(g_topo)
                typed_generated.append(g_typed)

            print(f'{time.time()-t_b:.1f}s  '
                  f'(total valid so far: {len(generated)})')

    elapsed = time.time() - t_start
    print(f'\nGenerated {len(generated)} valid graphs, '
          f'{skipped} skipped (empty/edgeless)  [{elapsed:.1f}s total]')

    if not generated:
        raise RuntimeError(
            'All generated graphs were empty — verify the checkpoint and dataset.')

    if eval_args.save_graphs:
        with open(eval_args.save_graphs, 'wb') as f:
            pickle.dump(generated, f)
        print(f'Saved generated graphs → {eval_args.save_graphs}')

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluator = test_evaluator if eval_args.split == 'test' else eval_evaluator
    print(f'\nEvaluating against {eval_args.split} split …')
    t_eval = time.time()
    metrics = evaluator.evaluate(generated)
    metrics.update(_compute_edge_overlap(generated, evaluator))

    # Molecular metrics (ZINC only): validity / uniqueness / novelty.
    # Requires mol_train_hashes and mol_max_valence to be set on the evaluator
    # by datasets/data.py (ZINC section).  Silently skipped for other datasets.
    _mol_hashes  = getattr(evaluator, 'mol_train_hashes', None)
    _mol_valence = getattr(evaluator, 'mol_max_valence',  None)
    if _mol_hashes is not None and typed_generated:
        from eval_utils.molecular_metrics import zinc_metrics
        mol_met = zinc_metrics(typed_generated, _mol_hashes, _mol_valence)
        metrics.update(mol_met)
        print(f'[mol] validity={mol_met["mol/validity"]:.3f}  '
              f'uniqueness={mol_met["mol/uniqueness"]:.3f}  '
              f'novelty={mol_met["mol/novelty"]:.3f}')

    print(f'Evaluation: {time.time()-t_eval:.1f}s')

    # ── Report ─────────────────────────────────────────────────────────────────
    print('\n── Metrics ───────────────────────────────────────────────')
    for k, v in sorted(metrics.items()):
        print(f'  {k:<35s} {float(v):.6f}')

    # Monitoring subset (same statistics the training loop uses for model selection)
    print('\n── Monitoring statistics (used for model selection) ──────')
    for k in monitoring_statistics:
        v = metrics.get(k, float('nan'))
        print(f'  {k:<35s} {float(v):.6f}')
    mean_monitor = np.nanmean([float(metrics.get(k, float('nan')))
                               for k in monitoring_statistics])
    print(f'  {"mean":<35s} {mean_monitor:.6f}')

    if eval_args.out:
        out_dict = {k: float(v) for k, v in metrics.items()}
        out_dict['_meta'] = {
            'run_dir':            eval_args.run_dir,
            'checkpoint_epoch':   eval_args.checkpoint_epoch,
            'saved_epoch':        saved_epoch,
            'num_samples':        total,
            'valid_samples':      len(generated),
            'skipped_empty':      skipped,
            'split':              eval_args.split,
            'device':             device,
            'seed':               eval_args.seed,
            'elapsed_s':          round(elapsed, 1),
            'inference_steps':    inference_steps,
            'steps_fraction':     eval_args.steps_fraction,
            'diffusion_steps_T':  getattr(args, 'diffusion_steps', None),
        }
        with open(eval_args.out, 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(f'\nSaved metrics → {eval_args.out}')


if __name__ == '__main__':
    main()
