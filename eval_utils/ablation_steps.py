#!/usr/bin/env python3
"""
eval_utils/ablation_steps.py  —  Inference-step ablation for ERGM-Anchored DDPM.

Generates graphs at multiple reverse-step fractions of the training T and
compares six structural metrics against the reference graph from the dataset.

Metrics:
  power_law_exp        power-law exponent α (fitted by powerlaw package)
  cpl                  characteristic path length on LCC
  assortativity        degree assortativity coefficient
  triangle_count       number of triangles
  edge_overlap         Jaccard |E_gen ∩ E_ref| / |E_gen ∪ E_ref| (single-graph datasets only)
  clustering_coeff     global clustering coefficient = 3Δ / wedges

Usage:
    python eval_utils/ablation_steps.py \\
        --run_dir ./wandb/polblogs/.../polblogs_ergm_20260802_011344 \\
        --checkpoint_epoch 3999 \\
        --num_samples 8 \\
        --batch_size 4 \\
        --fractions 0.1 0.3 0.5 0.7 0.9 1.0 \\
        --seed 0 \\
        --out ablation_steps.json
"""

import os
import sys
import pickle
import json
import argparse
import time

import numpy as np
import torch
import networkx as nx
import torch_geometric as pyg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data import get_data
from model import get_model
from eval_utils.graph_statistics import compute_graph_statistics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Inference-step ablation for a trained ERGM checkpoint.')
    p.add_argument('--run_dir', required=True,
        help='Path to run directory containing args.pickle and check/.')
    p.add_argument('--checkpoint_epoch', type=int, default=None,
        help='0-indexed epoch → check/checkpoint_{epoch}.pt. '
             'Omit to load check/checkpoint.pt (best-model save).')
    p.add_argument('--fractions', type=float, nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        help='Fractions of training T to use as inference steps. '
             'Default: 0.1 0.3 0.5 0.7 0.9 1.0')
    p.add_argument('--num_samples', type=int, default=8,
        help='Generated graphs per fraction (default 8).')
    p.add_argument('--batch_size', type=int, default=4,
        help='Graphs per model.sample() call (default 4).')
    p.add_argument('--split', choices=['eval', 'test'], default='test',
        help='Which evaluator to use for reference graphs (default test).')
    p.add_argument('--device', type=str, default=None,
        help='Torch device. Default: cuda:0 if available, else cpu.')
    p.add_argument('--seed', type=int, default=0,
        help='Base RNG seed. Each fraction offsets by its index (reproducible).')
    p.add_argument('--out', type=str, default=None,
        help='Save full results as JSON to this path (optional).')
    return p.parse_args()


# ─── Checkpoint / compat ───────────────────────────────────────────────────────

_OPTIONAL_HEAD_KEYS = frozenset([
    '_denoise_fn.deg_head.weight',
    '_denoise_fn.deg_head.bias',
    '_denoise_fn.tri_head.weight',
    '_denoise_fn.tri_head.bias',
])


def _load_ckpt(run_dir, checkpoint_epoch):
    check_dir = os.path.join(run_dir, 'check')
    name = (f'checkpoint_{checkpoint_epoch}.pt'
            if checkpoint_epoch is not None else 'checkpoint.pt')
    path = os.path.join(check_dir, name)
    if not os.path.exists(path):
        avail = sorted(f for f in os.listdir(check_dir)
                       if f.startswith('checkpoint') and f.endswith('.pt'))
        raise FileNotFoundError(f'{path} not found. Available: {avail}')
    return path, torch.load(path, map_location='cpu', weights_only=False)


def _load_state_dict_compat(model, state_dict, ckpt_path):
    result = model.load_state_dict(state_dict, strict=False)
    missing    = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if not missing and not unexpected:
        return
    non_opt = missing - _OPTIONAL_HEAD_KEYS
    if non_opt or unexpected:
        raise RuntimeError(
            f'State dict mismatch: {ckpt_path}\n'
            f'  Missing (non-optional): {sorted(non_opt)}\n'
            f'  Unexpected: {sorted(unexpected)}')
    print(f'[compat] Old checkpoint — deg_head/tri_head absent (safe at eval time).')


# ─── Graph helpers ─────────────────────────────────────────────────────────────

def _to_lcc(pyg_data):
    """PyG batch element → LCC as nx.Graph; None if edgeless."""
    g = pyg.utils.to_networkx(pyg_data, to_undirected=True)
    if g.number_of_edges() == 0:
        return None
    lcc_nodes = max(nx.connected_components(g), key=len)
    return g.subgraph(lcc_nodes).copy()


def _ref_graph(evaluator):
    """Return reference nx.Graph (NetworkEvaluator) or None (GenericGraphEvaluator)."""
    return getattr(evaluator, 'reference', None)


def _ref_statistics(evaluator):
    """
    Return reference structural statistics dict.

    NetworkEvaluator:      use pre-computed reference_stats (same formula as generated).
    GenericGraphEvaluator: convert each DGL reference graph, run compute_graph_statistics,
                           then average the resulting scalars.
    """
    if hasattr(evaluator, 'reference_stats'):
        return dict(evaluator.reference_stats)

    import dgl
    stats_list = []
    for dgl_g in evaluator.references:
        nx_g = dgl.to_networkx(dgl_g).to_undirected()
        A = nx.to_scipy_sparse_array(nx_g, format='csr', dtype=float)
        stats_list.append(compute_graph_statistics(A))

    keys = stats_list[0].keys()
    return {k: float(np.nanmean([s[k] for s in stats_list])) for k in keys}


# ─── Metric computation ────────────────────────────────────────────────────────

# Subset of compute_graph_statistics keys we display, plus edge_overlap
_STAT_KEYS = [
    'power_law_exp',
    'cpl',
    'assortativity',
    'triangle_count',
    'clustering_coefficient',
]
_STAT_LABELS = {
    'power_law_exp':        'Power-law α',
    'cpl':                  'CPL',
    'assortativity':        'Assortativity',
    'triangle_count':       'Triangles',
    'clustering_coefficient': 'CC (global)',
    'edge_overlap':         'EO (Jaccard)',
}
_ALL_KEYS = _STAT_KEYS + ['edge_overlap']


def _graph_stats(nx_graph, ref_nx_graph):
    """
    Compute _STAT_KEYS metrics + Jaccard edge_overlap for one nx.Graph.

    ref_nx_graph: the single reference nx.Graph (NetworkEvaluator) or None.
    """
    A = nx.to_scipy_sparse_array(nx_graph, format='csr', dtype=float)
    stats = compute_graph_statistics(A)
    out = {k: float(stats.get(k, np.nan)) for k in _STAT_KEYS}

    if ref_nx_graph is not None:
        edges_gen = set(frozenset(e) for e in nx_graph.edges())
        edges_ref = set(frozenset(e) for e in ref_nx_graph.edges())
        union = len(edges_gen | edges_ref)
        out['edge_overlap'] = (len(edges_gen & edges_ref) / union
                               if union > 0 else np.nan)
    else:
        out['edge_overlap'] = np.nan

    return out


def _generate_and_measure(model, num_samples, batch_size,
                          inference_steps, ref_nx_graph, device):
    """
    Sample `num_samples` graphs with the given inference_steps schedule,
    convert to nx (LCC), compute stats for each, and return (mean, std) dicts.
    """
    total     = num_samples
    n_batches = (total + batch_size - 1) // batch_size
    all_stats = []
    skipped   = 0

    with torch.no_grad():
        for b in range(n_batches):
            n_this   = min(batch_size, total - b * batch_size)
            batch_out = model.sample(n_this, inference_steps=inference_steps)
            for pyg_data in batch_out.to_data_list():
                g = _to_lcc(pyg_data)
                if g is None:
                    skipped += 1
                    continue
                all_stats.append(_graph_stats(g, ref_nx_graph))

    if not all_stats:
        raise RuntimeError('All generated graphs were empty — bad checkpoint?')

    if skipped:
        print(f'    ({skipped} edgeless graphs skipped)')

    mean = {k: float(np.nanmean([s[k] for s in all_stats])) for k in _ALL_KEYS}
    std  = {k: float(np.nanstd( [s[k] for s in all_stats])) for k in _ALL_KEYS}
    return mean, std, len(all_stats)


# ─── Table printing ────────────────────────────────────────────────────────────

def _fmt_val(v, key):
    """Format a scalar metric value for table display."""
    if not np.isfinite(v):
        return 'NaN'
    if key == 'triangle_count':
        return f'{v:,.0f}'
    return f'{v:.4f}'


def _fmt_nmae(gen, ref, key):
    """NMAE as percentage string, or '—' if undefined."""
    if not np.isfinite(ref) or ref == 0 or not np.isfinite(gen):
        return '—'
    return f'{abs(gen - ref) / abs(ref) * 100:.1f}%'


def _print_table(frac_labels, ref_stats, results, T):
    """
    Print the ablation comparison table.

    results: list of (mean_dict, std_dict, n_valid) — one per fraction.
    """
    COL  = 12   # width of each data column
    LCOL = 22   # width of metric label column
    RCOL = 11   # width of reference column

    divider = '─' * (LCOL + RCOL + 2 + len(frac_labels) * (COL + 2))

    # ── Header ─────────────────────────────────────────────────────────────────
    header = f"{'Metric':<{LCOL}}  {'Reference':>{RCOL}}"
    for lbl in frac_labels:
        header += f"  {lbl:>{COL}}"
    print('\n' + divider)
    print(header)
    print(divider)

    # ── One section per metric ─────────────────────────────────────────────────
    for key in _ALL_KEYS:
        label = _STAT_LABELS[key]
        ref_v = ref_stats.get(key, np.nan)

        # Special case: reference EO is the theoretical maximum (self = 1.0)
        if key == 'edge_overlap':
            ref_display = '1.0000' if np.isfinite(results[0][0].get('edge_overlap', np.nan)) else 'N/A'
        else:
            ref_display = _fmt_val(ref_v, key)

        # Values row
        row_vals = f"  {label:<{LCOL}}  {ref_display:>{RCOL}}"
        for mean, std, _ in results:
            v = mean.get(key, np.nan)
            row_vals += f"  {_fmt_val(v, key):>{COL}}"
        print(row_vals)

        # Std row
        row_std = f"  {'  ± std':<{LCOL}}  {'':>{RCOL}}"
        for mean, std, _ in results:
            s = std.get(key, np.nan)
            if np.isfinite(s):
                if key == 'triangle_count':
                    row_std += f"  {s:>{COL},.0f}"
                else:
                    row_std += f"  {s:>{COL}.4f}"
            else:
                row_std += f"  {'—':>{COL}}"
        print(row_std)

        # NMAE row
        row_nmae = f"  {'  NMAE %':<{LCOL}}  {'—':>{RCOL}}"
        for mean, std, _ in results:
            v = mean.get(key, np.nan)
            # For EO: NMAE vs 1.0 (perfect overlap)
            ref_for_nmae = 1.0 if key == 'edge_overlap' else ref_v
            row_nmae += f"  {_fmt_nmae(v, ref_for_nmae, key):>{COL}}"
        print(row_nmae)

        print()  # blank line between metrics

    print(divider)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # ── Device ─────────────────────────────────────────────────────────────────
    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load pickled training args ──────────────────────────────────────────────
    args_path = os.path.join(args.run_dir, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f'args.pickle not found at {args_path}')
    with open(args_path, 'rb') as f:
        train_args = pickle.load(f)

    train_args.device     = device
    train_args.num_workers = 0
    train_args.pin_memory  = False
    train_args.batch_size  = 1

    # ── Data & evaluators ───────────────────────────────────────────────────────
    print(f'Loading dataset: {train_args.dataset} …')
    (train_loader, eval_loader, test_loader,
     num_node_feat, num_node_classes, num_edge_classes,
     max_degree, augmented_feature_dict,
     initial_graph_sampler,
     eval_evaluator, test_evaluator,
     monitoring_statistics, train_density) = get_data(train_args)

    # Patch old pickles (same logic as evaluate.py)
    if not hasattr(train_args, 'train_density') or train_args.train_density is None:
        train_args.train_density = train_density
    if not hasattr(train_args, 'has_node_feature'):
        train_args.has_node_feature = False
    if not hasattr(train_args, 'final_prob_node') or train_args.final_prob_node is None:
        train_args.final_prob_node = [1 - 1e-12, 1e-12]
        train_args.num_node_classes = 2
        train_args.has_node_feature = False

    evaluator = test_evaluator if args.split == 'test' else eval_evaluator

    # ── Reference statistics ────────────────────────────────────────────────────
    print('Computing reference graph statistics …')
    ref_stats  = _ref_statistics(evaluator)
    ref_nx     = _ref_graph(evaluator)         # None for GenericGraphEvaluator

    T = getattr(train_args, 'diffusion_steps', None)
    if T is None:
        raise RuntimeError('args.diffusion_steps not found in pickle.')
    print(f'Training T = {T}')

    # ── Model ───────────────────────────────────────────────────────────────────
    model = get_model(train_args, initial_graph_sampler=initial_graph_sampler)
    ckpt_path, ckpt = _load_ckpt(args.run_dir, args.checkpoint_epoch)
    _load_state_dict_compat(model, ckpt['model'], ckpt_path)
    saved_epoch = ckpt.get('current_epoch', '?')
    print(f'Loaded: {ckpt_path}  (epoch={saved_epoch})')

    model = model.to(device)
    model.eval()

    # ── Ablation loop ───────────────────────────────────────────────────────────
    fractions   = sorted(set(args.fractions))
    frac_labels = [f'{f:.1f}T' if f < 1.0 else 'T' for f in fractions]
    results     = []   # list of (mean, std, n_valid)

    print(f'\nAblation over fractions: {fractions}')
    print(f'Graphs per fraction: {args.num_samples}  |  batch size: {args.batch_size}')

    for idx, (frac, lbl) in enumerate(zip(fractions, frac_labels)):
        steps = max(1, int(T * frac))
        # Per-fraction seed offset for reproducibility across runs
        torch.manual_seed(args.seed + idx)
        np.random.seed(args.seed + idx)

        t0 = time.time()
        print(f'\n  [{lbl}]  inference_steps={steps} ({frac:.1f}×{T}) …',
              end=' ', flush=True)

        mean, std, n_valid = _generate_and_measure(
            model, args.num_samples, args.batch_size,
            steps if frac < 1.0 else None,   # None = full T, no skip schedule
            ref_nx, device)

        print(f'{n_valid}/{args.num_samples} valid graphs  [{time.time()-t0:.1f}s]')
        results.append((mean, std, n_valid))

    # ── Table ───────────────────────────────────────────────────────────────────
    print(f'\n\nDataset: {train_args.dataset}   Checkpoint epoch: {saved_epoch}   '
          f'Seed: {args.seed}   Split: {args.split}   T={T}')
    _print_table(frac_labels, ref_stats, results, T)

    # ── JSON output ─────────────────────────────────────────────────────────────
    if args.out:
        out = {
            '_meta': {
                'run_dir':          args.run_dir,
                'checkpoint_epoch': args.checkpoint_epoch,
                'saved_epoch':      saved_epoch,
                'dataset':          train_args.dataset,
                'split':            args.split,
                'T':                T,
                'fractions':        fractions,
                'num_samples':      args.num_samples,
                'seed':             args.seed,
            },
            'reference': {k: (float(v) if np.isfinite(v) else None)
                          for k, v in ref_stats.items()},
        }
        for frac, lbl, (mean, std, n_valid) in zip(fractions, frac_labels, results):
            out[lbl] = {
                'fraction':         frac,
                'inference_steps':  max(1, int(T * frac)) if frac < 1.0 else T,
                'n_valid_graphs':   n_valid,
                'mean': {k: (float(v) if np.isfinite(v) else None) for k, v in mean.items()},
                'std':  {k: (float(v) if np.isfinite(v) else None) for k, v in std.items()},
            }
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
