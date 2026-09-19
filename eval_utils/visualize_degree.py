"""
eval_utils/visualize_degree.py — Degree-distribution visualization pipeline.

Loads a trained checkpoint exactly the way eval_utils/evaluate.py does (same
args.pickle → get_data() → get_model() → checkpoint state_dict flow), samples
graphs from it, and plots Reference vs. Generated node-degree distributions as
three separate single-panel PDFs: histogram, (horizontal split) violin, and
log-log CCDF.

This does NOT re-run evaluator.evaluate() — it only needs the reference graphs
and the generated graphs, so checkpoint/data loading is reused directly from
eval_utils.evaluate (imported, not duplicated).

Usage:
    python eval_utils/visualize_degree.py \
        --run_dir ./wandb/cora/.../cora_ergm_20250101_120000 \
        --checkpoint_epoch 4999 \
        --num_samples 16 \
        --batch_size 4 \
        --split test \
        --device cuda:0 \
        --out figures/cora_degree_distribution

Writes three files: figures/cora_degree_distribution_histogram.pdf,
_violin.pdf and _ccdf.pdf.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data import get_data
from model import get_model
from eval_utils.evaluate import (
    _load_checkpoint, _load_state_dict_compat, _resolve_inference_steps, _to_lcc,
    _detect_legacy_ergm_embed,
)

# ── Paper-wide style constants (mirrors scripts/plot_template.py) ─────────────
COLORS = {
    'reference': '#1B1B1B',   # dark, neutral — reference/real data
    'generated': '#0F6E56',   # teal — generated
}
LINESTYLES = {'reference': '-', 'generated': '--'}

plt.rcParams.update({
    'font.size':        11,
    'axes.labelsize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'pdf.fonttype':     42,     # embed TrueType fonts — text stays selectable
    'font.family':      'serif',
})


# ─── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate graphs from a checkpoint and plot degree-'
                     'distribution comparisons (histogram / violin / CCDF).')
    p.add_argument('--run_dir', type=str, required=True,
        help='Full path to the run directory containing args.pickle and check/.')
    p.add_argument('--checkpoint_epoch', type=int, default=None,
        help='0-indexed training epoch to load: check/checkpoint_{epoch}.pt. '
             'Omit to load check/checkpoint.pt.')
    p.add_argument('--num_samples', type=int, default=16,
        help='Total number of graphs to generate (default 16).')
    p.add_argument('--batch_size', type=int, default=4,
        help='Graphs per model.sample() call (default 4).')
    p.add_argument('--split', type=str, default='test', choices=['eval', 'test'],
        help='Reference set to compare against: "test" (default) or "eval".')
    p.add_argument('--device', type=str, default=None,
        help='Torch device string. Default: cuda:0 if available, else cpu.')
    p.add_argument('--seed', type=int, default=0,
        help='RNG seed for reproducible sampling (default 0).')
    p.add_argument('--inference_steps', type=int, default=None,
        help='Absolute number of reverse diffusion steps. None = full T.')
    p.add_argument('--steps_fraction', type=float, default=None,
        help='Inference steps as a fraction of training T (overrides '
             '--inference_steps if both are set).')
    p.add_argument('--out', type=str, default=None,
        help='Output base path (no extension). Three files are written: '
             '<out>_histogram.pdf, <out>_violin.pdf, <out>_ccdf.pdf. '
             'Default: figures/<run_name>_degree_epoch<E>')
    return p.parse_args()


# ─── Graph generation (mirrors eval_utils/evaluate.py main()) ────────────────

def _generate_graphs(model, total, batch_size, inference_steps):
    generated = []
    n_batches = (total + batch_size - 1) // batch_size
    print(f'\nGenerating {total} graph(s) ({n_batches} batch(es) of ≤{batch_size}) …')
    with torch.no_grad():
        for b_idx in range(n_batches):
            n_this = min(batch_size, total - b_idx * batch_size)
            t_b = time.time()
            batch_out = model.sample(n_this, inference_steps=inference_steps)
            for pyg_data in batch_out.to_data_list():
                g_topo, _ = _to_lcc(pyg_data)
                if g_topo is not None:
                    generated.append(g_topo)
            print(f'  Batch {b_idx+1}/{n_batches}: {time.time()-t_b:.1f}s '
                  f'(valid so far: {len(generated)})')
    if not generated:
        raise RuntimeError('All generated graphs were empty — verify the checkpoint.')
    return generated


def _reference_graphs(evaluator):
    """Recover a list of nx.Graph reference(s) from either evaluator type."""
    if hasattr(evaluator, 'reference'):
        return [evaluator.reference]
    if hasattr(evaluator, 'references'):
        # evaluator.references are dgl graphs (see datasets/evaluator.py);
        # nx.Graph(g.cpu().to_networkx()) is the conversion already used
        # throughout eval_utils/evaluation/graph_structure_evaluation.py.
        return [nx.Graph(g.cpu().to_networkx()) for g in evaluator.references]
    raise RuntimeError('Evaluator exposes neither .reference nor .references.')


def _pooled_degrees(graphs):
    degs = []
    for g in graphs:
        degs.extend(d for _, d in g.degree())
    return np.asarray(degs, dtype=float)


def _save_generated_edgelists(graphs, out_base):
    """Write each generated graph's edge list as plain text.

    Lets the degree distribution (or any other statistic) be recomputed and
    re-plotted later in a different format without re-running sampling.
    Lossless here specifically because every graph in `graphs` is a
    largest-connected-component (see _to_lcc): every node has degree >= 1
    and so appears in at least one edge line — nx.write_edgelist (which
    only round-trips nodes that appear in an edge) drops nothing.
    """
    graphs_dir = f'{out_base}_graphs'
    os.makedirs(graphs_dir, exist_ok=True)
    for i, g in enumerate(graphs):
        nx.write_edgelist(
            g, os.path.join(graphs_dir, f'generated_{i:03d}.edgelist'), data=False)
    return graphs_dir


# ─── Plotting ─────────────────────────────────────────────────────────────

# Fill alpha is kept separate from edge/hatch alpha: histograms and violins
# use a translucent RGBA facecolor (so overlapping Reference/Generated mass
# is visible through both) while edges and hatch lines stay fully opaque for
# legibility in print and grayscale.
FILL_ALPHA = 0.35

FIGSIZE = (3.3, 2.6)   # single-column width for a two-column paper


def _rgba(color, alpha):
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def _ccdf(x):
    xs = np.sort(x)[::-1]
    n = len(xs)
    ys = np.arange(1, n + 1) / n
    return xs, ys


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # A draw pass before the tight-bbox save is required so the layout
    # engine's text extents (esp. an outer axis label) are fully accounted
    # for — otherwise bbox_inches='tight' can clip it.
    fig.canvas.draw()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_histogram(ref_degrees, gen_degrees, out_path):
    """log10(Degree + 1) density histogram, Reference vs. Generated."""
    ref_t = np.log10(ref_degrees + 1)
    gen_t = np.log10(gen_degrees + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    bins = np.linspace(0, max(ref_t.max(), gen_t.max()) + 1e-6, 41)
    ax.hist(ref_t, bins=bins, density=True,
            facecolor=_rgba(COLORS['reference'], FILL_ALPHA),
            edgecolor=COLORS['reference'], linewidth=0.8)
    ax.hist(gen_t, bins=bins, density=True,
            facecolor=_rgba(COLORS['generated'], FILL_ALPHA),
            edgecolor=COLORS['generated'], linewidth=0.8, hatch='///')
    ax.axvline(ref_t.mean(), color=COLORS['reference'], linestyle='--', linewidth=1.3)
    ax.axvline(gen_t.mean(), color=COLORS['generated'], linestyle='--', linewidth=1.3)
    ax.set_xlabel('log$_{10}$(Degree + 1)')
    ax.set_ylabel('Density')

    legend_handles = [
        Patch(facecolor=_rgba(COLORS['reference'], FILL_ALPHA), edgecolor=COLORS['reference'], label='Reference'),
        Patch(facecolor=_rgba(COLORS['generated'], FILL_ALPHA), edgecolor=COLORS['generated'], hatch='///', label='Generated'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.3, label='Mean'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    _save(fig, out_path)


def plot_violin(ref_degrees, gen_degrees, out_path):
    """Horizontal split violin of log10(Degree + 1), with a dashed mean whisker
    that deliberately extends past the violin body so it reads clearly even
    where the body itself is narrow."""
    ref_t = np.log10(ref_degrees + 1)
    gen_t = np.log10(gen_degrees + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    violin_width = 0.8
    parts = ax.violinplot([ref_t, gen_t], vert=False, showmeans=False,
                           showextrema=False, widths=violin_width)
    for i, pc in enumerate(parts['bodies']):
        color = COLORS['reference'] if i == 0 else COLORS['generated']
        pc.set_facecolor(_rgba(color, FILL_ALPHA))
        pc.set_edgecolor(color)
        pc.set_linewidth(1.0)

    mean_half_len = violin_width / 2 + 0.15   # overhangs the violin body
    for pos, vals in [(1, ref_t), (2, gen_t)]:
        m = vals.mean()
        ax.plot([m, m], [pos - mean_half_len, pos + mean_half_len],
                color='black', linestyle='--', linewidth=1.6, zorder=5)

    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Reference', 'Generated'])
    ax.set_xlabel('log$_{10}$(Degree + 1)')

    legend_handles = [
        Patch(facecolor=_rgba(COLORS['reference'], FILL_ALPHA), edgecolor=COLORS['reference'], label='Reference'),
        Patch(facecolor=_rgba(COLORS['generated'], FILL_ALPHA), edgecolor=COLORS['generated'], label='Generated'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.6, label='Mean'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    _save(fig, out_path)


def plot_ccdf(ref_degrees, gen_degrees, out_path):
    """Log-log complementary CDF of (Degree + 1), Reference vs. Generated."""
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    xs_r, ys_r = _ccdf(ref_degrees + 1)
    xs_g, ys_g = _ccdf(gen_degrees + 1)
    ax.step(xs_r, ys_r, where='post', color=COLORS['reference'],
            linestyle=LINESTYLES['reference'], linewidth=1.6)
    ax.step(xs_g, ys_g, where='post', color=COLORS['generated'],
            linestyle=LINESTYLES['generated'], linewidth=1.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree + 1')
    ax.set_ylabel('P(Degree + 1 $\\geq$ x)')

    legend_handles = [
        Line2D([0], [0], color=COLORS['reference'], linestyle=LINESTYLES['reference'], linewidth=1.6, label='Reference'),
        Line2D([0], [0], color=COLORS['generated'], linestyle=LINESTYLES['generated'], linewidth=1.6, label='Generated'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    _save(fig, out_path)


def plot_degree_comparison(ref_degrees, gen_degrees, out_base):
    """Write the three degree-distribution figures and return their paths."""
    paths = {
        'histogram': f'{out_base}_histogram.pdf',
        'violin':    f'{out_base}_violin.pdf',
        'ccdf':      f'{out_base}_ccdf.pdf',
    }
    plot_histogram(ref_degrees, gen_degrees, paths['histogram'])
    plot_violin(ref_degrees, gen_degrees, paths['violin'])
    plot_ccdf(ref_degrees, gen_degrees, paths['ccdf'])
    return paths


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    eval_args = _parse_args()

    torch.manual_seed(eval_args.seed)
    np.random.seed(eval_args.seed)

    if eval_args.device is not None:
        device = eval_args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    import pickle
    args_path = os.path.join(eval_args.run_dir, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f'args.pickle not found at {args_path}.')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    args.device = device
    args.num_workers = 0
    args.pin_memory = False
    args.batch_size = 1

    print(f'Loading dataset: {args.dataset} …')
    (train_loader, eval_loader, test_loader,
     num_node_feat, num_node_classes, num_edge_classes,
     max_degree, augmented_feature_dict,
     initial_graph_sampler,
     eval_evaluator, test_evaluator,
     monitoring_statistics, train_density) = get_data(args)

    if not hasattr(args, 'train_density') or args.train_density is None:
        args.train_density = train_density
    if not hasattr(args, 'has_node_feature'):
        args.has_node_feature = False
    if not hasattr(args, 'final_prob_node') or args.final_prob_node is None:
        args.final_prob_node = [1 - 1e-12, 1e-12]
        args.num_node_classes = 2
        args.has_node_feature = False

    # Checkpoint must be loaded BEFORE get_model() so args.legacy_ergm_embed
    # can be set from its actual key shapes — see _detect_legacy_ergm_embed.
    ckpt_path, ckpt = _load_checkpoint(eval_args.run_dir, eval_args.checkpoint_epoch)
    args.legacy_ergm_embed = _detect_legacy_ergm_embed(ckpt['model'])
    if args.legacy_ergm_embed:
        print('[compat] Detected legacy GraphDenoisingERGM checkpoint '
              '(single-Linear node_embed, no node_embed_t, 2-layer tau_proj). '
              'Building model in the matching legacy shape.')

    model = get_model(args, initial_graph_sampler=initial_graph_sampler)
    _load_state_dict_compat(model, ckpt['model'], ckpt_path)
    saved_epoch = ckpt.get('current_epoch', 'unknown')
    print(f'Loaded checkpoint: {ckpt_path}  (saved epoch {saved_epoch})')

    model = model.to(device)
    model.eval()

    inference_steps = _resolve_inference_steps(
        eval_args, getattr(args, 'diffusion_steps', None))

    if eval_args.out:
        out_base = eval_args.out
        if out_base.endswith('.pdf'):
            out_base = out_base[:-len('.pdf')]
    else:
        run_name = os.path.basename(os.path.normpath(eval_args.run_dir))
        epoch_tag = eval_args.checkpoint_epoch if eval_args.checkpoint_epoch is not None else 'best'
        out_base = f'figures/{run_name}_degree_epoch{epoch_tag}'

    generated = _generate_graphs(
        model, eval_args.num_samples, eval_args.batch_size, inference_steps)

    graphs_dir = _save_generated_edgelists(generated, out_base)
    print(f'Saved {len(generated)} generated graph edge list(s) → {graphs_dir}/')

    evaluator = test_evaluator if eval_args.split == 'test' else eval_evaluator
    ref_graphs = _reference_graphs(evaluator)

    ref_degrees = _pooled_degrees(ref_graphs)
    gen_degrees = _pooled_degrees(generated)

    print(f'\nReference: {len(ref_graphs)} graph(s), {len(ref_degrees)} nodes, '
          f'mean degree {ref_degrees.mean():.3f}')
    print(f'Generated: {len(generated)} graph(s), {len(gen_degrees)} nodes, '
          f'mean degree {gen_degrees.mean():.3f}')

    paths = plot_degree_comparison(ref_degrees, gen_degrees, out_base)
    print('\nSaved figures:')
    for name, path in paths.items():
        print(f'  {name:<10s} {path}')


if __name__ == '__main__':
    main()
