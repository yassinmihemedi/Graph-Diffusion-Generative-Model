import torch
import os
import pickle
import numpy as np
from diffusion.loss import elbo_bpd
from diffusion.utils import add_parent_path
from scipy import sparse as sp
import torch_geometric as pyg
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: no tkinter, safe in any thread
import matplotlib.pyplot as plt

add_parent_path(level=2)
from diffusion.experiment import DiffusionExperiment
from diffusion.experiment import add_exp_args as add_exp_args_parent

_ZINC_DATASETS = {'zinc', 'zinc12k', 'zinc250k'}

# 28-slot color palette for ZINC atom types (tab20 + tab20b first 8 columns).
# Index = atom type 0-27.  Wrapped modulo for any unseen types.
_MOL_NODE_COLORS = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors[:8])

def _draw_molecule_graph(g, ax):
    """Draw a molecule graph with atom-type colors and bond-width by type."""
    colors = [_MOL_NODE_COLORS[g.nodes[n].get('node_attr', 0) % len(_MOL_NODE_COLORS)]
              for n in g.nodes]
    widths = [max(0.5, g.edges[e].get('edge_attr_bond', 1) * 0.8) for e in g.edges]
    nx.draw(g, ax=ax, node_size=60, node_color=colors, width=widths,
            with_labels=False)


# Metrics of interest for the two hypotheses (H1: quality, H2: speed).
# Printed to stdout at every eval/test so they appear in log files, not just wandb.
_HYPOTHESIS_METRICS = {
    # Graph statistics (NetworkEvaluator — polblogs/cora)
    'nmae/power_law_exp':          'PLE  (nmae)',
    'nmae/cpl':                    'CPL  (nmae)',
    'nmae/clustering_coefficient': 'ClustCoeff (nmae)',
    'nmae/assortativity':          'Assort (nmae)',
    'nmae/triangle_count':         'Triangles (nmae)',
    'nmae/d':                      'AvgDegree (nmae)',
    'nmae/LCC':                    'LCC (nmae)',
    'value/power_law_exp':         'PLE  (value)',
    'value/cpl':                   'CPL  (value)',
    'value/clustering_coefficient':'ClustCoeff (value)',
    # MMD metrics (GenericGraphEvaluator — Ego)
    'degree_mmd':    'Degree MMD',
    'clustering_mmd':'Clustering MMD',
    'spectral_mmd':  'Spectral MMD',
    'orbits_mmd':    'Orbits MMD',
    'mmd_rbf':       'MMD-RBF (GIN)',
    'mmd_linear':    'MMD-Linear (GIN)',
    'fid':           'FID (GIN)',
    # Discriminator health
    'disc_real':     'Disc-real (→1)',
    'disc_fake':     'Disc-fake (→0)',
    # Proposition diagnostics (theory validation)
    'prop/p1/pred_density_mean': 'P1 pred_density_mean (→ρ if trapped)',
    'prop/p1/pred_density_std':  'P1 pred_density_std  (→0 if trapped)',
    'prop/p2/kl_frac_edge_raw':  'P2 kl_frac_edge_raw (should≈ρ raw)',
    'prop/p2/w_pos_mean':        'P2 w+ mean          (should≈(1-ρ)/ρ)',
    'prop/p3/xt_density':        'P3 xt_density       (should≈ρ all t)',
    'prop/p3/stationary_target': 'P3 stationary_target (=m1, should=ρ)',
    # Molecular metrics — ZINC only (structural, not chemical — no bond types generated)
    'mol/validity':   'Mol Validity  (structural ↑)',
    'mol/uniqueness': 'Mol Uniqueness (WL-hash  ↑)',
    'mol/novelty':    'Mol Novelty   (vs train  ↑)',
}


def _print_graph_metrics(metrics: dict, prefix: str = ''):
    """Print graph quality metrics to stdout so they appear in log files."""
    lines = []
    for key, label in _HYPOTHESIS_METRICS.items():
        if key in metrics:
            v = metrics[key]
            if v is None or (hasattr(v, '__float__') and __import__('math').isnan(float(v))):
                lines.append(f'  {label:<28} NaN')
            else:
                lines.append(f'  {label:<28} {float(v):.6f}')
    if lines:
        print(f'\n{prefix} graph metrics:')
        for l in lines:
            print(l)
        print()


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--num_generation', type=int, default=64)

class GraphExperiment(DiffusionExperiment):
    def _get_disc_optimizer(self):
        """Lazily create a dedicated Adam optimizer for the discriminator."""
        if not hasattr(self, '_disc_optimizer'):
            self._disc_optimizer = torch.optim.Adam(
                self.model.discriminator.parameters(), lr=1e-4)
        return self._disc_optimizer

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        data_count = 0
        disc_loss_sum  = 0.0
        disc_real_sum  = 0.0   # discriminator score on real samples (→ 1 when healthy)
        disc_fake_sum  = 0.0   # discriminator score on fake samples (→ 0 when healthy)
        disc_score_cnt = 0
        # accumulators for loss-component monitoring (vb_kl_ce + ScalableGNN)
        comp_sums = {'kl_edge': 0., 'kl_node': 0., 'nll_edge': 0., 'nll_node': 0.,
                     'kl_prior': 0., 'ce_edge': 0., 'ce_node': 0., 'elbo': 0.,
                     'L_recon': 0., 'L_G': 0., 'total': 0.}
        comp_count = 0
        # accumulators for proposition diagnostics — dynamic so ERGM and
        # BinomialDiffusion keys both accumulate without explicit enumeration.
        prop_diag_sums = {}
        prop_diag_count = 0
        has_disc = (getattr(self.model, 'discriminator', None) is not None
                    and self.model.disc_lambda > 0)
        if has_disc:
            self.model.discriminator.train()

        for pyg_data in self.train_loader:
            self.optimizer.zero_grad()
            pyg_data = pyg_data.to(self.args.device)
            loss = elbo_bpd(self.model, pyg_data)

            # Guard: skip batches with non-finite OR unreasonably large loss.
            # A BPD > 100 bits/dim is numerical explosion (not a learning signal)
            # and would poison the Lt_history importance-sampling weights.
            if not torch.isfinite(loss) or loss.item() > 100.0:
                data_count += pyg_data.num_graphs
                tag = 'NaN/Inf' if not torch.isfinite(loss) else f'exploded({loss.item():.1f})'
                print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f} [skip: {}]'.format(
                    epoch+1, self.args.epochs, data_count,
                    len(self.train_loader.dataset), loss_sum/max(loss_count, 1), tag), end='\r')
                continue

            loss.backward()

            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()

            # collect per-component values when using vb_kl_ce
            if hasattr(self.model, '_loss_components'):
                comps = self.model._loss_components
                for k in comp_sums:
                    comp_sums[k] += comps.get(k, 0.0)
                comp_count += 1

            # collect proposition diagnostics (P1/P2/P3 / ERGM)
            _base_m = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(_base_m, '_prop_diagnostics'):
                pd = _base_m._prop_diagnostics
                for k, v in pd.items():
                    prop_diag_sums[k] = prop_diag_sums.get(k, 0.0) + v
                prop_diag_count += 1

            if has_disc:
                disc_opt = self._get_disc_optimizer()
                disc_opt.zero_grad()
                d_loss = self.model.disc_loss(pyg_data)
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1.0)
                    disc_opt.step()
                    disc_loss_sum += d_loss.detach().cpu().item()
                    # Collect real/fake scores from _disc_components
                    _base = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(_base, '_disc_components'):
                        disc_real_sum  += _base._disc_components.get('disc_real', 0.)
                        disc_fake_sum  += _base._disc_components.get('disc_fake', 0.)
                        disc_score_cnt += 1

            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
            loss_count += pyg_data.num_graphs
            data_count += pyg_data.num_graphs

            # per-batch terminal line
            if comp_count > 0:
                c = {k: v / comp_count for k, v in comp_sums.items()}
                # ScalableGNN uses L_recon/L_G keys; BinomialDiffusion uses kl_*/ce_*
                if c.get('L_recon', 0.) > 0 or c.get('total', 0.) > 0:
                    print('Epoch {}/{} [{}/{}] bpd={:.3f} | '
                          'L_recon={:.3f} L_G={:.4f} total={:.3f}'.format(
                              epoch+1, self.args.epochs, data_count,
                              len(self.train_loader.dataset),
                              loss_sum / loss_count,
                              c['L_recon'], c['L_G'], c['total']), end='\r')
                else:
                    print('Epoch {}/{} [{}/{}] bpd={:.3f} | '
                          'kl_e={:.3f} kl_n={:.3f} kl_prior={:.3f} | '
                          'ce_e={:.3f} ce_n={:.3f}'.format(
                              epoch+1, self.args.epochs, data_count,
                              len(self.train_loader.dataset),
                              loss_sum / loss_count,
                              c['kl_edge'], c['kl_node'], c['kl_prior'],
                              c['ce_edge'], c['ce_node']), end='\r')
            else:
                print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                    epoch+1, self.args.epochs, data_count,
                    len(self.train_loader.dataset), loss_sum/loss_count), end='\r')

        if self.scheduler_epoch: self.scheduler_epoch.step()
        metrics = {'bpd': loss_sum / loss_count, 'lr': self.optimizer.param_groups[0]['lr']}
        if has_disc:
            n_disc = max(len(self.train_loader), 1)
            metrics['disc_loss'] = disc_loss_sum / n_disc
            if disc_score_cnt > 0:
                metrics['disc_real'] = disc_real_sum  / disc_score_cnt
                metrics['disc_fake'] = disc_fake_sum  / disc_score_cnt
                print('  [disc scores]  real={:.3f}  fake={:.3f}  '
                      '(healthy: real→1, fake→0)'.format(
                          disc_real_sum / disc_score_cnt,
                          disc_fake_sum / disc_score_cnt))
        if comp_count > 0:
            c_ep = {k: v / comp_count for k, v in comp_sums.items()}
            print()  # newline after \r so epoch summary is visible
            if c_ep.get('L_recon', 0.) > 0 or c_ep.get('total', 0.) > 0:
                # ScalableGNN — only log its own meaningful keys, not the kl/nll zeros
                for k in ('L_recon', 'L_G', 'total'):
                    metrics[f'loss/{k}'] = c_ep[k]
                print('  [ScalableGNN loss] '
                      'L_recon={:.4f}  L_G={:.4f}  total={:.4f}'.format(
                          c_ep['L_recon'], c_ep['L_G'], c_ep['total']))
            else:
                # BinomialDiffusion (vb_kl_ce) — log only non-zero keys
                for k in ('kl_edge', 'kl_node', 'nll_edge', 'nll_node',
                          'kl_prior', 'ce_edge', 'ce_node', 'elbo'):
                    if c_ep.get(k, 0.) != 0.:
                        metrics[f'loss/{k}'] = c_ep[k]
                print('  [loss components] '
                      'kl_edge={:.4f}  kl_node={:.4f}  kl_prior={:.4f}  '
                      'ce_edge={:.4f}  ce_node={:.4f}  elbo={:.4f}'.format(
                          c_ep['kl_edge'], c_ep['kl_node'], c_ep['kl_prior'],
                          c_ep['ce_edge'], c_ep['ce_node'], c_ep['elbo']))
        if prop_diag_count > 0:
            if comp_count == 0:
                print()  # newline after the \r per-batch progress line
            pd_ep = {k: v / prop_diag_count for k, v in prop_diag_sums.items()}
            for k, v in pd_ep.items():
                metrics[f'prop/{k}'] = v
            if 'ergm/Q_frac' in pd_ep:
                # ERGM-Anchored Diffusion diagnostics
                print('  [ERGM] Q_frac={:.3f}  ht_lambda={:.4f}  tau_q={:.2f}'.format(
                    pd_ep.get('ergm/Q_frac', 0.), pd_ep.get('ergm/ht_lambda_mean', 0.),
                    pd_ep.get('ergm/tau_q_mean', 0.)))
                print('  [P2]   w_plus={:.1f}  '
                      '[P3]  xt_edge_density={:.4f}  xt_mask_frac={:.3f}'.format(
                    pd_ep.get('p2/w_plus_masked_mean', 0.),
                    pd_ep.get('p3/xt_edge_density', 0.),
                    pd_ep.get('p3/xt_mask_fraction', 0.)))
                print('  [P1]   pred_density={:.4f}'.format(
                    pd_ep.get('p1/pred_density_mean', 0.)))
            else:
                # BinomialDiffusion (original) diagnostics
                print('  [P1] pred_density  mean={:.4f}  std={:.4f}  '
                      '(theory: const-predictor → mean≈ρ, std≈0)'.format(
                          pd_ep.get('p1/pred_density_mean', 0.),
                          pd_ep.get('p1/pred_density_std', 0.)))
                print('  [P2] kl_frac_edge_raw={:.4f}  w+={:.1f}  '
                      '(theory: raw≈ρ before fix; w+ should drive weighted frac→0.5)'.format(
                          pd_ep.get('p2/kl_frac_edge_raw', 0.),
                          pd_ep.get('p2/w_pos_mean', 0.)))
                print('  [P3] xt_density={:.4f}  stationary_target={:.4f}  '
                      '(theory: marginal → xt_density≈target≈ρ for all t)'.format(
                          pd_ep.get('p3/xt_density', 0.),
                          pd_ep.get('p3/stationary_target', 0.)))
        return metrics

    def eval_fn(self, epoch):
        self.model.eval()
        eval_dict = {}
        _base = self.model.module if hasattr(self.model, 'module') else self.model
        has_disc = (getattr(_base, 'discriminator', None) is not None
                    and getattr(_base, 'disc_lambda', 0.0) > 0)
        disc_real_sum = disc_fake_sum = disc_loss_sum = 0.0
        disc_n = 0

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.eval_loader:
                pyg_data = pyg_data.to(self.args.device)
                loss = elbo_bpd(self.model, pyg_data)
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
                loss_count += pyg_data.num_graphs
                data_count += pyg_data.num_graphs

                # Discriminator health on eval data — same computation as training
                # but without backward (torch.no_grad is active)
                if has_disc:
                    try:
                        d_loss = _base.disc_loss(pyg_data)
                        if torch.isfinite(d_loss):
                            disc_loss_sum += d_loss.item()
                            if hasattr(_base, '_disc_components'):
                                disc_real_sum += _base._disc_components.get('disc_real', 0.)
                                disc_fake_sum += _base._disc_components.get('disc_fake', 0.)
                                disc_n += 1
                    except Exception:
                        pass

            print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                epoch+1, self.args.epochs, data_count,
                len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            eval_dict['bpd'] = loss_sum / loss_count
            if has_disc and disc_n > 0:
                eval_dict['disc_loss'] = disc_loss_sum / disc_n
                eval_dict['disc_real'] = disc_real_sum / disc_n
                eval_dict['disc_fake'] = disc_fake_sum / disc_n
                print('  [eval disc]  real={:.3f}  fake={:.3f}  '
                      '(healthy: real→1, fake→0)'.format(
                          disc_real_sum / disc_n, disc_fake_sum / disc_n))

            # Sampling + evaluation: only rank 0 generates graphs and writes plots.
            # Under DDP all ranks reach this point, but only the main process acts.
            # Free the last eval batch from GPU — for PubMed the eval pyg_data
            # holds full_edge_index (388M pairs ≈ 14 GB) that blocks sampling.
            try:
                del pyg_data
            except NameError:
                pass
            torch.cuda.empty_cache()
            if self.is_main_process:
                try:
                    infer_steps = getattr(self.args, 'inference_steps', None)
                    generated_pyg_datas = self.model.sample(
                        self.args.num_generation, inference_steps=infer_steps)
                    generated_graphs = []
                    typed_graphs     = []  # for molecule visualization (typed attrs)
                    _is_mol = getattr(self.args, 'dataset', '') in _ZINC_DATASETS

                    pyg_data_list = generated_pyg_datas.to_data_list()
                    for pyg_data in pyg_data_list:
                        # Plain topology graph — used for structural MMD evaluation
                        g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
                        generated_graphs.append(g_gen)
                        if _is_mol:
                            # Typed graph — used for molecule visualization only
                            _node_attrs = ['node_attr'] if hasattr(pyg_data, 'node_attr') else []
                            _edge_attrs = ['edge_attr_bond'] if hasattr(pyg_data, 'edge_attr_bond') else []
                            g_typed = pyg.utils.to_networkx(
                                pyg_data, node_attrs=_node_attrs,
                                edge_attrs=_edge_attrs, to_undirected=True)
                            typed_graphs.append(g_typed)

                    # Always log edge stats — visible in W&B even when generated
                    # graphs are empty (graph statistics return NaN, only bpd shows).
                    eval_dict['gen/mean_edges'] = float(np.mean(
                        [g.number_of_edges() for g in generated_graphs]))
                    eval_dict['gen/density'] = float(np.mean([
                        g.number_of_edges() / max(1, g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
                        for g in generated_graphs]))
                    print(f'\n[eval] mean_edges={eval_dict["gen/mean_edges"]:.1f}  '
                          f'density={eval_dict["gen/density"]:.4f}')

                    w = 8 if self.args.num_generation >= 64 else 2
                    fig, axes = plt.subplots(w, w, figsize=(17,17))
                    draw_list = typed_graphs if _is_mol and typed_graphs else generated_graphs
                    for i, g_draw in enumerate(draw_list[:w**2]):
                        if _is_mol:
                            _draw_molecule_graph(g_draw, axes[i%w][i//w])
                        else:
                            nx.draw(g_draw, ax=axes[i%w][i//w], node_size=30)
                    plt.savefig(os.path.join(self.log_path, f"eval/sample{epoch}.png"))
                    plt.close()

                    # Structural MMD metrics (all datasets).
                    # Always evaluate — empty graphs return NaN values which W&B
                    # logs as gaps. Skipping entirely means metric keys never
                    # appear in W&B at all (not even as gaps), breaking charts.
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        metrics = self.eval_evaluator.evaluate(generated_graphs)
                    eval_dict.update(metrics)
                    n_empty = sum(1 for g in generated_graphs if g.number_of_edges() == 0)
                    if n_empty > 0:
                        print(f'[eval] {n_empty}/{len(generated_graphs)} generated graphs empty')
                    _print_graph_metrics(metrics, prefix='[eval]')

                    # Molecular metrics (ZINC only): validity / uniqueness / novelty.
                    # Computed on typed_graphs (atom type + topology) not topology-only.
                    if _is_mol and typed_graphs:
                        from eval_utils.molecular_metrics import zinc_metrics
                        _mol_hashes  = getattr(self.eval_evaluator, 'mol_train_hashes', frozenset())
                        _mol_valence = getattr(self.eval_evaluator, 'mol_max_valence',  {})
                        mol_met = zinc_metrics(typed_graphs, _mol_hashes, _mol_valence)
                        eval_dict.update(mol_met)
                        print(f'[eval mol] validity={mol_met["mol/validity"]:.3f}  '
                              f'uniqueness={mol_met["mol/uniqueness"]:.3f}  '
                              f'novelty={mol_met["mol/novelty"]:.3f}')
                except Exception as e:
                    import traceback
                    print(f'\n[eval] sampling/evaluation failed at epoch {epoch+1}: {e}')
                    print(traceback.format_exc())
                    torch.cuda.empty_cache()

        return eval_dict

    def test_fn(self, epoch):
        self.model.eval()
        test_dict = {}
        _base = self.model.module if hasattr(self.model, 'module') else self.model
        has_disc = (getattr(_base, 'discriminator', None) is not None
                    and getattr(_base, 'disc_lambda', 0.0) > 0)
        disc_real_sum = disc_fake_sum = disc_loss_sum = 0.0
        disc_n = 0

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.test_loader:
                pyg_data = pyg_data.to(self.args.device)
                loss = elbo_bpd(self.model, pyg_data)
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
                loss_count += pyg_data.num_graphs
                data_count += pyg_data.num_graphs

                if has_disc:
                    try:
                        d_loss = _base.disc_loss(pyg_data)
                        if torch.isfinite(d_loss):
                            disc_loss_sum += d_loss.item()
                            if hasattr(_base, '_disc_components'):
                                disc_real_sum += _base._disc_components.get('disc_real', 0.)
                                disc_fake_sum += _base._disc_components.get('disc_fake', 0.)
                                disc_n += 1
                    except Exception:
                        pass

            print('Test evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                epoch+1, self.args.epochs, data_count,
                len(self.test_loader.dataset), loss_sum/loss_count), end='\r')
            test_dict['bpd'] = loss_sum / loss_count
            if has_disc and disc_n > 0:
                test_dict['disc_loss'] = disc_loss_sum / disc_n
                test_dict['disc_real'] = disc_real_sum / disc_n
                test_dict['disc_fake'] = disc_fake_sum / disc_n
                print('  [test disc]  real={:.3f}  fake={:.3f}  '
                      '(healthy: real→1, fake→0)'.format(
                          disc_real_sum / disc_n, disc_fake_sum / disc_n))

            try:
                del pyg_data
            except NameError:
                pass
            torch.cuda.empty_cache()
            if self.is_main_process:
                try:
                    # Bug fix: pass inference_steps so H2 (50-step) tests consistently
                    infer_steps = getattr(self.args, 'inference_steps', None)
                    generated_pyg_datas = self.model.sample(
                        self.args.num_generation, inference_steps=infer_steps)
                    generated_graphs = []
                    typed_graphs     = []
                    _is_mol = getattr(self.args, 'dataset', '') in _ZINC_DATASETS

                    pyg_data_list = generated_pyg_datas.to_data_list()
                    for pyg_data in pyg_data_list:
                        g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
                        generated_graphs.append(g_gen)
                        if _is_mol:
                            _node_attrs = ['node_attr'] if hasattr(pyg_data, 'node_attr') else []
                            _edge_attrs = ['edge_attr_bond'] if hasattr(pyg_data, 'edge_attr_bond') else []
                            g_typed = pyg.utils.to_networkx(
                                pyg_data, node_attrs=_node_attrs,
                                edge_attrs=_edge_attrs, to_undirected=True)
                            typed_graphs.append(g_typed)

                    test_dict['gen/mean_edges'] = float(np.mean(
                        [g.number_of_edges() for g in generated_graphs]))
                    test_dict['gen/density'] = float(np.mean([
                        g.number_of_edges() / max(1, g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
                        for g in generated_graphs]))
                    print(f'\n[test] mean_edges={test_dict["gen/mean_edges"]:.1f}  '
                          f'density={test_dict["gen/density"]:.4f}')

                    w = 8 if self.args.num_generation >= 64 else 2
                    fig, axes = plt.subplots(w, w, figsize=(17,17))
                    draw_list = typed_graphs if _is_mol and typed_graphs else generated_graphs
                    for i, g_draw in enumerate(draw_list[:w**2]):
                        if _is_mol:
                            _draw_molecule_graph(g_draw, axes[i%w][i//w])
                        else:
                            nx.draw(g_draw, ax=axes[i%w][i//w], node_size=30)
                    plt.savefig(os.path.join(self.log_path, f"test/sample{epoch}.png"))
                    plt.close()

                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        metrics = self.test_evaluator.evaluate(generated_graphs)
                    test_dict.update(metrics)
                    n_empty = sum(1 for g in generated_graphs if g.number_of_edges() == 0)
                    if n_empty > 0:
                        print(f'[test] {n_empty}/{len(generated_graphs)} generated graphs empty')
                    _print_graph_metrics(metrics, prefix='[test]')

                    # Molecular metrics (ZINC only): validity / uniqueness / novelty.
                    if _is_mol and typed_graphs:
                        from eval_utils.molecular_metrics import zinc_metrics
                        _mol_hashes  = getattr(self.test_evaluator, 'mol_train_hashes', frozenset())
                        _mol_valence = getattr(self.test_evaluator, 'mol_max_valence',  {})
                        mol_met = zinc_metrics(typed_graphs, _mol_hashes, _mol_valence)
                        test_dict.update(mol_met)
                        print(f'[test mol] validity={mol_met["mol/validity"]:.3f}  '
                              f'uniqueness={mol_met["mol/uniqueness"]:.3f}  '
                              f'novelty={mol_met["mol/novelty"]:.3f}')
                except Exception as e:
                    import traceback
                    print(f'\n[test] sampling/evaluation failed at epoch {epoch+1}: {e}')
                    print(traceback.format_exc())
                    torch.cuda.empty_cache()

        return test_dict