import csv as _csv
import math
import torch
import os
import networkx as nx
import numpy as np

import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from datasets.data_utils import EmpiricalEmptyGraphGenerator, NeuralEmptyGraphGenerator, preprocess, resample_neg_edges, collate_fn, FEATURE_EXTRACTOR
from datasets.evaluator import NetworkEvaluator, GenericGraphEvaluator
from diffusion.utils import str_to_bool


class NetworkDataset(Dataset):
    def __init__(self, pyg_graph, num_iter, transform=None):
        super().__init__()
        self.pyg_data = pyg_graph
        self.transform = transform
        self.num_iter = num_iter

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.pyg_graph)
        resample_neg_edges(self.pyg_data)   # refresh sparse non-edge sample each step
        return self.pyg_data

    def __len__(self):
        return self.num_iter


class GraphDataset(Dataset):
    def __init__(self, pyg_datas):
        super().__init__()
        self.pyg_datas = pyg_datas

    def __getitem__(self, index):
        return self.pyg_datas[index]#, self.denses[index]

    def __len__(self):
        return len(self.pyg_datas)


def _load_elliptic_node_meta(node_csv):
    """Parse node CSV → (txid_to_nodeid, node_to_ts) dicts.

    node_id   : sequential int 0..N-1 assigned in the merged CSV
    txId      : original Elliptic transaction identifier used in the edge file
    Time step : int 1..49 (with gaps)
    """
    txid_to_nodeid, node_to_ts = {}, {}
    with open(node_csv) as f:
        for row in _csv.DictReader(f):
            nid  = int(row['node_id'])
            txid = int(row['txId'])
            ts   = int(row['Time step'])
            txid_to_nodeid[txid] = nid
            node_to_ts[nid]      = ts
    return txid_to_nodeid, node_to_ts


def _load_elliptic_edges(edge_csv, txid_to_nodeid, node_to_ts):
    """Parse edge CSV → list of (src_node_id, dst_node_id).

    Auto-detects column format:
      txId1/txId2       — original Elliptic txId references (maps through txid_to_nodeid)
      node_id1/node_id2 — pre-mapped sequential IDs matching merged_nodes_final.csv
      source/target     — pre-mapped sequential IDs (e.g. merged_edges_gaussian_timestamp.csv);
                           treated identically to node_id1/node_id2
      first two cols    — fallback, treated as txId references

    Self-loops are dropped.

    Edges where either endpoint is absent from the node CSV are dropped.
    This implements the induced-subgraph semantics: merged_nodes_final.csv is
    a deliberate induced subgraph of the full 203K-node Elliptic graph (chosen
    for GPU feasibility), so only edges whose BOTH endpoints belong to the
    18,945-node induced set are structurally valid for generation.
    When using the full Elliptic edge file as input, the discarded entries are
    edges to nodes outside the induced subgraph — not structural information loss.
    When using a pre-filtered edge file for the 18,945 nodes, nothing is dropped.
    """
    raw_edges = []
    with open(edge_csv) as f:
        reader = _csv.DictReader(f)
        cols = reader.fieldnames or []
        if 'txId1' in cols and 'txId2' in cols:
            c1, c2, use_txid = 'txId1', 'txId2', True
        elif 'node_id1' in cols and 'node_id2' in cols:
            c1, c2, use_txid = 'node_id1', 'node_id2', False
        elif 'source' in cols and 'target' in cols:
            c1, c2, use_txid = 'source', 'target', False
        else:
            c1, c2, use_txid = cols[0], cols[1], True

        for row in reader:
            s_raw, d_raw = int(row[c1]), int(row[c2])
            if use_txid:
                s = txid_to_nodeid.get(s_raw)
                d = txid_to_nodeid.get(d_raw)
                if s is None or d is None:
                    continue
            else:
                s, d = s_raw, d_raw
            if s != d:
                raw_edges.append((s, d))
    return raw_edges


def load_elliptic_ts_graphs(node_csv, edge_csv, min_nodes=50, cache_path=None):
    """Build per-timestep undirected LCC graphs from the Elliptic induced subgraph.

    merged_nodes_final.csv is an induced subgraph of the full 203K-node Elliptic
    Bitcoin transaction graph, selected for GPU feasibility (N=18,945 nodes,
    N(N-1)/2≈179M pairs).  Each node carries a 'Time step' label (1–49, 46
    distinct values) identifying the 2-week window in which the transaction occurred.

    This function extracts one graph per timestep.  Only intra-timestep edges
    (both endpoints share the same Time step) are included: each timestep graph
    represents the induced subgraph of the induced subgraph for that window.
    Cross-timestep edges (transactions that flow between 2-week periods) are
    excluded because they connect structurally different time windows and make
    the per-timestep generative task ill-defined.

    Edge file accepted:
      * Full Elliptic edge list (txId1,txId2) — edges outside the 18,945-node
        induced set are filtered automatically.
      * Pre-filtered list for the 18,945-node induced subgraph — used as-is.

    Args:
        node_csv   : path to merged_nodes_final.csv
        edge_csv   : path to edge list CSV (txId1,txId2 or node_id1,node_id2)
        min_nodes  : drop timestep LCCs smaller than this threshold (default 50)
        cache_path : load from / save to this pkl path to skip re-parsing

    Returns:
        list of (int timestep, nx.Graph) sorted chronologically,
        nodes 0-indexed, no attributes.
    """
    if cache_path and os.path.exists(cache_path):
        return pkl.load(open(cache_path, 'rb'))

    txid_to_nodeid, node_to_ts = _load_elliptic_node_meta(node_csv)
    raw_edges = _load_elliptic_edges(edge_csv, txid_to_nodeid, node_to_ts)

    # Bucket edges by timestep (intra-ts only, deduplicated undirected)
    edges_by_ts = {}
    for s, d in raw_edges:
        ts_s = node_to_ts.get(s)
        ts_d = node_to_ts.get(d)
        if ts_s is None or ts_d is None or ts_s != ts_d:
            continue
        key = (min(s, d), max(s, d))
        edges_by_ts.setdefault(ts_s, set()).add(key)

    # Group nodes by timestep
    nodes_by_ts = {}
    for nid, ts in node_to_ts.items():
        nodes_by_ts.setdefault(ts, []).append(nid)

    result = []
    for ts in sorted(nodes_by_ts.keys()):
        G = nx.Graph()
        G.add_nodes_from(nodes_by_ts[ts])
        G.add_edges_from(edges_by_ts.get(ts, set()))

        comps = list(nx.connected_components(G))
        if not comps:
            continue
        lcc = G.subgraph(max(comps, key=len)).copy()
        lcc = nx.convert_node_labels_to_integers(lcc, first_label=0, ordering='sorted')
        if lcc.number_of_nodes() >= min_nodes:
            result.append((ts, lcc))

    if cache_path:
        pkl.dump(result, open(cache_path, 'wb'))
        print(f'[Elliptic] cached {len(result)} timestep graphs → {cache_path}')

    return result


def load_elliptic_temporal_graphs(edge_csv, resolution, min_nodes=20,
                                   cache_path=None):
    """Build multi-graph dataset from the Elliptic induced subgraph at a
    specified temporal resolution.

    The merged_edges_gaussian_timestamp.csv file carries three discrete
    temporal attributes per edge — month (1–12), weekday (0–6), hour (0–23) —
    derived from the original Bitcoin transaction timestamp.  This function
    groups the 21,660 edges by one of those attributes and produces one
    undirected LCC subgraph per group value, mirroring the elliptic_ts
    approach but at a different temporal granularity.

    Practical viability of each resolution (measured on 18,945-node induced
    subgraph; LCC sizes determine ERGM training quality):
      month   : 12 groups  |  LCC 9–120 nodes   — small; use min_nodes≥9
      weekday :  7 groups  |  LCC 73–146 nodes  — best; comparable to elliptic_ts
      hour    : 24 groups  |  LCC 3–42 nodes    — very small; near-trivial graphs

    Args:
        edge_csv   : path to merged_edges_gaussian_timestamp.csv
        resolution : 'month' | 'weekday' | 'hour'
        min_nodes  : drop LCCs smaller than this threshold (default 20)
        cache_path : load from / save to this pkl path

    Returns:
        list of (resolution_value, nx.Graph) sorted by resolution_value,
        nodes 0-indexed, no attributes.
    """
    if resolution not in ('month', 'weekday', 'hour'):
        raise ValueError(
            f"resolution must be 'month', 'weekday', or 'hour'; got '{resolution}'")

    if cache_path and os.path.exists(cache_path):
        return pkl.load(open(cache_path, 'rb'))

    import pandas as pd
    df = pd.read_csv(edge_csv)

    result = []
    for val in sorted(df[resolution].unique()):
        grp = df[df[resolution] == val]
        G = nx.Graph()
        G.add_edges_from(zip(grp['source'], grp['target']))
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        if G.number_of_nodes() == 0:
            continue
        comps = list(nx.connected_components(G))
        lcc = G.subgraph(max(comps, key=len)).copy()
        lcc = nx.convert_node_labels_to_integers(lcc, first_label=0,
                                                  ordering='sorted')
        if lcc.number_of_nodes() >= min_nodes:
            result.append((int(val), lcc))

    if cache_path:
        pkl.dump(result, open(cache_path, 'wb'))
        print(f'[Elliptic-{resolution}] cached {len(result)} graphs → {cache_path}')

    return result


def load_elliptic_full_graph(node_csv, edge_csv, cache_path=None):
    """Build the full induced-subgraph LCC including cross-timestep edges.

    Uses all 18,945 nodes from merged_nodes_final.csv and all edges within the
    induced subgraph (both intra- and cross-timestep).  LCC is extracted and
    nodes are relabelled 0..N-1.

    Memory note: N≈18,945 is equivalent to PubMed (N=19,717) in scale.
    Full candidate space N(N-1)/2≈179M pairs requires ~40 GB GPU for ERGM.
    Prefer --dataset elliptic_ts (per-timestep, N≈400) on ≤24 GB GPUs.
    Use --edge_fraction to subsample non-edges if a single-graph run is required
    on a smaller GPU.
    """
    if cache_path and os.path.exists(cache_path):
        return pkl.load(open(cache_path, 'rb'))

    txid_to_nodeid, node_to_ts = _load_elliptic_node_meta(node_csv)
    raw_edges = _load_elliptic_edges(edge_csv, txid_to_nodeid, node_to_ts)

    G = nx.Graph()
    G.add_nodes_from(node_to_ts.keys())
    undirected = {(min(s, d), max(s, d)) for s, d in raw_edges}
    G.add_edges_from(undirected)

    comps = list(nx.connected_components(G))
    lcc   = G.subgraph(max(comps, key=len)).copy()
    lcc   = nx.convert_node_labels_to_integers(lcc, first_label=0, ordering='sorted')

    if cache_path:
        pkl.dump(lcc, open(cache_path, 'wb'))
        print(f'[Elliptic] cached full graph N={lcc.number_of_nodes()} '
              f'M={lcc.number_of_edges()} → {cache_path}')

    return lcc


def _pyg_directed_to_nx_lcc(pyg_data, min_nodes, max_nodes):
    """Convert a directed PyG graph (no node features) to an undirected NetworkX LCC.

    Function call graphs in MalNetTiny are directed (caller → callee).
    Making them undirected captures co-occurrence structure used by the
    ERGM triangle feature (A and B share a common callee/caller → wedge/triangle).

    Self-loops and multi-edges are collapsed by nx.Graph automatically.
    Returns None if the LCC falls outside [min_nodes, max_nodes].
    """
    import torch
    G = nx.Graph()
    G.add_nodes_from(range(pyg_data.num_nodes))
    if pyg_data.edge_index.numel() > 0:
        ei = pyg_data.edge_index.t().tolist()
        G.add_edges_from((s, d) for s, d in ei if s != d)

    comps = list(nx.connected_components(G))
    if not comps:
        return None
    lcc = G.subgraph(max(comps, key=len)).copy()
    n   = lcc.number_of_nodes()
    if not (min_nodes <= n <= max_nodes):
        return None
    return nx.convert_node_labels_to_integers(lcc, first_label=0, ordering='sorted')


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='Ego')
    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=str_to_bool, default=True)

    parser.add_argument('--empty_graph_sampler', type=str, default='empirical', help='empirical | neural') 
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--augmented_features', type=str, nargs="*", default=[])
    parser.add_argument('--rwse_k', type=int, default=0,
                        help='RWSE walk steps k=1..K injected as node features in GraphDenoising '
                             '(0=disabled). Computed dynamically from noisy x_t each forward pass.')
    parser.add_argument('--malnet_root', type=str, default='graphs/malnet_tiny',
                        help='Root directory for MalNetTiny dataset storage. '
                             'First run downloads ~1.5 GB from malnet.cc.gatech.edu; '
                             'do this on a login node before submitting SLURM jobs.')
    parser.add_argument('--malnet_max_nodes', type=int, default=500,
                        help='Discard MalNetTiny graphs (after LCC) with more nodes than this. '
                             'N(N-1)/2 candidate pairs grow quadratically: N=500→125K, '
                             'N=1000→500K. Default 500 keeps memory under 4 GB per ERGM step.')
    parser.add_argument('--malnet_min_nodes', type=int, default=20,
                        help='Discard MalNetTiny graphs with fewer nodes than this (default 20).')
    parser.add_argument('--elliptic_edge_file', type=str, default=None,
                        help='Path to Elliptic edge list CSV (required for --dataset elliptic '
                             'or elliptic_ts). Standard format: two columns txId1,txId2 '
                             '(original Elliptic) or node_id1,node_id2 (pre-mapped).')
    parser.add_argument('--elliptic_min_nodes', type=int, default=50,
                        help='Drop per-timestep LCCs with fewer nodes than this '
                             '(elliptic_ts only). Default 50.')
    parser.add_argument('--zinc_root', type=str, default='graphs/zinc',
                        help='Root directory for ZINC dataset storage. '
                             'First run downloads automatically via PyG. '
                             'Use --dataset zinc for ZINC12k (10k train), '
                             'or --dataset zinc250k for the full 250k split.')
    parser.add_argument('--zinc_min_nodes', type=int, default=5,
                        help='Discard ZINC graphs (after LCC) with fewer nodes (default 5).')

def get_data_id(args):
    return '{}'.format(args.dataset)

def get_data(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'polblogs', 'Homo_sapiens',
                        'bitcoin_alpha', 'bitcoin_otc']:
        repeat = 1
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat = None
        nx_graph = pkl.load(open(f'graphs/{args.dataset}.pkl','rb'))
        pyg_graph = preprocess(nx_graph, degree=args.degree,
                               edge_fraction=getattr(args, 'edge_fraction', 1.0))
        max_degree = max([d for _, d in nx_graph.degree()])
        n = nx_graph.number_of_nodes()
        train_density = nx_graph.number_of_edges() / (n * (n - 1) / 2)
        train_set = NetworkDataset(pyg_graph, num_iter=args.num_iter * args.batch_size, transform=None)
        test_set = eval_set = NetworkDataset(pyg_graph, num_iter=100, transform=None)
        initial_graph_sampler = EmpiricalEmptyGraphGenerator(
            [train_set[0]], degree=args.degree,
            augment_features=args.augmented_features)
        eval_evaluator = test_evaluator = NetworkEvaluator(nx_graph)
        monitoring_statistics = ['nmae/assortativity','nmae/triangle_count', 'nmae/clustering_coefficient']
 
    elif args.dataset in ['community', 'Ego', 'ego']:
        repeat = 64 
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat = None
        dataset_name = 'Ego' if args.dataset.lower() == 'ego' else args.dataset
        nx_graphs = pkl.load(open(f"graphs/{dataset_name}.pkl", 'rb'))
        # Fixed-seed shuffle so train/eval/test split is identical across every
        # training run and every subsequent evaluate.py / ablation_steps.py call.
        # Uses a private Random instance so the global random state is untouched.
        random.Random(42).shuffle(nx_graphs)
        l = len(nx_graphs)
        train_nx_graphs = nx_graphs[:int(0.8*l)]
        eval_nx_graphs  = nx_graphs[int(0.8*l):int(0.9*l)]   # non-overlapping 10 %
        test_nx_graphs  = nx_graphs[int(0.9*l):]              # non-overlapping 10 % 

        train_pygraphs = []
        eval_pygraphs = []
        test_pygraphs = []

        max_degree = max([max([d for n, d in train_nx_graph.degree()]) for train_nx_graph in train_nx_graphs])
        train_density = np.mean([
            g.number_of_edges() / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
            for g in train_nx_graphs])
        for nx_graph in train_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            train_pygraphs.append(pyg_data)

        for nx_graph in eval_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            eval_pygraphs.append(pyg_data)

        for nx_graph in test_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            test_pygraphs.append(pyg_data)
            
        train_set = ConcatDataset([GraphDataset(train_pygraphs) for _ in range(repeat)])
        eval_set = GraphDataset(eval_pygraphs)
        test_set = GraphDataset(test_pygraphs)

        if args.empty_graph_sampler == 'empirical':
            initial_graph_sampler = EmpiricalEmptyGraphGenerator(train_pygraphs, degree=args.degree)
        elif args.empty_graph_sampler == 'neural':
            neural_attr_sampler = torch.load(f'graphs/{dataset_name}_degree_sampler.pt', map_location=args.device)
            initial_graph_sampler = NeuralEmptyGraphGenerator(train_pygraphs, neural_attr_sampler, degree=args.degree, device=args.device)

        eval_evaluator = GenericGraphEvaluator(eval_nx_graphs, device=args.device)
        test_evaluator = GenericGraphEvaluator(test_nx_graphs, device=args.device)

        # Metrics: Deg. Clus. Orb. FID RBF MMD  (as reported in graph-gen literature)
        # degree_mmd=Deg  clustering_mmd=Clus  orbits_mmd=Orb
        # mmd_linear=FID  mmd_rbf=RBF  nspdk_mmd=MMD
        monitoring_statistics = ['degree_mmd', 'clustering_mmd', 'orbits_mmd',
                                  'mmd_linear', 'mmd_rbf', 'nspdk_mmd']

    elif args.dataset == 'malnet_tiny':
        # ── MalNetTiny: 5,000 function call graphs, ≤5,000 nodes each ────────
        # Directed graphs (caller→callee) made undirected for the ERGM model.
        # Pre-defined train/val/test splits (from the MalNetTiny authors).
        # node_attr=0 (no node features in the raw data) → node KL=0.
        # Triangles exist from: shared callees (A→C, B→C, A→B), mutual recursion
        # (A→B→C→A), and library re-use patterns — more than Bitcoin DAGs.
        #
        # IMPORTANT: first run downloads ~1.5 GB from malnet.cc.gatech.edu.
        # Run: python -c "from torch_geometric.datasets import MalNetTiny;
        #                 MalNetTiny('graphs/malnet_tiny')"
        # on a login node WITH internet access BEFORE submitting SLURM jobs.
        from torch_geometric.datasets import MalNetTiny

        malnet_root = getattr(args, 'malnet_root', 'graphs/malnet_tiny')
        max_n = getattr(args, 'malnet_max_nodes', 500)
        min_n = getattr(args, 'malnet_min_nodes', 20)

        # Load official splits — avoids temporal/label leakage across sets
        train_raw = MalNetTiny(malnet_root, split='train')
        val_raw   = MalNetTiny(malnet_root, split='val')
        test_raw  = MalNetTiny(malnet_root, split='test')

        def _filter(pyg_split):
            graphs = []
            for d in pyg_split:
                g = _pyg_directed_to_nx_lcc(d, min_nodes=min_n, max_nodes=max_n)
                if g is not None:
                    graphs.append(g)
            return graphs

        train_nx = _filter(train_raw)
        eval_nx  = _filter(val_raw)
        test_nx  = _filter(test_raw)

        if not train_nx:
            raise RuntimeError(
                f'No MalNetTiny training graphs with [{min_n}, {max_n}] nodes '
                f'after LCC extraction. Adjust --malnet_min_nodes / --malnet_max_nodes.')

        # Diagnostics — printed once so the user can verify structural fitness
        all_nx  = train_nx + eval_nx + test_nx
        ns      = [g.number_of_nodes() for g in all_nx]
        ms      = [g.number_of_edges() for g in all_nx]
        rho_all = [m / (n*(n-1)/2) for n, m in zip(ns, ms) if n > 1]
        # Triangle count on a sample (full set can be slow for 5K graphs)
        sample  = train_nx[:min(200, len(train_nx))]
        tris    = [sum(nx.triangles(g).values()) // 3 for g in sample]
        print(f'[MalNetTiny] size filter [{min_n},{max_n}]: '
              f'train={len(train_nx)} val={len(eval_nx)} test={len(test_nx)} '
              f'(discarded {5000 - len(all_nx)} graphs)')
        print(f'  N: {min(ns)}-{max(ns)} (mean {np.mean(ns):.0f}) | '
              f'M: {min(ms)}-{max(ms)} (mean {np.mean(ms):.0f}) | '
              f'rho_mean={np.mean(rho_all):.4f} | '
              f'triangles (train sample): min={min(tris)} max={max(tris)} mean={np.mean(tris):.1f}')

        max_degree    = max(max(d for _, d in g.degree()) for g in train_nx)
        train_density = np.mean([
            g.number_of_edges() / (g.number_of_nodes() * (g.number_of_nodes()-1) / 2)
            for g in train_nx])
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat    = None

        train_pygraphs = [preprocess(g, degree=args.degree,
                                     augmented_features=args.augmented_features)
                          for g in train_nx]
        eval_pygraphs  = [preprocess(g, degree=args.degree) for g in eval_nx]
        test_pygraphs  = [preprocess(g, degree=args.degree) for g in test_nx]

        # repeat: scale dataset to ~2,000 samples minimum so each epoch has
        # at least 500 gradient steps at batch_size=4. Capped at 32 to avoid
        # excessive memory in the ConcatDataset.
        repeat    = max(1, min(32, 2000 // max(1, len(train_nx))))
        train_set = ConcatDataset([GraphDataset(train_pygraphs) for _ in range(repeat)])
        eval_set  = GraphDataset(eval_pygraphs)
        test_set  = GraphDataset(test_pygraphs)

        initial_graph_sampler = EmpiricalEmptyGraphGenerator(
            train_pygraphs, degree=args.degree,
            augment_features=args.augmented_features,
            sbm_n_clusters=getattr(args, 'sbm_n_clusters', None))

        eval_evaluator = GenericGraphEvaluator(eval_nx, device=args.device)
        test_evaluator = GenericGraphEvaluator(test_nx, device=args.device)
        monitoring_statistics = ['clustering_mmd', 'orbits_mmd', 'spectral_mmd',
                                 'degree_mmd', 'mmd_linear', 'mmd_rbf']

    elif args.dataset in ['elliptic_month', 'elliptic_weekday', 'elliptic_hour']:
        # ── Elliptic temporal-resolution multi-graph datasets ─────────────────
        # Source: graphs/merged_edges_gaussian_timestamp.csv
        # Edges are grouped by one of three discrete temporal attributes:
        #   elliptic_month   → month   (1–12): 12 graphs  LCC 9–120 nodes
        #   elliptic_weekday → weekday (0–6) :  7 graphs  LCC 73–146 nodes  ← best
        #   elliptic_hour    → hour    (0–23): 24 graphs  LCC 3–42 nodes
        #
        # These realise the paper's "multiple temporal resolutions" experiment:
        # the same 18,945-node induced subgraph is sliced at coarser (monthly)
        # and finer (hourly) granularities to study how time granularity affects
        # generation performance.  elliptic_weekday is the most viable resolution
        # for ERGM: LCC sizes (73–146) are comparable to small elliptic_ts graphs.
        # Monthly and hourly LCCs are small but structurally valid.
        #
        # Evaluation: GenericGraphEvaluator with MMD metrics (same as elliptic_ts).
        res_map = {
            'elliptic_month':   ('month',   20, 'graphs/elliptic_month_graphs.pkl'),
            'elliptic_weekday': ('weekday',  20, 'graphs/elliptic_weekday_graphs.pkl'),
            'elliptic_hour':    ('hour',      5, 'graphs/elliptic_hour_graphs.pkl'),
        }
        resolution, min_n, cache = res_map[args.dataset]

        edge_file = getattr(args, 'elliptic_edge_file',
                            'graphs/merged_edges_gaussian_timestamp.csv')
        temporal_graphs = load_elliptic_temporal_graphs(
            edge_file, resolution=resolution, min_nodes=min_n, cache_path=cache)

        if not temporal_graphs:
            raise RuntimeError(
                f'No Elliptic-{resolution} graphs with N≥{min_n} after LCC '
                f'extraction. Lower --elliptic_min_nodes or check edge file.')

        nx_graphs = [g for _, g in temporal_graphs]
        ns  = [g.number_of_nodes() for g in nx_graphs]
        ms  = [g.number_of_edges() for g in nx_graphs]
        rho = np.mean([m / (n*(n-1)/2) for n, m in zip(ns, ms) if n > 1])
        print(f'[Elliptic-{resolution}] {len(nx_graphs)} graphs | '
              f'N: {min(ns)}–{max(ns)} (mean {np.mean(ns):.0f}) | '
              f'M: {min(ms)}–{max(ms)} (mean {np.mean(ms):.0f}) | '
              f'rho_mean={rho:.4f}')

        # Temporal order: train on earlier values, eval/test on later
        # month: train 1–8, val 9–10, test 11–12
        # weekday: train 0–4 (Mon–Fri), val 5 (Sat), test 6 (Sun)
        # hour: train 0–17, val 18–20, test 21–23
        split_map = {
            'month':   (8, 10),    # train<= 8, val 9-10, test 11-12
            'weekday': (5, 6),     # train<= 5, val 6, test 6 (overlap)
            'hour':    (18, 21),   # train<=18, val 18-21, test 21+
        }
        vals = [v for v, _ in temporal_graphs]
        split1, split2 = split_map[resolution]
        train_nx = [g for v, g in temporal_graphs if v <  split1]
        eval_nx  = [g for v, g in temporal_graphs if split1 <= v < split2]
        test_nx  = [g for v, g in temporal_graphs if v >= split2]
        # Ensure non-empty splits
        if not train_nx: train_nx = nx_graphs[:-2] or [nx_graphs[0]]
        if not eval_nx:  eval_nx  = [nx_graphs[-2]] if len(nx_graphs) >= 2 else [nx_graphs[-1]]
        if not test_nx:  test_nx  = [nx_graphs[-1]]

        num_node_classes  = None
        num_edge_classes  = 2
        num_node_feat     = None
        max_degree        = max(max(d for _, d in g.degree()) for g in nx_graphs)
        train_density     = np.mean([
            g.number_of_edges() / (g.number_of_nodes() * (g.number_of_nodes()-1) / 2)
            for g in train_nx])

        train_pygraphs = [preprocess(g, degree=args.degree,
                                     augmented_features=args.augmented_features)
                          for g in train_nx]
        eval_pygraphs  = [preprocess(g, degree=args.degree) for g in eval_nx]
        test_pygraphs  = [preprocess(g, degree=args.degree) for g in test_nx]

        repeat    = max(1, min(32, 500 // max(1, len(train_nx))))
        train_set = ConcatDataset([GraphDataset(train_pygraphs) for _ in range(repeat)])
        eval_set  = GraphDataset(eval_pygraphs)
        test_set  = GraphDataset(test_pygraphs)

        initial_graph_sampler = EmpiricalEmptyGraphGenerator(
            train_pygraphs, degree=args.degree,
            augment_features=args.augmented_features,
            sbm_n_clusters=getattr(args, 'sbm_n_clusters', None))

        # Population-based metrics need a real reference population, not one
        # temporal slice. PRDC (density/coverage) estimates a k-NN radius
        # among the reference graphs themselves (default nearest_k=5, so it
        # needs >=6 reference graphs just to not raise "kth out of bounds");
        # eval_nx/test_nx alone can be as small as 1 graph (elliptic_weekday:
        # 5 train / 1 eval / 1 test, only 7 graphs total), which makes
        # density/coverage mathematically undefined and turns every MMD here
        # into a single-sample point estimate. Pool the full temporal graph
        # set as the evaluator's reference population instead, while leaving
        # eval_nx/test_nx untouched for what the model actually trains/is
        # monitored against above. This trades strict train/eval/test
        # separation in the *evaluator* (some reference graphs may have been
        # seen in training) for a metric that is at least computable and
        # lower-variance — with only 7-24 graphs total per resolution here,
        # a truly held-out population large enough for these metrics doesn't
        # exist either way.
        eval_evaluator = GenericGraphEvaluator(nx_graphs, device=args.device)
        test_evaluator = GenericGraphEvaluator(nx_graphs, device=args.device)
        monitoring_statistics = ['degree_mmd', 'clustering_mmd', 'orbits_mmd',
                                  'mmd_linear', 'mmd_rbf', 'nspdk_mmd']

    elif args.dataset in ['elliptic_ts', 'elliptic']:
        # ── Elliptic Bitcoin Transaction Network (induced subgraph) ──────────
        # merged_nodes_final.csv is a pre-constructed induced subgraph of the
        # full 203K-node Elliptic dataset: 18,945 nodes selected for GPU
        # feasibility, preserving graph structure (degrees, triangles, etc.).
        # node_attr=0 for all → node KL=0, model focuses on structural generation.
        # class 1=illicit(518), 2=licit(2388), 3=unknown(16039) — stored in CSV
        # but not used as node attributes (all nodes treated identically).
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat    = None

        edge_file = getattr(args, 'elliptic_edge_file', None)
        if edge_file is None:
            raise ValueError(
                '--elliptic_edge_file is required for the elliptic dataset.\n'
                'Download elliptic_txs_edgelist.csv from the Elliptic dataset '
                '(Kaggle / IEEE DataPort) and pass its path with --elliptic_edge_file.')

        node_csv = 'graphs/merged_nodes_final.csv'
        min_n    = getattr(args, 'elliptic_min_nodes', 50)

        if args.dataset == 'elliptic_ts':
            # ── Multi-graph mode: one graph per 2-week timestep ───────────────
            # Temporal split (chronological): 70% train, 15% val, 15% test.
            # This mirrors the standard Elliptic evaluation protocol and prevents
            # future-graph leakage into training.
            cache = 'graphs/elliptic_ts_graphs.pkl'
            ts_graphs = load_elliptic_ts_graphs(node_csv, edge_file,
                                                min_nodes=min_n, cache_path=cache)
            if not ts_graphs:
                raise RuntimeError(
                    'No valid Elliptic timestep graphs found after LCC + min_nodes '
                    f'filter (min_nodes={min_n}). Check --elliptic_edge_file path.')

            nx_graphs = [g for _, g in ts_graphs]
            n_total   = len(nx_graphs)
            n_train   = max(1, int(0.70 * n_total))
            n_val     = max(1, int(0.15 * n_total))
            # test gets the remainder (at least 1)
            n_val     = min(n_val, n_total - n_train - 1)

            train_nx = nx_graphs[:n_train]
            eval_nx  = nx_graphs[n_train:n_train + n_val]
            test_nx  = nx_graphs[n_train + n_val:]
            if not eval_nx:
                eval_nx = [nx_graphs[-1]]
            if not test_nx:
                test_nx = [nx_graphs[-1]]

            ns  = [g.number_of_nodes() for g in nx_graphs]
            ms  = [g.number_of_edges() for g in nx_graphs]
            rho = np.mean([m / (n*(n-1)/2) for n, m in zip(ns, ms) if n > 1])
            tris = [sum(nx.triangles(g).values()) // 3 for g in nx_graphs]
            print(f'[Elliptic-ts] {n_total} timestep graphs | '
                  f'N: {min(ns)}–{max(ns)} (mean {np.mean(ns):.0f}) | '
                  f'M: {min(ms)}–{max(ms)} (mean {np.mean(ms):.0f}) | '
                  f'rho_mean={rho:.4f} | triangles: {min(tris)}–{max(tris)}'
                  f' → train={len(train_nx)} val={len(eval_nx)} test={len(test_nx)}')

            max_degree    = max(max(d for _, d in g.degree())
                                for g in nx_graphs if g.number_of_edges() > 0)
            train_density = np.mean([
                g.number_of_edges() / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
                for g in train_nx])

            train_pygraphs = [preprocess(g, degree=args.degree,
                                         augmented_features=args.augmented_features)
                              for g in train_nx]
            eval_pygraphs  = [preprocess(g, degree=args.degree) for g in eval_nx]
            test_pygraphs  = [preprocess(g, degree=args.degree) for g in test_nx]

            # repeat=16 gives enough training examples; small graphs are fast
            repeat    = 16
            train_set = ConcatDataset([GraphDataset(train_pygraphs) for _ in range(repeat)])
            eval_set  = GraphDataset(eval_pygraphs)
            test_set  = GraphDataset(test_pygraphs)

            initial_graph_sampler = EmpiricalEmptyGraphGenerator(
                train_pygraphs, degree=args.degree,
                augment_features=args.augmented_features)

            eval_evaluator = GenericGraphEvaluator(eval_nx,  device=args.device)
            test_evaluator = GenericGraphEvaluator(test_nx,  device=args.device)
            monitoring_statistics = ['clustering_mmd', 'orbits_mmd', 'spectral_mmd',
                                     'degree_mmd', 'mmd_linear', 'mmd_rbf']

        else:  # 'elliptic' — full induced subgraph (all timesteps, all edges)
            # N≈18,945 is equivalent in scale to PubMed (N=19,717).
            # N(N-1)/2≈179M pairs → requires ~40 GB GPU for ERGM without subsampling.
            # Use --edge_fraction 0.01 (~2M candidate pairs/step) for ≤24 GB GPUs.
            cache = 'graphs/elliptic_full_graph.pkl'
            nx_graph = load_elliptic_full_graph(node_csv, edge_file, cache_path=cache)

            n             = nx_graph.number_of_nodes()
            max_degree    = max(d for _, d in nx_graph.degree())
            train_density = nx_graph.number_of_edges() / (n * (n - 1) / 2)

            print(f'[Elliptic-full] N={n} M={nx_graph.number_of_edges()} '
                  f'rho={train_density:.6f}')

            pyg_graph = preprocess(nx_graph, degree=args.degree,
                                   edge_fraction=getattr(args, 'edge_fraction', 1.0))
            repeat    = 1
            train_set = NetworkDataset(pyg_graph,
                                       num_iter=args.num_iter * args.batch_size)
            eval_set  = test_set = NetworkDataset(pyg_graph, num_iter=100)

            initial_graph_sampler = EmpiricalEmptyGraphGenerator(
                [train_set[0]], degree=args.degree,
                augment_features=args.augmented_features)

            eval_evaluator = test_evaluator = NetworkEvaluator(nx_graph)
            monitoring_statistics = ['nmae/assortativity', 'nmae/triangle_count',
                                     'nmae/clustering_coefficient']

    elif args.dataset == 'elliptic_induced':
        # ── Elliptic induced subgraph: full disconnected single-graph mode ──────
        # Node authority: graphs/merged_nodes_final.csv (18,945 pre-mapped node_ids).
        # Edge source:    graphs/merged_edges_gaussian_timestamp.csv (source/target
        #                 are the same pre-mapped node_ids — all 21,660 edges are
        #                 intra-timestep; filtering the full Kaggle edge list by
        #                 these 18,945 nodes yields the identical edge set).
        # No Kaggle download required.
        #
        # Verified graph properties (LCC + full graph):
        #   N=18,945  M=21,660  components=233  d_max=163  PLE≈2.5
        #   LCC: N=725  CPL=242.0  avg_d=1.997  d_max=2  (path/chain structure)
        #   Full graph: triangles=2,725  wedges=95,981  Q=117,641
        #
        # Memory: preprocess() materialises a dense N×N adj matrix (~3 GB RAM).
        # Requires 64 GB CPU RAM.  --edge_fraction 0.02 subsamples non-edges for
        # the VB loss (179M pairs → 3.6M), fitting a 40 GB A100.
        # ERGM loss always uses Q=117,641 pairs (unaffected by edge_fraction).
        #
        # DACA: d_max/d_mean=71 — use --daca_gamma 0.0 (uniform noise).
        # With gamma=1.0, hub_beta=107 and hub edges vanish at t=1.
        import pandas as pd
        node_csv  = 'graphs/merged_nodes_final.csv'
        edge_file = getattr(args, 'elliptic_edge_file',
                            'graphs/merged_edges_gaussian_timestamp.csv')
        cache = 'graphs/elliptic_induced.pkl'
        if os.path.exists(cache):
            nx_graph = pkl.load(open(cache, 'rb'))
        else:
            # Load all 18,945 authoritative node IDs from the node CSV
            all_node_ids = []
            with open(node_csv) as _f:
                for _row in _csv.DictReader(_f):
                    all_node_ids.append(int(_row['node_id']))

            # Build graph: add ALL nodes first (guards against isolated nodes),
            # then add edges from the pre-mapped edge CSV
            df = pd.read_csv(edge_file)
            G = nx.Graph()
            G.add_nodes_from(all_node_ids)
            G.add_edges_from(zip(df['source'], df['target']))
            G.remove_edges_from(list(nx.selfloop_edges(G)))
            # Relabel to compact 0..N-1 sorted by original node_id
            nx_graph = nx.convert_node_labels_to_integers(G, first_label=0,
                                                          ordering='sorted')
            pkl.dump(nx_graph, open(cache, 'wb'))
            print(f'[Elliptic-induced] cached N={nx_graph.number_of_nodes()} '
                  f'M={nx_graph.number_of_edges()} → {cache}')

        n             = nx_graph.number_of_nodes()
        max_degree    = max(d for _, d in nx_graph.degree())
        train_density = nx_graph.number_of_edges() / (n * (n - 1) / 2)
        print(f'[Elliptic-induced] N={n}  M={nx_graph.number_of_edges()}  '
              f'LCC=725(CPL≈242)  d_max={max_degree}  PLE≈2.5  '
              f'rho={train_density:.2e}  Q=117641')

        repeat          = 1
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat    = None

        pyg_graph = preprocess(nx_graph, degree=args.degree,
                               edge_fraction=getattr(args, 'edge_fraction', 1.0))
        train_set = NetworkDataset(pyg_graph,
                                   num_iter=args.num_iter * args.batch_size)
        eval_set = test_set = NetworkDataset(pyg_graph, num_iter=100)
        initial_graph_sampler = EmpiricalEmptyGraphGenerator(
            [train_set[0]], degree=args.degree,
            augment_features=args.augmented_features)
        eval_evaluator = test_evaluator = NetworkEvaluator(nx_graph)
        monitoring_statistics = ['nmae/assortativity', 'nmae/triangle_count',
                                 'nmae/clustering_coefficient']

    elif args.dataset in ['zinc', 'zinc12k', 'zinc250k']:
        # ── ZINC: molecule track — typed atoms, binary edges ─────────────────
        #
        # Node features: atom type ∈ {0..27} (28 types in ZINC12k/250k).
        #   num_node_classes=28, final_prob_node=empirical atom marginal.
        #   The 28-class node diffusion generates atom types.
        #
        # Edge diffusion: BINARY presence/absence (num_edge_classes=2).
        #   The absorbing ERGM diffusion is designed for binary edge graphs.
        #   Bond types (single/double/triple) are stored in edge_attr_bond for
        #   visualization ONLY — they are NOT part of the generation process.
        #   Extending the ERGM to typed edges requires a full CE-loss rewrite.
        #
        # args.final_prob_node: set here so train.py's guard
        #   (final_prob_node is None → collapse to binary/no-node mode) does not
        #   trigger.  With final_prob_node set, has_node_feature=True and the
        #   model outputs 28-class node logits alongside 2-class edge logits.
        #
        # ZINC12k  (--dataset zinc or zinc12k): 10k train / 1k val / 1k test.
        # ZINC250k (--dataset zinc250k): 220k train / 24k val / 5k test.
        from torch_geometric.datasets import ZINC

        def _zinc_to_pyg(raw, min_nodes):
            """Convert raw ZINC PyG item: atom types + LCC extraction.

            Returns a pyg.data.Data with:
              node_attr      : (N,) atom types 0-27  (used in 28-class node diffusion)
              edge_index     : (2, |E|) bidirectional pairs for the LCC
              edge_attr_bond : (|E|,) bond types 1-indexed (visualization only)
              num_nodes      : int
            edge_attr is NOT set so preprocess creates binary full_edge_attr.
            Returns None if the LCC is smaller than min_nodes.
            """
            node_attr  = raw.x.view(-1).long()
            bond_types = raw.edge_attr.view(-1).long() + 1  # 1-indexed for display

            # Build plain topology graph for LCC extraction
            g_plain = nx.Graph()
            g_plain.add_nodes_from(range(raw.num_nodes))
            ei = raw.edge_index
            for k in range(ei.shape[1]):
                u, v = ei[0, k].item(), ei[1, k].item()
                if u < v:
                    g_plain.add_edge(u, v)

            if not nx.is_connected(g_plain):
                lcc_nodes = sorted(max(nx.connected_components(g_plain), key=len))
            else:
                lcc_nodes = list(range(raw.num_nodes))

            if len(lcc_nodes) < min_nodes:
                return None

            if len(lcc_nodes) < raw.num_nodes:
                node_set = set(lcc_nodes)
                node_map = {old: new for new, old in enumerate(lcc_nodes)}
                keep_mask = torch.tensor([
                    ei[0, k].item() in node_set and ei[1, k].item() in node_set
                    for k in range(ei.shape[1])], dtype=torch.bool)
                ei_k  = ei[:, keep_mask]
                new_ei    = torch.stack([
                    torch.tensor([node_map[u.item()] for u in ei_k[0]]),
                    torch.tensor([node_map[v.item()] for v in ei_k[1]])])
                new_bond  = bond_types[keep_mask]
                new_na    = node_attr[torch.tensor(lcc_nodes)]
                num_nodes = len(lcc_nodes)
            else:
                new_ei    = ei.clone()
                new_bond  = bond_types
                new_na    = node_attr
                num_nodes = raw.num_nodes

            out = pyg.data.Data()
            out.edge_index     = new_ei
            out.edge_attr_bond = new_bond  # bond types — visualization only
            out.node_attr      = new_na    # atom types — used in 28-class node diffusion
            out.num_nodes      = num_nodes
            # edge_attr intentionally NOT set → preprocess falls back to binary full_edge_attr
            return out

        def _pyg_to_plain_nx(d):
            """Topology-only nx.Graph from molecule PyG data (for structural MMD)."""
            g = nx.Graph()
            g.add_nodes_from(range(d.num_nodes))
            ei = d.edge_index
            for k in range(ei.shape[1]):
                u, v = ei[0, k].item(), ei[1, k].item()
                if u < v:
                    g.add_edge(u, v)
            return g

        def _pyg_to_typed_nx(d):
            """Typed nx.Graph: topology + integer 'node_attr' for WL-hash metrics."""
            g = nx.Graph()
            g.add_nodes_from(range(d.num_nodes))
            na = d.node_attr.tolist() if hasattr(d, 'node_attr') else [0] * d.num_nodes
            for v, at in enumerate(na):
                g.nodes[v]['node_attr'] = int(at)
            ei = d.edge_index
            for k in range(ei.shape[1]):
                u, v = ei[0, k].item(), ei[1, k].item()
                if u < v:
                    g.add_edge(u, v)
            return g

        subset    = (args.dataset != 'zinc250k')
        zinc_root = getattr(args, 'zinc_root', 'graphs/zinc')
        min_n     = getattr(args, 'zinc_min_nodes', 5)

        train_raw = ZINC(zinc_root, subset=subset, split='train')
        val_raw   = ZINC(zinc_root, subset=subset, split='val')
        test_raw  = ZINC(zinc_root, subset=subset, split='test')

        train_pygs = [_zinc_to_pyg(d, min_n) for d in train_raw]
        train_pygs = [d for d in train_pygs if d is not None]
        eval_pygs  = [_zinc_to_pyg(d, min_n) for d in val_raw]
        eval_pygs  = [d for d in eval_pygs  if d is not None]
        test_pygs  = [_zinc_to_pyg(d, min_n) for d in test_raw]
        test_pygs  = [d for d in test_pygs  if d is not None]

        # Atom type marginal — MUST be set before train.py's final_prob_node guard
        all_node_types   = torch.cat([d.node_attr for d in train_pygs])
        num_node_classes = int(all_node_types.max().item()) + 1   # typically 28
        type_counts      = torch.bincount(all_node_types, minlength=num_node_classes)
        args.final_prob_node = (type_counts.float() / type_counts.sum()).tolist()

        # Edges remain binary for ERGM absorbing diffusion
        num_edge_classes = 2

        ns      = [d.num_nodes for d in train_pygs]
        ms      = [d.edge_index.shape[1] // 2 for d in train_pygs]  # //2: bidirectional
        rho_all = [m / (n*(n-1)/2) for n, m in zip(ns, ms) if n > 1]
        print(f'[ZINC{"250k" if not subset else "12k"}] molecule track  '
              f'train={len(train_pygs)} val={len(eval_pygs)} test={len(test_pygs)}')
        print(f'  N: {min(ns)}-{max(ns)} (mean {np.mean(ns):.1f}) | '
              f'M mean={np.mean(ms):.1f} | rho_mean={np.mean(rho_all):.4f}')
        print(f'  num_node_classes={num_node_classes} (atom types)  '
              f'num_edge_classes={num_edge_classes} (binary, bond-type in edge_attr_bond)')

        num_node_feat = None
        max_degree    = max(
            d.edge_index[0].bincount(minlength=d.num_nodes).max().item()
            for d in train_pygs)
        train_density = np.mean(rho_all)

        train_pygraphs = [preprocess(d, degree=args.degree,
                                     augmented_features=args.augmented_features)
                          for d in train_pygs]
        eval_pygraphs  = [preprocess(d, degree=args.degree) for d in eval_pygs]
        test_pygraphs  = [preprocess(d, degree=args.degree) for d in test_pygs]

        repeat    = 1
        train_set = GraphDataset(train_pygraphs)
        eval_set  = GraphDataset(eval_pygraphs)
        test_set  = GraphDataset(test_pygraphs)

        initial_graph_sampler = EmpiricalEmptyGraphGenerator(
            train_pygraphs, degree=args.degree,
            augment_features=args.augmented_features,
            sbm_n_clusters=getattr(args, 'sbm_n_clusters', None))

        # Topology-only reference graphs for structural MMD
        eval_nx = [_pyg_to_plain_nx(d) for d in eval_pygs]
        test_nx = [_pyg_to_plain_nx(d) for d in test_pygs]

        eval_evaluator = GenericGraphEvaluator(eval_nx, device=args.device)
        test_evaluator = GenericGraphEvaluator(test_nx, device=args.device)
        monitoring_statistics = ['degree_mmd', 'clustering_mmd', 'orbits_mmd',
                                 'mmd_linear', 'mmd_rbf', 'nspdk_mmd']

        # Molecular metrics precomputation (validity / uniqueness / novelty).
        # Both evaluators share the same training-set references.
        # Done here so compute runs once, not per epoch.
        from eval_utils.molecular_metrics import build_train_hashes, build_max_valence_by_type
        print('[ZINC] Precomputing molecular metric references '
              f'(WL-hashes for {len(train_pygs)} training molecules) ...')
        _train_typed_nx = [_pyg_to_typed_nx(d) for d in train_pygs]
        _mol_train_hashes  = build_train_hashes(_train_typed_nx)
        _mol_max_valence   = build_max_valence_by_type(_train_typed_nx)
        print(f'  train_hashes={len(_mol_train_hashes)} unique  '
              f'max_valence_by_type={dict(sorted(_mol_max_valence.items()))}')
        for _ev in (eval_evaluator, test_evaluator):
            _ev.mol_train_hashes = _mol_train_hashes
            _ev.mol_max_valence  = _mol_max_valence
        del _train_typed_nx  # free memory; hashes are the only reference we keep

    else:
        raise NotImplementedError
    augmented_feature_dict = {k:FEATURE_EXTRACTOR[k]['data_spec'] for k in args.augmented_features}

    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size*repeat, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=partial(collate_fn))
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics, train_density
 
