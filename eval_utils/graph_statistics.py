import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw


def max_degree(A):
    try:
        degrees = A.sum(axis=-1)
        return np.max(degrees)
    except:
        return np.nan


def min_degree(A):
    try:
        degrees = A.sum(axis=-1)
        return np.min(degrees)
    except:
        return np.nan


def average_degree(A):
    try:
        degrees = A.sum(axis=-1)
        return np.mean(degrees)
    except:
        return np.nan


def LCC(A):
    try:
        G = nx.from_scipy_sparse_array(A)
        return max(len(c) for c in nx.connected_components(G))
    except:
        return np.nan


def wedge_count(A):
    try:
        degrees = np.array(A.sum(axis=-1))
        return 0.5 * np.dot(degrees.T, degrees - 1).reshape([])
    except:
        return np.nan


def claw_count(A):
    try:
        degrees = np.array(A.sum(axis=-1))
        return 1 / 6 * np.sum(degrees * (degrees - 1) * (degrees - 2))
    except:
        return np.nan


def triangle_count(A):
    try:
        A_graph = nx.from_scipy_sparse_array(A)
        triangles = nx.triangles(A_graph)
        t = np.sum(list(triangles.values())) / 3
        return int(t)
    except:
        return np.nan


def square_count(A, max_nodes_exact=5000):
    try:
        if A.shape[0] > max_nodes_exact:
            # A² is dense for large graphs → skip to avoid OOM
            return np.nan
        A_squared = A @ A
        common_neighbors = sp.triu(A_squared, k=1).tocsr()
        num_common_neighbors = common_neighbors.data
        return np.dot(num_common_neighbors, num_common_neighbors - 1) / 4
    except:
        return np.nan


def power_law_alpha(A):
    try:
        degrees = np.array(A.sum(axis=-1)).flatten()
        valid = degrees[degrees >= 1]
        if len(valid) < 2 or np.all(valid == valid[0]):
            return np.nan
        alpha = powerlaw.Fit(valid, xmin=1, verbose=False).power_law.alpha
        return alpha if np.isfinite(alpha) else np.nan
    except:
        return np.nan


def gini(A):
    try:
        N = A.shape[0]
        degrees_sorted = np.sort(np.array(A.sum(axis=-1)).flatten())
        total = np.sum(degrees_sorted)
        if total == 0:
            return np.nan
        return (
            2 * np.dot(degrees_sorted, np.arange(1, N + 1)) / (N * total)
            - (N + 1) / N
        )
    except:
        return np.nan


def edge_distribution_entropy(A):
    try:
        N = A.shape[0]
        if N < 2:
            return np.nan
        degrees = np.array(A.sum(axis=-1)).flatten()
        total = degrees.sum()
        if total == 0:
            return np.nan
        p = degrees / total
        # Treat isolated nodes (p=0) as 0 contribution: 0*log(0) = 0
        mask = p > 0
        return -np.dot(np.log(p[mask]), p[mask]) / np.log(N)
    except:
        return np.nan


def assortativity(A):
    try:
        G = nx.from_scipy_sparse_array(A)
        return nx.degree_assortativity_coefficient(G)
    except:
        return np.nan


def clustering_coefficient(A):
    try:
        t = triangle_count(A)
        w = wedge_count(A)
        if np.isnan(t) or np.isnan(w) or w == 0:
            return np.nan
        return 3 * t / w
    except:
        return np.nan


def cpl(A, max_nodes_exact=2000, sample_sources=500, seed=42):
    """Characteristic path length.

    For N <= max_nodes_exact: exact O(N²) APSP via scipy.
    For N > max_nodes_exact: unbiased estimate via BFS from `sample_sources`
        random source nodes — O(k × m) instead of O(N × m), no N×N allocation.
    """
    try:
        N = A.shape[0]
        G = nx.from_scipy_sparse_array(A)
        if N <= max_nodes_exact:
            P = sp.csgraph.shortest_path(A)
            mask = (1 - np.isinf(P)) * (1 - np.eye(N))
            reachable = P[mask.astype(bool)]
            if len(reachable) == 0:
                return np.nan
            return reachable.mean()
        else:
            rng = np.random.default_rng(seed)
            sources = rng.choice(N, size=min(sample_sources, N), replace=False)
            total, count = 0.0, 0
            for s in sources:
                lengths = nx.single_source_shortest_path_length(G, s)
                for t, d in lengths.items():
                    if t != s:
                        total += d
                        count += 1
            return total / count if count > 0 else np.nan
    except:
        return np.nan


def compute_graph_statistics(A):
    statistics = {
        "d_max":                max_degree(A),
        "d_min":                min_degree(A),
        "d":                    average_degree(A),
        "LCC":                  LCC(A),
        "wedge_count":          wedge_count(A),
        "claw_count":           claw_count(A),
        "triangle_count":       triangle_count(A),
        "square_count":         square_count(A),
        "power_law_exp":        power_law_alpha(A),
        "gini":                 gini(A),
        "rel_edge_distr_entropy": edge_distribution_entropy(A),
        "assortativity":        assortativity(A),
        "clustering_coefficient": clustering_coefficient(A),
        "cpl":                  cpl(A),
    }
    return statistics


def edge_overlap(A, B):
    try:
        return A.multiply(B).sum() / 2
    except:
        return np.nan
