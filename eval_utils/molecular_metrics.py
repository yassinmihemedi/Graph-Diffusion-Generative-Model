"""
eval_utils/molecular_metrics.py — Structural generation metrics for ZINC.

The model generates atom types (28-class node diffusion) and binary edge
topology, but NOT bond types.  Consequently:

  validity   = STRUCTURAL, not chemical.
               Necessary condition: connected + every atom's degree ≤
               max_valence[atom_type] (empirically derived from training data).
               A graph that passes may still be chemically invalid if the
               required bond orders cannot be assigned.  True chemical validity
               requires extending the model to generate bond types and running
               RDKit sanitization.

  uniqueness = |unique WL-hashes| / |total generated|
               WL hash uses atom type + topology → detects graph isomorphism
               up to type-labelled automorphism.  Two molecules with the same
               topology and atom types but different bond orders are considered
               identical (bond types are not generated).

  novelty    = |unique generated ∩ ¬train| / |unique generated|
               Fraction of unique generated structures not in the training set.
               Training hashes are precomputed at data-load time.

Usage in experiment.py:
    from eval_utils.molecular_metrics import zinc_metrics
    mol_met = zinc_metrics(typed_graphs, evaluator.mol_train_hashes,
                           evaluator.mol_max_valence)
    eval_dict.update(mol_met)

Dependencies: networkx (stdlib); no RDKit required.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, FrozenSet, List


# ── WL hash helpers ────────────────────────────────────────────────────────────

def _to_int_attr(v) -> int:
    """Coerce node_attr value (tensor, float, int) to Python int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def wl_hash_typed(g: nx.Graph) -> str:
    """
    WL graph hash with integer 'node_attr' as the node label.

    Operates on a COPY so the caller's graph is not mutated.
    """
    g2 = nx.Graph()
    g2.add_nodes_from(range(g.number_of_nodes()))
    for v in g.nodes:
        g2.nodes[v]['_l'] = str(_to_int_attr(g.nodes[v].get('node_attr', 0)))
    for u, v in g.edges:
        g2.add_edge(u, v)
    return nx.weisfeiler_lehman_graph_hash(g2, node_attr='_l')


# ── Training-set precomputation ────────────────────────────────────────────────

def build_train_hashes(train_typed_nx: List[nx.Graph]) -> FrozenSet[str]:
    """
    Compute WL-hashes for all training typed nx.Graphs.

    Call once at data-load time; store result on the evaluator.
    O(|train| * N) — ~1-2 s for ZINC12k (10k mols, N≈29).
    """
    return frozenset(wl_hash_typed(g) for g in train_typed_nx)


def build_max_valence_by_type(train_typed_nx: List[nx.Graph]) -> Dict[int, int]:
    """
    Empirical max degree per atom type across all training graphs.

    These degrees are computed from the actual (binary topology) training
    graphs — where an edge represents *any* bond regardless of order.
    They serve as the per-type degree upper bound for the validity check.

    For ZINC12k, empirical values (training set):
      C(0):4  O(1):2  N(2):3  F(3):1  S(4→):4  Cl:1  Br:1  ...
    (exact values derived at runtime from training data)
    """
    max_val: Dict[int, int] = {}
    for g in train_typed_nx:
        for v in g.nodes:
            at = _to_int_attr(g.nodes[v].get('node_attr', 0))
            dg = g.degree(v)
            if dg > max_val.get(at, 0):
                max_val[at] = dg
    return max_val


# ── Validity check ─────────────────────────────────────────────────────────────

def structural_valid(g: nx.Graph, max_valence: Dict[int, int]) -> bool:
    """
    Structural validity for a typed generated molecule.

    Conditions (all must hold):
      1. Non-empty: at least 1 node and 1 edge.
      2. Connected: single connected component.
      3. Degree constraint: for every node v,
           degree(v) ≤ max_valence[atom_type(v)]
         where max_valence is empirically derived from training graphs.

    Condition 3 is NECESSARY but not SUFFICIENT for chemical validity.
    Example where it fails: F with degree 2 (F is monovalent → caught).
    Example where it passes but is chemically invalid: a topology that
    satisfies degree constraints with single bonds but cannot accommodate
    required double bonds without exceeding the valence of another atom.
    """
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return False
    if not nx.is_connected(g):
        return False
    for v in g.nodes:
        at  = _to_int_attr(g.nodes[v].get('node_attr', 0))
        dg  = g.degree(v)
        cap = max_valence.get(at, 4)  # fallback 4 (carbon-like unknown type)
        if dg > cap:
            return False
    return True


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def zinc_metrics(
    typed_graphs: List[nx.Graph],
    train_hashes: FrozenSet[str],
    max_valence: Dict[int, int],
) -> dict:
    """
    Compute validity, uniqueness, and novelty for a batch of generated molecules.

    Args:
        typed_graphs  : generated nx.Graphs with integer 'node_attr' on nodes.
        train_hashes  : frozenset of WL-hashes for ALL training molecules
                        (precomputed by build_train_hashes at data-load time).
        max_valence   : {atom_type: max_degree} from build_max_valence_by_type.

    Returns:
        dict with keys mol/validity, mol/uniqueness, mol/novelty (floats ∈ [0,1]).
        NaN values when the denominator is 0 (e.g., uniqueness=0 → novelty=NaN).

    Definitions:
        validity   = n_valid / n_total
        uniqueness = |unique WL-hashes| / n_total
        novelty    = |unique generated ∩ ¬train| / |unique generated|
                   (NaN when no unique generated molecules)

    Note: uniqueness and novelty denominators differ.  Uniqueness penalises
    duplicates against the TOTAL generated count; novelty is conditioned on
    uniqueness (deduplication happens first, then novel fraction is computed).
    This matches the GDSS / GRAN / DiGress convention.
    """
    nan = float('nan')
    if not typed_graphs:
        return {'mol/validity': nan, 'mol/uniqueness': nan, 'mol/novelty': nan}

    n_total = len(typed_graphs)

    n_valid = sum(1 for g in typed_graphs if structural_valid(g, max_valence))

    all_hashes  = [wl_hash_typed(g) for g in typed_graphs]
    unique_set  = set(all_hashes)
    n_unique    = len(unique_set)
    n_novel     = sum(1 for h in unique_set if h not in train_hashes)

    validity   = n_valid / n_total
    uniqueness = n_unique / n_total
    novelty    = (n_novel / n_unique) if n_unique > 0 else nan

    return {
        'mol/validity':   validity,
        'mol/uniqueness': uniqueness,
        'mol/novelty':    novelty,
    }
