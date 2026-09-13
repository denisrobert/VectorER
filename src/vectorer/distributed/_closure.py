"""Distributed exact connected-components closure (R-Swoosh / D-Swoosh).

The Swoosh closure over the above-``tau`` edges is the exact transitive
closure: :func:`distributed_closure` on one node, :func:`streaming_distributed_closure`
over edge chunks (bounded memory), and :func:`distributed_closure_reduce`
across machines (per-worker union-find + a shared-node merge into min-position
ids).  All three are bit-for-bit identical to the single-process closure.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, Sequence

from ..clustering import Cluster, ClusterAssignment, ScoredPair, _DisjointSet, _build
from ._core import hash_pair


def distributed_closure(
    edges: Sequence[ScoredPair],
    n: int,
    n_workers: int = 1,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Exact connected components over ScoredPair edges (transitive closure).

    Equivalent to the single-process ``SwooshClusterer.cluster`` on the same
    above-tau edges: union-find over all edges, deterministic min-position ids.
    """
    del n_workers  # the union-find is naturally order-independent
    ds = _DisjointSet(n)
    for e in edges:
        ds.union(e.left_position, e.right_position)
    node_cluster, grouped = _build(n, ds)
    clusters = {}
    for cid, pos in grouped.items():
        rep_pos = min(pos)
        clusters[cid] = Cluster(
            cluster_id=cid,
            member_positions=set(pos),
            representative_position=rep_pos,
            representative=records[rep_pos] if records is not None else None,
        )
    return ClusterAssignment(
        node_cluster=node_cluster,
        clusters=clusters,
        n_pairs_evaluated=len(edges),
        n_pairs_matched=len(edges),
    )


def _dense_assignment(node_cluster: dict[int, int], n: int,
                      records: Optional[Sequence[Any]], n_matched: int = 0) -> ClusterAssignment:
    """Build a ClusterAssignment from a node->cluster-id map.

    ``node_cluster`` must already map every node to its (min-position) cluster
    id.  Singletons are added automatically for untouched nodes.
    """
    groups: dict[int, list[int]] = {}
    for node in range(n):
        cid = node_cluster.get(node, node)
        groups.setdefault(cid, []).append(node)
    clusters = {}
    for cid, pos in groups.items():
        rep_pos = min(pos)
        clusters[cid] = Cluster(
            cluster_id=cid,
            member_positions=set(pos),
            representative_position=rep_pos,
            representative=records[rep_pos] if records is not None else None,
        )
    return ClusterAssignment(
        node_cluster={node: node_cluster.get(node, node) for node in range(n)},
        clusters=clusters,
        n_pairs_evaluated=n_matched,
        n_pairs_matched=n_matched,
    )


def streaming_distributed_closure(
    edge_chunks,
    n: int,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Transitive closure over a **stream** of ScoredPair chunks (streaming reduce).

    Consumes an iterable of edge chunks (each a sequence of ``ScoredPair``)
    and union-finds incrementally, so peak memory is bounded by the largest
    chunk rather than the full edge set.  Identical to the single-process
    transitive closure given the same edges.
    """
    ds = _DisjointSet(n)
    n_matched = 0
    for chunk in edge_chunks:
        for e in chunk:
            ds.union(e.left_position, e.right_position)
            n_matched += 1
    node_cluster, _ = _build(n, ds)
    return _dense_assignment(node_cluster, n, records, n_matched)


def _local_component_map(edges, n: int) -> dict[int, int]:
    """Union-find one edge partition; return ``{touched_node: local_min_node}``."""
    ds = _DisjointSet(n)
    for e in edges:
        ds.union(e.left_position, e.right_position)
    groups: dict[int, list[int]] = {}
    for e in edges:
        for node in (e.left_position, e.right_position):
            r = ds.find(node)
            groups.setdefault(r, []).append(node)
    comp_min = {r: min(nodes) for r, nodes in groups.items()}
    return {node: comp_min[ds.find(node)] for e in edges for node in (e.left_position, e.right_position)}


def distributed_closure_reduce(
    edges: Sequence[ScoredPair],
    n: int,
    n_workers: int = 1,
    executor: Optional[Any] = None,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Exact connected components across **machines** (distributed reduce).

    Edges are partitioned by :func:`hash_pair` owner; each worker union-finds
    its own partition and returns ``{node: local_min}`` for touched nodes; the
    driver merges worker-local components that share a node (into min-position
    ids).  The result is bit-for-bit identical to the single-process closure.

    ``executor`` is optional; when ``None`` a :class:`ProcessPoolExecutor` is
    used.  With ``n_workers == 1`` this degenerates to the single union-find.
    """
    if n_workers == 1:
        return distributed_closure(edges, n, records=records)

    owned: dict[int, list[ScoredPair]] = {w: [] for w in range(n_workers)}
    for e in edges:
        owner = hash_pair(e.left_position, e.right_position, n_workers)
        owned[owner].append(e)

    from functools import partial

    worker = partial(_local_component_map, n=n)
    if executor is not None:
        maps = list(executor.map(worker, [owned[w] for w in range(n_workers)]))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            maps = list(ex.map(worker, [owned[w] for w in range(n_workers)]))
    return _merge_local_components(maps, n, records, len(edges))


def _merge_local_components(worker_maps, n: int,
                            records: Optional[Sequence[Any]], n_matched: int) -> ClusterAssignment:
    """Merge per-worker ``{node: local_min}`` maps into global min-position ids."""
    # Union (worker, local_label) keys that share a node.
    id_of: dict[tuple[int, int], int] = {}
    node_keys: dict[int, list[tuple[int, int]]] = {}
    for w, comp_map in enumerate(worker_maps):
        for node, label in comp_map.items():
            key = (w, label)
            if key not in id_of:
                id_of[key] = len(id_of)
            node_keys.setdefault(node, []).append(key)

    # Untouched nodes are singletons; no keys needed.
    touched = set(node_keys)
    uf = _DisjointSet(len(id_of) or 1)
    for keys in node_keys.values():
        first = id_of[keys[0]]
        for key in keys[1:]:
            uf.union(first, id_of[key])

    # Per merged key-component: the minimum node-label (a node id).  The key is
    # (worker, local_label), so the node label is key[1].
    comp_min: dict[int, int] = {}
    for key, idx in id_of.items():
        root = uf.find(idx)
        comp_min[root] = min(comp_min.get(root, 10 ** 18), key[1])

    node_cluster: dict[int, int] = {}
    for node, keys in node_keys.items():
        root = uf.find(id_of[keys[0]])
        node_cluster[node] = comp_min[root]
    return _dense_assignment(node_cluster, n, records, n_matched)