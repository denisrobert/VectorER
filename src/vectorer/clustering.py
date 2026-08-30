"""Swoosh clustering over Fellegi-Sunter scored pairs.

Swoosh (Benjelloun *et al.*, "Swoosh: A Generic Approach to Entity
Resolution") resolves a set of records into clusters by repeatedly *merging*
matching records: when two cluster representatives match, the clusters are
fused and a single representative is produced by a user-supplied merge rule.
The match test is the expensive stage (here the Fellegi-Sunter posterior at or
above a threshold); the merge rule decides who represents the fused cluster.

:func:`gswoosh` runs the algorithm over an adjacency (candidate) pair set --
the output of canopy blocking -- re-visiting pairs until no further merges
occur, which keeps a merged cluster's new representative available for
subsequent matches.  :class:`SwooshClusterer` provides the pair-driven entry
point used by the batch pipeline and returns a :class:`ClusterAssignment`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Sequence

from .records import to_record_dict

T = Any


@dataclass(frozen=True)
class ScoredPair:
    """A scored (record, record) pair from the Fellegi-Sunter stage."""

    left_position: int
    right_position: int
    probability: float
    match_weight: float = 0.0


@dataclass
class Cluster:
    """An equivalence class produced by Swoosh."""

    cluster_id: int
    member_positions: set[int] = field(default_factory=set)
    representative_position: int = -1
    representative: Any = None

    def add_member(self, position: int) -> None:
        self.member_positions.add(position)


@dataclass
class ClusterAssignment:
    """Result of a clustering run."""

    node_cluster: dict[int, int]
    clusters: dict[int, Cluster]
    n_pairs_evaluated: int = 0
    n_pairs_matched: int = 0

    def cluster_of(self, position: int) -> int:
        return self.node_cluster[position]

    def members_of(self, cluster_id: int) -> list[int]:
        return sorted(self.clusters[cluster_id].member_positions)

    def to_dict(self, records: Optional[Sequence[Any]] = None) -> dict:
        out: dict[int, dict] = {}
        for cluster_id, cluster in self.clusters.items():
            entry = {
                "cluster_id": int(cluster_id),
                "members": sorted(cluster.member_positions),
                "representative_position": int(cluster.representative_position),
            }
            if records is not None:
                entry["representative"] = records[cluster.representative_position]
            out[int(cluster_id)] = entry
        return out


def select_representative(
    records: Sequence[Any],
    positions: Sequence[int],
) -> tuple[Any, int]:
    """Pick the representative of a group of records.

    The richest record wins: the one with the most non-``None`` field values (a
    completeness heuristic), tie-broken by smallest position.
    """
    best_pos = positions[0]
    best_score = _completeness(records[best_pos])
    for position in positions[1:]:
        score = _completeness(records[position])
        if score > best_score or (score == best_score and position < best_pos):
            best_pos = position
            best_score = score
    return records[best_pos], best_pos


def _completeness(record: Any) -> int:
    if isinstance(record, dict):
        return sum(1 for v in record.values() if v is not None)
    data = getattr(record, "to_dict", None)
    if callable(data):
        return sum(1 for v in data().values() if v is not None)
    return 0


def _same_value(a: Any, b: Any) -> bool:
    """Content equality (``None`` equals only ``None``)."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _dedupe(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if not any(_same_value(value, existing) for existing in out):
            out.append(value)
    return out


def union_merge(
    records: Sequence[Any],
    positions: Sequence[int],
    fields: Optional[Sequence[str]] = None,
) -> tuple[dict, int]:
    """Union merge: a *master record* whose fields hold the union of every value
    seen across the matched records (the Swoosh Union Class).

    For each field, the result keeps the distinct non-``None`` values
    contributed by the members; a value that is already a ``set``/``frozenset``
    is treated as a union value and its elements are flattened in, while
    list/tuple values are kept whole (they may be comparison columns such as
    embedding vectors).  When every member is missing a field the field is
    ``None``.  The returned record is **synthetic** (it is not any member), so
    its position is ``-1`` and the Swoosh algorithms re-match it through the
    scorer's Union-Class lift (max posterior over cross value pairs).

    This merge function, paired with the existence-lifted match function, gives
    an ICAR-compatible (Union-Class) match/merge pair when the base match is
    reflexive and commutative.

    Returns
    -------
    (master_record, -1)
        ``master_record`` has frozenset-valued fields for every non-empty
        attribute union.
    """
    member_records = [to_record_dict(records[p]) for p in positions]
    keys = fields if fields is not None else sorted(
        {key for rec in member_records for key in rec}
    )
    result: dict[str, Any] = {}
    for key in keys:
        values: list[Any] = []
        for rec in member_records:
            value = rec.get(key)
            if value is None:
                continue
            if isinstance(value, (set, frozenset)):
                values.extend(list(value))
            else:
                values.append(value)
        unique = _dedupe(values)
        if not unique:
            result[key] = None
            continue
        try:
            result[key] = frozenset(unique)
        except TypeError:
            # Unhashable member values (e.g. dicts): keep a list.  Lists are
            # not treated as union values by the matcher's lift.
            result[key] = list(unique)
    return result, -1


def _timestamp_seconds(value: Any) -> float:
    if value is None:
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return float("-inf")
    return float("-inf")


def latest_merge(
    records: Sequence[Any],
    positions: Sequence[int],
    timestamp_field: str = "timestamp",
    fields: Optional[Sequence[str]] = None,
) -> tuple[dict, int]:
    """Timestamp merge: a *master record* whose fields hold the most recent
    value seen for each attribute, keyed on ``timestamp_field``.

    For each field the result keeps the value of the member with the latest
    ``timestamp_field`` that has a non-``None`` value for that field (so a
    field is filled from the newest record that actually carries it); the
    ``timestamp_field`` itself takes the newest timestamp.  The returned record
    is synthetic; its position is the index of the newest member, kept as an
    anchor for reporting.

    Returns
    -------
    (master_record, newest_member_position)
    """
    member_records = [to_record_dict(records[p]) for p in positions]
    ordered = sorted(member_records, key=lambda rec: _timestamp_seconds(rec.get(timestamp_field)), reverse=True)
    newest = ordered[0]
    newest_position = positions[member_records.index(newest)]
    keys = fields if fields is not None else sorted(
        {key for rec in member_records for key in rec}
    )
    result: dict[str, Any] = {}
    for key in keys:
        if key == timestamp_field:
            result[key] = newest.get(timestamp_field)
            continue
        for rec in ordered:  # newest first; first non-null wins
            value = rec.get(key)
            if value is not None:
                result[key] = value
                break
        else:
            result[key] = None
    return result, newest_position


class _DisjointSet:
    """Union-find over integer positions."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _build(n: int, ds: _DisjointSet) -> tuple[dict[int, int], dict[int, dict]]:
    """Group positions by root and assign deterministic (min-position) ids."""
    groups: dict[int, list[int]] = {}
    for position in range(n):
        groups.setdefault(ds.find(position), []).append(position)
    node_cluster: dict[int, int] = {}
    clusters: dict[int, dict] = {}
    for root, positions in groups.items():
        cluster_id = min(positions)
        for position in positions:
            node_cluster[position] = cluster_id
        clusters[cluster_id] = positions
    return node_cluster, clusters


def gswoosh(
    records: Sequence[Any],
    pairs: Sequence[tuple[int, int]],
    match_probability: Callable[[Any, Any], float],
    tau: float,
    merge: Callable[[Sequence[Any], Sequence[int]], tuple[Any, int]]
    = select_representative,
) -> ClusterAssignment:
    """G-Swoosh over ``records`` using candidate ``pairs`` as the scan set.

    Every pair of distinct clusters is tested with ``match_probability``, which
    receives the two **representative records** and returns the Fellegi-Sunter
    posterior; pairs at/above ``tau`` trigger a merge.  After a merge the
    merged cluster's representative is available to later pairs, so pairs are
    re-visited until a full pass produces no merges.

    Pair results are cached per representative pair, so re-scans do not
    re-evaluate the expensive scorer for unchanged representatives.

    Parameters
    ----------
    records:
        The reference population (positions referenced by ``pairs``).
    pairs:
        Candidate (position_i, position_j) pairs from the blocking stage.
    match_probability:
        ``match_probability(left_rep, right_rep) -> posterior`` (the match
        test).  The representatives are **record objects**, which may be
        synthetic master records produced by a merge function such as
        :func:`union_merge` or :func:`latest_merge`.
    tau:
        Pairs with posterior ``>= tau`` are matches.
    merge:
        Returns ``(representative_record, representative_position)`` for a
        group of records; defaults to :func:`select_representative`.  The
        representative need not be one of the original records (a synthetic
        master record returns position ``-1``); ``gswoosh`` stores the record
        object itself, so merge functions like :func:`union_merge` /
        :func:`latest_merge` work directly.
    """
    n = len(records)
    ds = _DisjointSet(n)
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    reps: dict[int, Any] = {i: records[i] for i in range(n)}
    rep_pos: dict[int, int] = {i: i for i in range(n)}
    prob_cache: dict[tuple[int, int], float] = {}

    evaluated = 0
    matched = 0
    pairs_sorted = sorted({tuple(sorted(p)) for p in pairs})
    while True:
        did_merge = False
        for i, j in pairs_sorted:
            ri, rj = ds.find(i), ds.find(j)
            if ri == rj:
                continue
            left_rep, right_rep = reps[ri], reps[rj]
            key = (id(left_rep), id(right_rep))
            if id(left_rep) > id(right_rep):
                key = (key[1], key[0])
            prob = prob_cache.get(key)
            if prob is None:
                prob = float(match_probability(left_rep, right_rep))
                prob_cache[key] = prob
                evaluated += 1
            if prob < tau:
                continue
            matched += 1
            merged_rep, merged_pos = merge(records, sorted(members[ri] | members[rj]))
            # Union the smaller root into the larger.
            if len(members[ri]) < len(members[rj]):
                ri, rj = rj, ri
            ds.union(ri, rj)
            root = ds.find(ri)
            members[root] = members[ri] | members[rj]
            if ri != root:
                members.pop(ri, None)
            if rj != root:
                members.pop(rj, None)
            reps[root] = merged_rep
            rep_pos[root] = merged_pos
            did_merge = True
        if not did_merge:
            break

    node_cluster, grouped = _build(n, ds)
    clusters: dict[int, Cluster] = {}
    for cluster_id, positions in grouped.items():
        root = ds.find(positions[0])
        clusters[cluster_id] = Cluster(
            cluster_id=cluster_id,
            member_positions=set(positions),
            representative_position=rep_pos[root],
            representative=reps[root],
        )
    return ClusterAssignment(
        node_cluster=node_cluster,
        clusters=clusters,
        n_pairs_evaluated=evaluated,
        n_pairs_matched=matched,
    )


def connected_components(
    n: int,
    pairs: Sequence[tuple[int, int]],
) -> dict[int, int]:
    """Connected components over candidate pairs (position -> component id).

    The component id is the minimum position in the component.  Building
    connected components is equivalent to Swoosh when the merge rule ignores
    downstream matches (i.e. matches are transitive).
    """
    ds = _DisjointSet(n)
    for i, j in pairs:
        ds.union(i, j)
    node_cluster, _ = _build(n, ds)
    return node_cluster


class SwooshClusterer:
    """Cluster records from scored Fellegi-Sunter candidate pairs.

    The batch pipeline scores the canopy candidate pairs, then this stage
    merges the above-threshold pairs.  Two modes:

    * :meth:`cluster` - transitive closure over the pre-scored pairs (cheap,
      the standard "score then cluster" workflow);
    * :meth:`cluster_with_merger` - full G-Swoosh, where a supplied
      ``scorer_match`` re-scores representative pairs lazily after merges.

    The ``merge`` function receives ``(records, positions)`` and returns
    ``(representative_record, representative_position)``; it may produce a
    *synthetic* master record (position ``-1``), e.g. via :func:`union_merge`
    or :func:`latest_merge`.  Representative records are stored by object, so
    synthetic representatives flow through transparently (cluster members are
    still the original positions).
    """

    def __init__(
        self,
        tau: float = 0.85,
        merge: Callable[[Sequence[Any], Sequence[int]], tuple[Any, int]]
        = select_representative,
    ) -> None:
        self.tau = float(tau)
        self.merge = merge

    def cluster(
        self,
        records: Sequence[Any],
        scored_pairs: Sequence[ScoredPair],
    ) -> ClusterAssignment:
        """Transitive closure over the scored pairs at/above ``tau``."""
        above = [
            (p.left_position, p.right_position)
            for p in scored_pairs
            if p.probability >= self.tau
        ]
        n = len(records)
        ds = _DisjointSet(n)
        for i, j in above:
            ds.union(i, j)
        node_cluster, grouped = _build(n, ds)
        clusters: dict[int, Cluster] = {}
        for cluster_id, positions in grouped.items():
            representative, rep_pos = self.merge(records, positions)
            clusters[cluster_id] = Cluster(
                cluster_id=cluster_id,
                member_positions=set(positions),
                representative_position=rep_pos,
                representative=representative,
            )
        return ClusterAssignment(
            node_cluster=node_cluster,
            clusters=clusters,
            n_pairs_evaluated=len(scored_pairs),
            n_pairs_matched=len(above),
        )

    def cluster_with_merger(
        self,
        records: Sequence[Any],
        pairs: Sequence[tuple[int, int]],
        scorer_match: Callable[[Any, Any], float],
    ) -> ClusterAssignment:
        """Full G-Swoosh: re-match for merged representatives via ``scorer_match``.

        ``scorer_match`` receives the two **representative records** (not
        positions) and returns the posterior; with :func:`union_merge` those
        representatives are synthetic master records whose set-valued fields
        are scored through the scorer's Union-Class lift.
        """
        return gswoosh(
            records=records,
            pairs=pairs,
            match_probability=scorer_match,
            tau=self.tau,
            merge=self.merge,
        )