"""Group (composite) comparisons for co-dependent fields.

Conditional independence between comparisons is Fellegi-Sunter's usual
assumption, and it is routinely violated in practice: two compared fields can
agree/disagree *together* (an address and its postcode corrupted by the same
transcription error; a forename and surname from the same mis-keyed record).

FS under independence is a **main-effects-only** model of the comparison vector,
so its match weight is additive,

    W(a, p) = w_a(a) + w_p(p),

and dependence is an **interaction** term ``delta(a, p)`` that the additive
score cannot express.  A group comparison replaces two or more marginal
comparisons with **one** comparison whose levels are the cross-product of the
members' levels, so its per-level ``m/u`` are estimated **jointly** -- which
contains ``delta``:

    W(cell) = log[ m(cell) / u(cell) ].

How it is built
---------------
* Each member keeps its own levels and its own first-true-wins assignment
  semantics (null first, ELSE last); a pair’s **cell** is the tuple of its
  members’ assigned level indices.
* ``combine`` maps that tuple to a composite level index.  The default is the
  identity (full cross-product: every cell is its own level); any callable
  ``(tuple[int, ...]) -> int`` can instead **merge** cells (e.g. coarsen to
  “both agree / one agrees / neither”).
* Level ``m/u`` are **seeded as the product of the members’ marginal ``m/u``**
  (the independence baseline, ``m`` renormalised to sum to 1), and ``u`` as the
  product of the members' ``u``.  EM (or supervised calibration) then refines
  the joint cells -- i.e. it learns ``delta``, the departure from independence.
* The cell where **every** member is at a null level is marked ``is_null`` (no
  evidence when all members are missing); other partial-missing cells are
  ordinary joint cells, so a member being missing is modelled jointly rather
  than collapsed.
* The composite is exhaustive, so the level holding the all-ELSE cell (the
  least-agreeing combination) is placed last and given ``test=None`` -- the
  required ELSE fallback.

Use it **instead of**, not in addition to, the member comparisons (otherwise
their evidence is double-counted and the composite becomes dependent with the
leftover marginals); :func:`replace_with_group` does the swap.

Caveats
-------
* The cell count is the product of the members’ level counts, so **coarsen the
  members’ levels** (and/or merge cells with ``combine``) to keep cells
  estimable; a guard rejects pathologically large cross-products.
* Dependence *within* the group is modelled; independence is still assumed
  *between* the composite and the other comparisons (and between groups).
* ``combine`` (like ``custom_comparison``'s ``test`` callables) is a Python
  callable, so a group comparison is built programmatically and is not a
  JSON-declarable comparison for persistence/round-tripping.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ._comparison import Comparison, comparison_from_dict
from ._core import ComparisonSpec, PairValues, apply_default_mu, build_spec
from ._registry import register_comparison

__all__ = [
    "group_comparison",
    "group_comparison_spec",
    "group_comparison_builder",
    "replace_with_group",
]

#: Guard on the cross-product size (product of member level counts).
_MAX_CELLS = 100_000


def _as_spec(item: Any) -> ComparisonSpec:
    """Normalize a member declaration to a ``ComparisonSpec`` with default m/u."""
    if isinstance(item, ComparisonSpec):
        spec = item
    elif isinstance(item, Comparison):
        spec = item.spec()
    elif isinstance(item, dict):
        # Accept either the serialized form ({"type", "params"}) or the
        # comparison-set form ({"type": name, **kwargs}).
        if "params" in item:
            spec = comparison_from_dict(item).spec()
        else:
            from ._comparison import make_comparison

            kwargs = dict(item)
            name = kwargs.pop("type")
            spec = make_comparison(name, **kwargs).spec()
    elif isinstance(item, str):
        from ._comparison import make_comparison

        spec = make_comparison(item).spec()
    else:
        spec_method = getattr(item, "spec", None)
        if spec_method is None:
            raise TypeError(
                "group_comparison members must be ComparisonSpec / Comparison / "
                f"comparison dict / registered name; got {type(item).__name__}"
            )
        spec = spec_method()
    apply_default_mu(spec)  # fill any missing defaults (idempotent)
    return spec


def group_comparison(
    output_column_name: str,
    members: Sequence[Any],
    *,
    combine: Optional[Callable[[tuple[int, ...]], int]] = None,
    labels: Optional[Sequence[str]] = None,
) -> Comparison:
    """Declare a composite comparison over co-dependent comparisons.

    Returns a :class:`Comparison` (registry name ``"group_comparison"``) whose
    :meth:`Comparison.spec` builds the composite; see :func:`group_comparison_spec`
    for the underlying spec and the full description.  Because the members and
    ``combine`` are constructor parameters, the declaration round-trips through
    the registry (EM calibration, ``resolved()``/``from_resolved()``) in-process;
    it is **not** JSON-serializable when ``combine`` is a callable.
    """
    members = list(members)
    fields = _member_fields([_as_spec(m) for m in members])
    return Comparison(
        name="group_comparison",
        params={
            "output_column_name": output_column_name,
            "members": members,
            "combine": combine,
            "labels": list(labels) if labels is not None else None,
        },
        fields=fields,
    )


def group_comparison_spec(
    output_column_name: str,
    members: Sequence[Any],
    *,
    combine: Optional[Callable[[tuple[int, ...]], int]] = None,
    labels: Optional[Sequence[str]] = None,
) -> ComparisonSpec:
    """Build a composite comparison over two or more co-dependent comparisons.

    Parameters
    ----------
    output_column_name:
        Name of the composite comparison (its ``output_column_name``).
    members:
        The comparisons whose dependence is modelled: each a
        :class:`ComparisonSpec`, :class:`Comparison`, comparison dict, or
        registered comparison name.  Ordered by decreasing agreement.
    combine:
        Optional ``(tuple_of_member_level_indices) -> int`` merging cells into
        composite levels; ``None`` (default) keeps the full cross-product.
    labels:
        Optional labels for the composite levels (length = number of composite
        levels); defaults are derived from the members' level labels.

    Returns
    -------
    ComparisonSpec -- use it in place of the member comparisons.
    """
    specs = [_as_spec(m) for m in members]
    if len(specs) < 2:
        raise ValueError("group_comparison needs at least two member comparisons")
    dims = [len(s.levels) for s in specs]
    total = int(np.prod(dims))
    if total > _MAX_CELLS:
        raise ValueError(
            f"group_comparison cross-product has {total:,} cells (> {_MAX_CELLS:,}); "
            "coarsen the members' levels or pass a `combine` that merges cells"
        )

    cells = list(itertools.product(*[range(d) for d in dims]))  # C-order: last axis fastest
    if combine is None:
        cell_level = np.arange(total, dtype=np.int64)
    else:
        raw = np.asarray([combine(t) for t in cells], dtype=np.int64)
        if raw.min() < 0:
            raise ValueError("combine() must return non-negative level indices")
        _, cell_level = np.unique(raw, return_inverse=True)  # compact to 0..C-1
    n_levels = int(cell_level.max()) + 1

    # Place the level holding the all-ELSE cell (last cell) last, so it can be
    # the required ELSE fallback (test=None).
    else_level = int(cell_level[total - 1])
    order = [c for c in range(n_levels) if c != else_level] + [else_level]
    remap = np.empty(n_levels, dtype=np.int64)
    remap[order] = np.arange(n_levels, dtype=np.int64)
    cell_level = remap[cell_level]

    # A cell is "all-null" when every member is at one of its null levels.
    null_sets = [
        {j for j, lv in enumerate(s.levels) if lv.is_null} for s in specs
    ]
    all_null = np.array(
        [all(t[i] in null_sets[i] for i in range(len(specs))) for t in cells], dtype=bool
    )

    def _cell_is_null(level: int) -> bool:
        cells_of = cell_level == level
        return bool(cells_of.any() and all_null[cells_of].all())

    # Seed m/u as the product of the members' marginal m/u (independence
    # baseline); a member at a null level contributes a neutral factor 1.
    def _seed(cell_index: int) -> tuple[float, float]:
        t = cells[cell_index]
        m = u = 1.0
        for i, j in enumerate(t):
            lv = specs[i].levels[j]
            if lv.is_null:
                continue
            m *= float(lv.m) if lv.m is not None else 1.0
            u *= float(lv.u) if lv.u is not None else 1.0
        return m, u

    def _label(cell_index: int) -> str:
        parts = []
        for i, j in enumerate(cells[cell_index]):
            lbl = specs[i].levels[j].label
            if lbl:
                parts.append(lbl)
        return " & ".join(parts) or f"group level {cell_index}"

    level_dicts: list[dict] = []
    non_null_ms: list[float] = []
    reps: list[int] = []
    for c in range(n_levels):
        rep = int(np.argmax(cell_level == c))  # first cell of this level
        reps.append(rep)
        is_null = _cell_is_null(c)
        entry: dict = {
            "label_for_charts": (labels[c] if labels is not None else _label(rep)),
            "is_null_level": bool(is_null),
        }
        if not is_null:
            m, u = _seed(rep)
            entry["m_probability"] = m
            entry["u_probability"] = u
            non_null_ms.append(m)
        if c != n_levels - 1:  # last level is the ELSE fallback
            entry["test"] = _level_test(c)
        level_dicts.append(entry)

    # Renormalize the seeded m-probabilities over the non-null levels.
    total_m = sum(non_null_ms)
    if total_m > 0:
        pos = 0
        for entry in level_dicts:
            if "m_probability" in entry:
                entry["m_probability"] = non_null_ms[pos] / total_m
                pos += 1

    fields = _member_fields(specs)
    return build_spec(
        output_column_name=output_column_name,
        level_dicts=level_dicts,
        fields=fields,
        prescore=_group_prescore(specs, tuple(dims), cell_level),
    )


def _level_test(level: int) -> Callable[[PairValues, Optional[dict]], np.ndarray]:
    def test(pv: PairValues, cache: Optional[dict] = None) -> np.ndarray:
        return np.asarray(cache["_group"]) == level

    return test


def _group_prescore(
    specs: Sequence[ComparisonSpec],
    dims: tuple[int, ...],
    cell_level: np.ndarray,
) -> Callable[[PairValues], dict]:
    def prescore(pv: PairValues) -> dict:
        # Lazy import: ``scoring._levels`` imports ``comparisons`` (cycle).
        from ..scoring._levels import _assign_levels

        if pv.n == 0:
            return {"_group": np.empty(0, dtype=np.int64)}
        assigned = [_assign_levels(spec, pv) for spec in specs]
        cell = np.ravel_multi_index(tuple(assigned), dims)  # C-order, matches cells order
        return {"_group": cell_level[cell]}

    return prescore


def _member_fields(specs: Sequence[ComparisonSpec]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for field in spec.fields:
            if field not in seen:
                seen.add(field)
                out.append(field)
    return tuple(out)


def group_comparison_builder(
    *,
    members: Sequence[Any],
    output_column_name: str = "group",
    combine: Optional[Callable[[tuple[int, ...]], int]] = None,
    labels: Optional[Sequence[str]] = None,
) -> ComparisonSpec:
    """Registry factory for ``group_comparison`` (keyword args -> spec)."""
    return group_comparison_spec(
        output_column_name, members, combine=combine, labels=labels
    )


def replace_with_group(
    comparisons: Sequence[Any],
    group: Any,
    members: Sequence[Any],
) -> list[Any]:
    """Return ``comparisons`` with ``members`` replaced by the ``group`` spec.

    Members are matched by ``output_column_name`` (against both ``Comparison``
    declarations and raw specs); the group spec is appended.  Use this so the
    marginal comparisons are not double-counted alongside the composite.
    """
    drop_names = {_as_spec(m).output_column_name for m in members}
    out: list[Any] = []
    for item in comparisons:
        name = (
            item.output_column_name()
            if isinstance(item, Comparison)
            else getattr(item, "output_column_name", None)
        )
        if name in drop_names:
            continue
        out.append(item)
    out.append(group)
    return out


# Register for discoverability (mirrors ``time_decayed_comparison``); note the
# callable ``combine`` is not JSON-declarable (see the module docstring).
register_comparison(
    "group_comparison",
    group_comparison_builder,
    fields=(),
    defaults={"output_column_name": "group"},
    description="Composite comparison over co-dependent comparisons (models conditional dependence).",
)
