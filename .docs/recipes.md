# Recipes

Pragmatic, copy-pasteable recipes for building specialised comparators and
pipelines on top of the framework's extension points.  Each recipe is
self-contained and uses only public API.  For the underlying concepts, see
[`architecture.md`](architecture.md) and the [`user_guide.md`](user_guide.md).

- [Recipe: a time-sliced comparator (time-decayed address matching)](#recipe-a-time-sliced-comparator-time-decayed-address-matching)
  - [The one-line version: `time_decay_wrapper`](#the-one-line-version-time_decay_wrapper)
  - [The hand-rolled version (background)](#the-hand-rolled-version-background)

---

## Recipe: a time-sliced comparator (time-decayed address matching)

### The one-line version: `time_decay_wrapper`

The framework ships a wrapper that **turns any comparison into its time-sliced,
time-decayed version**, so you don't hand-write the band levels.  It crosses
every non-null level of the inner comparison with a set of time bands, decays
each level's `m` by the band's weight (and renormalizes so the comparison's
m-probabilities sum to 1), and adds a null level for a missing/invalid
timestamp (so a missing date never reads as a huge time gap):

```python
from vectorer.comparisons import (
    make_comparison,
    time_decay_wrapper,
    time_decayed_comparison_builder,
)

base = make_comparison("jaro_winkler_at_thresholds", col_name="address",
                       score_threshold_or_thresholds=[0.95, 0.8])

decayed = time_decay_wrapper(
    base,
    time_col="event_date",
    bands=[(0, 90, 1.0),        # (lo_days, hi_days, weight): close = full weight
           (90, 365, 0.4),      #     mid = 0.4x
           (365, 10**9, 0.05)], #     far = 0.05x
)

# or via the registered name (col_name forwarded when `comparison` is a name):
decayed = time_decayed_comparison_builder("exact_match", col_name="email",
                                          time_col="ts", bands=[(0, 30, 1.0)])
```

Then use it like any comparison:

```python
from vectorer.scoring import FellegiSunterScorer

scorer = FellegiSunterScorer.from_comparisons([decayed, *other_comparisons])
# or fit the close-vs-far weights from labelled pairs:
scorer = scorer.calibrate_from_pairs(labelled_pairs, smoothing=0.1)
```

Verified behaviour (default m/u): an exact address close in time scores strongly
(≈0.067), the same address far in time scores ~18x weaker (≈0.0036), and a
missing date scores the base prior (≈0.0001).  The inner comparison's own
`prescore` (cached score arrays) is preserved — the wrapper merges its time
difference into the same cache, so banded scoring still costs one similarity
pass per batch.

**Details / options:**

- `time_decay_wrapper(inner, time_col, bands, keep_inner_null=False)` takes a
  `ComparisonSpec`, a declared `Comparison`, or a `{"type", ...}` dict as
  `inner`.  With `keep_inner_null=True`, the inner comparison's own null level
  (e.g. a missing address) is preserved on top of the time-null level.
- `bands` default to `[(0, 30, 1.0), (30, 365, 0.3), (365, 1e9, 0.05)]`.
- The wrapper output column is `"<inner>_decayed"`, and its `fields` are the
  inner comparison's fields **plus** `time_col` — so the timestamp must be a
  column present in the scored records.

### The hand-rolled version (background)

For reference / custom control, the wrapper is a thin composition over the same
public primitives a hand-written time comparator uses.  Read this if you want to
understand the mechanism or build a variant (e.g. non-multiplicative decays,
per-field decay, or a bespoke weighting).

A "time comparator" is not a single similarity — it is a **composite
comparator** over two columns (`address`, `event_date`).  The comparison is a
list of ordered FS levels, each one a conjunction:

```
level k = (address similarity >= addr_min_k)  AND  (time difference in [lo_k, hi_k])
```

Each level carries its own `m`/`u`, so `weight_k = log(m_k / u_k)` encodes the
time decay: a tight time band on an exact address gets a large weight; a wide
time band on a fuzzy address gets a small one.

### How it fits the framework (no framework changes needed)

- A custom comparison whose levels are **vectorized `test(PairValues, cache)`**
  predicates — the same machinery the built-in comparisons use.
- **`prescore`** computes the address similarity and the timestamp difference
  *once* per batch; every band-level test reads the cached arrays.  So the
  `[0,30]`, `[30,365]`, `[365,∞)` bands cost one Jaro pass, not three.
- **`fields=(addr_col, time_col)`** tells the scorer which record columns to
  materialize into `PairValues` — critical for a multi-column comparator.
- Explicit per-level `m/u` works immediately, and `calibrate_from_pairs` /
  `fit_em` can fit them from data (they drive the same level-assignment path).

### Recipe

```python
import numpy as np

from vectorer.comparisons import build_spec
from vectorer.sim import absolute_seconds_difference, jaro_winkler_similarity

DAY = 86400.0          # seconds per day


def address_time_comparison(addr_col="address", time_col="event_date"):
    """Ordered FS comparator: (address similarity) x (time-distance band).

    Levels: exact address in a tight band (strong) ... fuzzy address in a
    wide band (weak).  The per-level m/u below are illustrative — prefer
    calibrating them from labelled pairs.
    """

    def null_test(pv, cache):
        # Missing address, or unparseable/missing date -> NO evidence (BF=1),
        # so a missing date never reads as "huge time gap".
        addr_present = (pv.left(addr_col) != None) & (pv.right(addr_col) != None)  # noqa: E711
        return ~(addr_present & np.isfinite(cache["tdiff"]))

    def band(addr_min, lo_days, hi_days):
        def test(pv, cache):
            addr_ok = cache["addr_sim"] >= addr_min
            in_time = (cache["tdiff"] >= lo_days * DAY) & (cache["tdiff"] <= hi_days * DAY)
            return addr_ok & in_time
        return test

    def prescore(pv):
        # One Jaro pass + one date-parse pass, shared by every band level.
        return {
            "addr_sim": jaro_winkler_similarity(pv.left(addr_col), pv.right(addr_col)),
            "tdiff": absolute_seconds_difference(pv.left(time_col), pv.right(time_col)),
        }

    levels = [
        {"label_for_charts": "address or date missing", "is_null_level": True,
         "test": null_test},
        # strongest -> weakest. Each level's m/u IS the time weighting.
        {"label_for_charts": "addr exact & t <= 30d",   "test": band(0.99, 0,    30),
         "m_probability": 0.45, "u_probability": 1e-6},
        {"label_for_charts": "addr exact & 30 < t <= 365d",
         "test": band(0.99, 30, 365), "m_probability": 0.30, "u_probability": 1e-5},
        {"label_for_charts": "addr exact & t > 365d",
         "test": band(0.99, 365, 1e9), "m_probability": 0.15, "u_probability": 1e-4},
        {"label_for_charts": "addr fuzzy & t <= 365d",
         "test": band(0.85, 0, 365), "m_probability": 0.08, "u_probability": 1e-3},
        {"label_for_charts": "addr fuzzy & t > 365d",
         "test": band(0.85, 365, 1e9), "m_probability": 0.02, "u_probability": 1e-2},
        {"label_for_charts": "else", "test": None},
    ]

    return build_spec(
        output_column_name="address_time",
        level_dicts=levels,
        fields=(addr_col, time_col),   # <-- both columns, or scoring breaks
        prescore=prescore,
    )
```

### Use it

```python
from vectorer.scoring import FellegiSunterScorer
from vectorer.comparisons import make_comparison

time_comp = address_time_comparison(addr_col="address", time_col="event_date")

# 1. With the explicit (illustrative) m/u above:
scorer = FellegiSunterScorer.from_comparisons([time_comp, *other_comparisons])

# 2. Better: fit the close-vs-far weights from labelled address-time pairs.
#    Provide is_match + "<field>_l" / "<field>_r" for the canonical fields.
scorer = FellegiSunterScorer.from_comparisons([time_comp, *other_comparisons])
scorer = scorer.calibrate_from_pairs(labelled_pairs, smoothing=0.1)
```

Both the incremental and batch pipelines, and the Link mode, accept this
comparison wherever a `comparison` is expected — pass `scorer=` (or the
comparison in a `build_*_pipeline`/`RecordLinker`).

### Make it first-class (optional)

If you use this often, register it as a named, configurable comparison:

```python
from vectorer.comparisons import register_comparison

register_comparison(
    "address_time_comparison",
    lambda addr_col, time_col, **kw: address_time_comparison(addr_col, time_col),
    fields=("addr_col", "time_col"),
    defaults={"addr_col": "address", "time_col": "event_date"},
    description="Address similarity weighted by time-distance band",
)

# then:
from vectorer.comparisons import make_comparison
make_comparison("address_time_comparison", addr_col="addr", time_col="ts")
```

### Caveats

- **Null handling matters.** If the address *or* the date is missing/invalid,
  the pair must hit the null level (no evidence), not a "far in time" band.
  Otherwise a missing date is read as a huge time gap and wrongly weakens an
  exact address match.  The `null_test` above handles this.
- **Order levels strongest → weakest.** FS takes the *first* matching level, so
  keep the grid consistent (close+exact first, then widening time, then fuzzing
  the address) so a pair never matches a weaker level when a stronger one
  applies.
- **Timestamps must be canonical.** The unit is seconds and the format/offset
  must be consistent across the two sides (relevant in the Link mode, whose
  `FieldMap` normalisers can align them).
- **Weights are relative, not absolute.** The `m/u` in the recipe are
  illustrative; calibrate (or EM) them so the close-vs-far gap reflects your
  actual data.
- **Multi-column comparators need `fields=`.** Use `build_spec(..., fields=
  (addr_col, time_col))`, not the declarative-conditions `custom_comparison`,
  because callable-level settings can't infer the columns automatically.
