# Recipes

Pragmatic, copy-pasteable recipes for building specialised comparators and
pipelines on top of the framework's extension points.  Each recipe is
self-contained and uses only public API.  For the underlying concepts, see
[`architecture.md`](architecture.md) and the [`user_guide.md`](user_guide.md).

- [Recipe: a time-sliced comparator (time-decayed address matching)](#recipe-a-time-sliced-comparator-time-decayed-address-matching)

---

## Recipe: a time-sliced comparator (time-decayed address matching)

### Goal

Compare records on an address **and** a timestamp, so that a match on an exact
address is strong evidence when the two records are close in time but weak
evidence when they are far apart in time.  Concretely: under Fellegi–Sunter, an
*exact address within 30 days* should outweigh an *exact address from years
ago*.

### Idea

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
