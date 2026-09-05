"""Vectorized string, date and geo similarity/distance functions.

Every function here is computed natively in NumPy over whole batches of
pairs (no SQL engine, no external fuzzy-matching dependency):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - function
     - semantics
   * - ``jaro_similarity``
     - Jaro similarity in [0, 1]
   * - ``jaro_winkler_similarity``
     - Jaro-Winkler (prefix 0.1, len 4)
   * - ``levenshtein_distance``
     - edit distance
   * - ``damerau_levenshtein_distance``
     - optimal-string-alignment distance
   * - ``jaccard``
     - Jaccard index over list columns
   * - ``cosine_similarity``
     - cosine over list columns
   * - ``array_intersect_size``
     - overlap size
   * - ``absolute_seconds_difference``
     - abs gap in seconds
   * - ``haversine_km``
     - spherical-law-of-cosines km

All functions accept parallel sequences of scalar values (``None`` = missing)
and return a ``np.ndarray`` of the same length, so they vectorize the scoring
stage across every candidate pair at once.  ``jaro*``/``levenshtein*`` use the
shared batched char-matrix machinery in :mod:`vectorer.sim`; multi-argument
functions (dates, coordinates) operate on parallel sequences.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Sequence, Tuple

import numpy as np

#: Earth radius (km) used by the spherical law of cosines.
EARTH_RADIUS_KM = 6371.0

#: Seconds per month/year used by the date-difference thresholds.
SECONDS_PER_MONTH = 2629800.0
SECONDS_PER_YEAR = 31557600.0

_ISO_STRIP = re.compile(r"Z$")
_UTC = timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)


# ---------------------------------------------------------------------------
# scalar value helpers (None-resistant, cast to str for string comparators)
# ---------------------------------------------------------------------------


def _string_pairs(a: Sequence, b: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize pair values to object arrays of strings plus a "both present" mask.

    ``None`` maps to the *empty, absent* sentinel so it can never accidentally
    equal another value; real empty strings remain present (empty strings
    are data, not null).
    """
    n = len(a)
    left = np.empty(n, dtype=object)
    right = np.empty(n, dtype=object)
    present = np.empty(n, dtype=bool)
    for i in range(n):
        av = a[i]
        bv = b[i]
        if av is None or bv is None:
            left[i] = "__vectorer_missing__"
            right[i] = "__vectorer_missing__"
            present[i] = False
            continue
        left[i] = str(av)
        right[i] = str(bv)
        present[i] = True
    return left, right, present


def _to_char_matrix(values: np.ndarray, lengths: np.ndarray, width: int) -> np.ndarray:
    """Pad strings into an ``(m, width)`` int32 matrix of ordinal values."""
    m = len(values)
    out = np.zeros((m, width), dtype=np.int32)
    for k in range(m):
        s = values[k]
        if not len(s):
            continue
        if len(s) != len(s.encode("utf-16-le", "surrogatepass")) // 2:
            # Non-BMP characters: fall back to per-character ord.
            out[k, : len(s)] = np.array([ord(ch) for ch in s], dtype=np.int32)
        else:
            codes = np.frombuffer(s.encode("utf-16-le"), dtype="<u2")
            out[k, : len(s)] = codes.astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Scalar primitives (used for small batches, where per-call NumPy overhead
# dominates the vectorized paths)
# ---------------------------------------------------------------------------

#: Below this many *non-equal* pairs the scalar loops beat the vectorized
#: inner loops; above it the vectorized-over-batch path wins.
_SMALL_BATCH = 64


def _scalar_jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_dist = max(la, lb) // 2 - 1
    match_dist = max(match_dist, 0)
    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    for i in range(la):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, lb)
        for j in range(lo, hi):
            if not b_matches[j] and a[i] == b[j]:
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(la):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    m = matches
    return (m / la + m / lb + (m - transpositions) / m) / 3.0


def _scalar_jaro_winkler(
    a: str,
    b: str,
    prefix_scale: float = 0.1,
    prefix_length: int = 4,
    boost_threshold: float = 0.7,
) -> float:
    sj = _scalar_jaro(a, b)
    if sj <= boost_threshold:
        return sj
    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix += 1
        else:
            break
    prefix = min(prefix, prefix_length)
    return sj + prefix * prefix_scale * (1 - sj)


def _scalar_levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _scalar_osa(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


# ---------------------------------------------------------------------------
# Jaro / Jaro-Winkler (batched, fully vectorized over pairs)
# ---------------------------------------------------------------------------


def jaro_similarity(a: Sequence, b: Sequence) -> np.ndarray:
    """Vectorized Jaro similarity in [0, 1]."""
    a_str, b_str, present = _string_pairs(a, b)
    n = len(a_str)
    out = np.zeros(n, dtype=np.float64)

    equal = present & (a_str == b_str)
    out[equal] = 1.0
    work = np.flatnonzero(~equal & present)
    m = len(work)
    if m == 0:
        return out

    if m <= _SMALL_BATCH:
        for i in range(m):
            out[work[i]] = _scalar_jaro(a_str[work[i]], b_str[work[i]])
        return out

    la = np.array([len(a_str[i]) for i in work], dtype=np.int64)
    lb = np.array([len(b_str[i]) for i in work], dtype=np.int64)
    width = max(int(la.max()), int(lb.max()), 1)
    A = _to_char_matrix(a_str[work], la, width)
    B = _to_char_matrix(b_str[work], lb, width)

    match_dist = np.maximum(np.maximum(la, lb) // 2 - 1, 0)
    matched_a = np.zeros((m, width), dtype=bool)
    matched_b = np.zeros((m, width), dtype=bool)
    count = np.zeros(m, dtype=np.int64)

    for i in range(width):
        active = i < la
        if not active.any():
            break
        j_low = np.maximum(0, i - match_dist)
        j_high = np.minimum(lb - 1, i + match_dist)
        Ai = A[:, i]
        for j in range(width):
            candidate = (
                active
                & (j >= j_low)
                & (j <= j_high)
                & (~matched_b[:, j])
                & (~matched_a[:, i])
                & (Ai == B[:, j])
            )
            if candidate.any():
                matched_a[candidate, i] = True
                matched_b[candidate, j] = True
                count[candidate] += 1

    # Transpositions: imperfect matches between the matched-character sequences.
    max_matches = int(count.max()) if m else 0
    if max_matches > 0:
        cols = np.arange(width)[None, :]
        ma_vals = np.where(matched_a, cols, width)
        mb_vals = np.where(matched_b, cols, width)
        ma_vals = np.sort(ma_vals, axis=1)[:, :max_matches]
        mb_vals = np.sort(mb_vals, axis=1)[:, :max_matches]
        use = np.arange(max_matches)[None, :] < count[:, None]
        # Clip padded column indices before indexing; ``use`` discards them.
        ma_idx = np.clip(ma_vals, 0, width - 1)
        mb_idx = np.clip(mb_vals, 0, width - 1)
        aligned_a = np.where(use, A[np.arange(m)[:, None], ma_idx], -1)
        aligned_b = np.where(use, B[np.arange(m)[:, None], mb_idx], -2)
        mismatch = (aligned_a != aligned_b) & use
        transpositions = mismatch.sum(axis=1) // 2
    else:
        transpositions = np.zeros(m)

    ok = count > 0
    score = np.zeros(m, dtype=np.float64)
    c = count[ok].astype(np.float64)
    score[ok] = (
        c / la[ok]
        + c / lb[ok]
        + (c - transpositions[ok]) / c
    ) / 3.0
    # Guard: transpositions may (rarely) exceed matches on pathological aliasing.
    score = np.clip(score, 0.0, 1.0)
    out[work] = score
    return out


def jaro_winkler_similarity(
    a: Sequence,
    b: Sequence,
    prefix_scale: float = 0.1,
    prefix_length: int = 4,
    boost_threshold: float = 0.7,
) -> np.ndarray:
    """Vectorized Jaro-Winkler similarity (prefix bonus above 0.7)."""
    a_str, b_str, present = _string_pairs(a, b)
    n = len(a_str)
    if n <= _SMALL_BATCH:
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if present[i]:
                out[i] = _scalar_jaro_winkler(
                    a_str[i], b_str[i],
                    prefix_scale=prefix_scale,
                    prefix_length=prefix_length,
                    boost_threshold=boost_threshold,
                )
        return out

    base = jaro_similarity(a, b)
    if not len(base):
        return base
    a_str, b_str, present = _string_pairs(a, b)
    # Prefix bonus is only applied above the boost threshold.
    boost = base > boost_threshold
    if not boost.any():
        return base

    la = np.array([len(a_str[i]) for i in range(len(a_str))])
    lb = np.array([len(b_str[i]) for i in range(len(b_str))])
    width = max(int(la.max()), int(lb.max()), 1)
    A = _to_char_matrix(a_str, la, width)
    B = _to_char_matrix(b_str, lb, width)
    running = np.ones(len(a_str), dtype=bool)
    prefix = np.zeros(len(a_str), dtype=np.int64)
    for k in range(int(prefix_length)):
        if k >= width:
            break
        step = running & (k < la) & (k < lb) & (A[:, k] == B[:, k])
        prefix += step.astype(np.int64)
        running &= step

    scale = np.minimum(prefix.astype(np.float64), prefix_length)
    bonus = scale * prefix_scale * (1.0 - base)
    out = base + np.where(boost, bonus, 0.0)
    return out


# ---------------------------------------------------------------------------
# Edit distances (batched dynamic programming over pairs)
# ---------------------------------------------------------------------------


def _char_pairs_padded(
    a: Sequence, b: Sequence
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_str, b_str, present = _string_pairs(a, b)
    return a_str, b_str, present


def _batch_dp(
    a: Sequence,
    b: Sequence,
    transpositions: bool,
    chunk: int = 8192,
) -> np.ndarray:
    """Batched Levenshtein / OSA Damerau-Levenshtein distance.

    For large batches the DP recurrence is evaluated column-by-column,
    vectorized across the whole chunk; for small batches a scalar loop over the
    pairs is faster (no per-cell NumPy overhead).
    """
    a_str, b_str, present = _char_pairs_padded(a, b)
    n = len(a_str)
    out = np.zeros(n, dtype=np.int64)

    equal = (a_str == b_str) & present
    out[equal] = 0
    work = np.flatnonzero(~equal & present)
    if len(work) == 0:
        return out

    if len(work) <= _SMALL_BATCH:
        fn = _scalar_osa if transpositions else _scalar_levenshtein
        for i in range(len(work)):
            out[work[i]] = fn(a_str[work[i]], b_str[work[i]])
        return out

    # Padded char matrices for exactly the pairs that need computing.
    la = np.array([len(a_str[i]) for i in work], dtype=np.int64)
    lb = np.array([len(b_str[i]) for i in work], dtype=np.int64)
    width = max(int(la.max()), int(lb.max()), 1)
    A = _to_char_matrix(a_str[work], la, width)
    B = _to_char_matrix(b_str[work], lb, width)

    result = np.zeros(len(work), dtype=np.int64)
    for start in range(0, len(work), chunk):
        stop = min(start + chunk, len(work))
        ca = A[start:stop]
        cb = B[start:stop]
        c_la = la[start:stop]
        c_lb = lb[start:stop]
        c = len(ca)
        L = width + 1
        # d[i, j] laid out as (chunk, L, L)
        d = np.zeros((c, L, L), dtype=np.int64)
        d[:, 1:, 0] = np.arange(1, L)[None, :]
        d[:, 0, 1:] = np.arange(1, L)[None, :]
        for i in range(1, L):
            ai = ca[:, i - 1]
            d_i_1 = d[:, i - 1]
            for j in range(1, L):
                cost = (ai != cb[:, j - 1]).astype(np.int64)
                best = np.minimum(
                    np.minimum(d_i_1[:, j] + 1, d[:, i, j - 1] + 1),
                    d_i_1[:, j - 1] + cost,
                )
                if transpositions and i >= 2 and j >= 2:
                    trans_ok = (ai == cb[:, j - 2]) & (ca[:, i - 2] == cb[:, j - 1])
                    cand = d[:, i - 2, j - 2] + 1
                    best = np.where(trans_ok, np.minimum(best, cand), best)
                d[:, i, j] = best
        result[start:stop] = d[np.arange(c), c_la, c_lb]
    out[work] = result
    return out


def levenshtein_distance(a: Sequence, b: Sequence) -> np.ndarray:
    """Vectorized Levenshtein distance."""
    return _batch_dp(a, b, transpositions=False)


def damerau_levenshtein_distance(a: Sequence, b: Sequence) -> np.ndarray:
    """Vectorized Damerau-Levenshtein (OSA) distance.

    The optimal-string-alignment variant (adjacent transpositions cost one).
    """
    return _batch_dp(a, b, transpositions=True)


# ---------------------------------------------------------------------------
# List-valued comparators
# ---------------------------------------------------------------------------


def _coerce_lists(a: Sequence, b: Sequence) -> Tuple[list, list]:
    n = len(a)
    left = []
    right = []
    for i in range(n):
        av = a[i] if a[i] is not None else []
        bv = b[i] if b[i] is not None else []
        left.append(list(av))
        right.append(list(bv))
    return left, right


def jaccard(a: Sequence, b: Sequence) -> np.ndarray:
    """Vectorized Jaccard index over list-valued columns."""
    left, right = _coerce_lists(a, b)
    out = np.zeros(len(left), dtype=np.float64)
    for i, (l, r) in enumerate(zip(left, right)):
        if not l and not r:
            out[i] = 1.0
            continue
        if not l or not r:
            continue
        s_l = set(l)
        s_r = set(r)
        union = s_l | s_r
        if not union:
            out[i] = 1.0
            continue
        out[i] = len(s_l & s_r) / len(union)
    return out


def cosine_similarity(a: Sequence, b: Sequence) -> np.ndarray:
    """Cosine similarity over vector-valued columns."""
    n = len(a)
    width = 0
    for i in range(n):
        if a[i] is not None:
            width = max(width, len(a[i]))
        if b[i] is not None:
            width = max(width, len(b[i]))
    if width == 0:
        return np.zeros(n, dtype=np.float64)
    left = np.zeros((n, width), dtype=np.float64)
    right = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        if a[i] is not None:
            left[i, : len(a[i])] = a[i]
        if b[i] is not None:
            right[i, : len(b[i])] = b[i]
    num = (left * right).sum(axis=1)
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.zeros(n, dtype=np.float64))


def array_intersect_size(a: Sequence, b: Sequence) -> np.ndarray:
    """Overlap size of list-valued columns."""
    left, right = _coerce_lists(a, b)
    out = np.zeros(len(left), dtype=np.int64)
    for i, (l, r) in enumerate(zip(left, right)):
        out[i] = len(set(l) & set(r))
    return out


def pairwise_max_similarity(
    a: Sequence,
    b: Sequence,
    sim_fn,
) -> np.ndarray:
    """Max cross-pair similarity between two list-valued columns.

    Takes the maximum similarity over every (x in ``a``, y in ``b``) combination.
    """
    left, right = _coerce_lists(a, b)
    n = len(left)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        l, r = left[i], right[i]
        if not l or not r:
            continue
        best = 0.0
        for x in l:
            xr = [x] * len(r)
            sims = sim_fn(xr, r)
            best = max(best, float(sims.max()))
        out[i] = best
    return out


# ---------------------------------------------------------------------------
# Date / time
# ---------------------------------------------------------------------------

_EPOCH_CACHE: dict = {}


def _parse_epoch_value(value: Any, fmt: str | None) -> float:
    try:
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=_UTC)
        elif isinstance(value, str):
            s = value.strip()
            if fmt is None or fmt in ("ISO8601", "auto"):
                dt = datetime.fromisoformat(_ISO_STRIP.sub("+00:00", s))
            else:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_UTC)
        else:
            dt = datetime.fromisoformat(str(value))
        return (dt.astimezone(_UTC) - _EPOCH).total_seconds()
    except ValueError:
        return float("nan")


def _epoch_seconds(values: Sequence, fmt: str | None) -> np.ndarray:
    """Parse ISO-ish strings (or datetimes) to epoch seconds; NaN on failure.

    Parsing is cached per ``(format, value)`` so repeated comparisons over the
    same reference population parse each distinct value only once.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v = values[i]
        if v is None:
            continue
        key = (fmt, v)
        cached = _EPOCH_CACHE.get(key, None)
        if cached is None:
            e = _parse_epoch_value(v, fmt)
            if not np.isfinite(e):
                continue
            _EPOCH_CACHE[key] = e
            cached = e
        out[i] = float(cached)
    return out


def absolute_seconds_difference(a: Sequence, b: Sequence, fmt: str | None = None) -> np.ndarray:
    """Absolute difference in seconds between two date/time columns (NaN if invalid)."""
    la = _epoch_seconds(a, fmt)
    rb = _epoch_seconds(b, fmt)
    return np.abs(la - rb)


def invalid_date_mask(a: Sequence, b: Sequence, fmt: str | None = None) -> np.ndarray:
    """True where either date is unparseable (or missing) -- "transformed is NULL"."""
    la = _epoch_seconds(a, fmt)
    rb = _epoch_seconds(b, fmt)
    return ~np.isfinite(la) | ~np.isfinite(rb)


# ---------------------------------------------------------------------------
# Geo distance
# ---------------------------------------------------------------------------


def haversine_km(
    lat_l: Sequence, long_l: Sequence, lat_r: Sequence, long_r: Sequence
) -> np.ndarray:
    """Great-circle distance in km (spherical law of cosines).

    Clamps the cosine argument to [-1, 1].
    """
    lat1 = np.radians(np.asarray([float(v) for v in lat_l]))
    lon1 = np.radians(np.asarray([float(v) for v in long_l]))
    lat2 = np.radians(np.asarray([float(v) for v in lat_r]))
    lon2 = np.radians(np.asarray([float(v) for v in long_r]))
    cos_d = np.clip(
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
        -1.0,
        1.0,
    )
    return np.arccos(cos_d) * EARTH_RADIUS_KM


# ---------------------------------------------------------------------------
# Exact-match helpers and small vectorized text extras
# ---------------------------------------------------------------------------


def exact_equals(a: Sequence, b: Sequence) -> np.ndarray:
    """``a_l = a_r`` as a boolean mask (record equality)."""
    a_str, b_str, present = _string_pairs(a, b)
    return present & (a_str == b_str)


def null_or(other: np.ndarray, *others: np.ndarray) -> np.ndarray:
    """Truth table helper for null levels (used internally by the registry)."""
    out = other
    for o in others:
        out = out | o
    return out


def regexp_extract_group(value, pattern: str) -> str | None:
    """``regexp_extract(col, pattern, 0)`` -> match string or ``None`` when empty."""
    m = re.search(pattern, value)
    if m is None or m.group(0) == "":
        return None
    return m.group(0)


def postcode_parts(postcode: str) -> tuple[str | None, str | None, str | None]:
    """``(sector, district, area)`` extracted with the standard postcode patterns."""
    sector = regexp_extract_group(postcode, r"^[A-Za-z]{1,2}[0-9][A-Za-z0-9]? [0-9]")
    district = regexp_extract_group(postcode, r"^[A-Za-z]{1,2}[0-9][A-Za-z0-9]?")
    area = regexp_extract_group(postcode, r"^[A-Za-z]{1,2}")
    return sector, district, area


def email_parts(email: str) -> str | None:
    """Username (local part) for email comparisons; ``None`` when absent."""
    at = email.split("@")
    if len(at) != 2 or not at[0]:
        return None
    return at[0]