"""Benchmark data loading.

Shared by the benchmark scripts.  Each benchmark normally synthesizes its own
population via its local generator; passing ``--data-file path/to/records.jsonl``
(or ``.json``) makes the benchmark use the supplied dataset instead of the
synthetic one.

File formats
------------
* ``.jsonl`` — one JSON object per line (most common for exports), or a
  whitespace-separated stream of objects.
* ``.json`` — either a JSON *list* of records, or a single object with a
  key under ``--data-key`` holding a list of records (e.g. ``{"people": [...]}``).

All records are plain mappings.  The benchmarks assume the compared columns
(``first_name``, ``last_name``, ``date_of_birth``, ``email``, ``address``) are
present, optionally ``None``; provide a real dataset with those fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_records(path: str, key: str | None = None) -> list[dict]:
    """Load records from a JSONL or JSON file.

    ``key`` selects the list inside a single-object JSON file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data-file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl" or p.suffix.lower() == ".ndjson":
        records: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict):
                records.append(obj)
            else:
                raise ValueError(f"expected a dict or list per JSONL line in {p}, got {type(obj).__name__}")
        return records

    if p.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            if key is None:
                raise ValueError(
                    f"{p} is a single object; pass --data-key to select the records "
                    f"list (available keys: {sorted(k for k, v in obj.items() if isinstance(v, list))})"
                )
            records = obj.get(key)
            if records is None:
                raise ValueError(f"no key {key!r} in {p}; available: {sorted(obj)}")
            return [r for r in records if isinstance(r, dict)]
        raise ValueError(f"{p} must contain a list of records or an object with a records list")

    raise ValueError(f"unsupported data-file extension {p.suffix!r}; use .jsonl or .json")


def require_compared_fields(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    allow_none: bool = True,
) -> None:
    """Warn (not fail) if the dataset lacks a compared column the benchmarks expect."""
    missing = [f for f in fields if not any(f in r for r in records)]
    if missing:
        import sys

        print(
            f"  [warning] the supplied dataset has no columns named {missing}; "
            f"those comparisons will degrade to null (no evidence) levels.",
            file=sys.stderr,
        )