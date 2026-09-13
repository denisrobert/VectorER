"""Shared primitives for the distributed executors."""


def hash_pair(i: int, j: int, n_workers: int) -> int:
    """Deterministic owner slot for an unordered (i, j) record pair."""
    a, b = (i, j) if i <= j else (j, i)
    return hash((a, b)) % int(n_workers)