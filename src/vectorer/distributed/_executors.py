"""Executor backends for the distributed stage map.

``process`` (`ProcessPoolExecutor`; default, extra-free), ``thread``
(`ThreadPoolExecutor`), and ``ray`` (:class:`RayExecutor`, requires the ``ray``
package).  Worker callables are module-level functions so processes and Ray
can pickle them by reference.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional


class RayExecutor:
    """A minimal Ray-backed executor exposing ``map(fn, items)``.

    Optional: requires the ``ray`` package (installed separately).  The worker
    callable must be a module-level (picklable) function, not a closure --
    closures are only safe on the thread/process executors.
    """

    def __init__(self, n_workers: int = 2, address: Optional[str] = None) -> None:
        import ray
        import os

        if not ray.is_initialized():
            env_addr = os.environ.get("RAY_ADDRESS")
            if address in (None, "auto") and env_addr:
                ray.init(address=env_addr)
            elif address in (None, "auto"):
                # No cluster specified and no RAY_ADDRESS: start a **local** one
                # (no GCS), so `--ray-address auto` works as a one-host demo of
                # the same code path; a real multi-node cluster is joined by
                # passing the head's ip:port (or setting RAY_ADDRESS).
                ray.init(num_cpus=int(n_workers))
            else:
                ray.init(address=address, num_cpus=int(n_workers))
        self.n_workers = int(n_workers)

    def map(self, fn, items):
        import ray

        @ray.remote
        def _call(f, item):  # noqa: E306  (remote fn must be module-scoped)
            return f(item)

        futures = [_call.remote(fn, item) for item in items]
        return list(ray.get(futures))


def create_executor(
    kind: str = "process",
    n_workers: int = 2,
    address: Optional[str] = None,
):
    """Create an executor backend by name.

    ``kind`` in {"process", "thread", "ray"}:
    * "process" -- ``ProcessPoolExecutor`` (default; extra-free).
    * "thread"  -- ``ThreadPoolExecutor``.
    * "ray"     -- :class:`RayExecutor` (requires the ``ray`` package).
    """
    if kind == "process":
        return ProcessPoolExecutor(max_workers=int(n_workers))
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=int(n_workers))
    if kind == "ray":
        return RayExecutor(n_workers=n_workers, address=address)
    raise ValueError(f"unknown executor kind {kind!r}; use process, thread, or ray")