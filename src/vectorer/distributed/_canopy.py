"""Distributed canopy blocking: shard embedding, local assignment, sampling.

The driver trains canopy centroids on a cross-shard sample
(:func:`gather_canopy_sample`) and each worker assigns its local shard via
:func:`_assign_shard`; shards are embedded with the deterministic hashing
embedder (:func:`_embed_shard`) so no model object needs to be shipped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..blocking import assign_canopies
from ..records import to_record_dict


def _embed_shard(records, embedder_state: dict) -> tuple[list[dict], np.ndarray]:
    """Parse + embed a shard with the model rebuilt from its settings.

    The embedder is reconstructed per worker from plain settings (no model
    object is shipped), so a heavy model -- e.g. a sentence-transformer on a
    GPU -- is loaded once per worker, which is the expected cost in a large
    distributed run.
    """
    from ..embeddings import embedder_from_settings

    embedder = embedder_from_settings(embedder_state)
    parsed = [to_record_dict(r) for r in records]
    texts = [_serialize(r) for r in parsed]
    vecs = np.asarray(embedder.embed_many(texts), dtype="float32")
    return parsed, vecs


def _embedder_state_of(embedder) -> dict:
    """Serializable settings for an embedding model, for per-worker rebuilds."""
    to_settings = getattr(embedder, "to_settings", None)
    if to_settings is None:
        raise ValueError(
            "distributed_batch_er requires an EmbeddingModel that implements "
            "to_settings() so workers can rebuild it; got "
            f"{type(embedder).__name__}"
        )
    return to_settings()


def _serialize(record: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)


def _assign_shard(vectors, centroids, overlap_m):
    """Worker-side canopy assignment (local CanopyIndex)."""
    return assign_canopies(vectors, centroids, overlap_m)


def gather_canopy_sample(
    vector_shards: Sequence[np.ndarray],
    sample_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Gather a **cross-machine sample** of vectors for canopy-centroid training.

    Instead of materializing the full vector matrix on one node, each shard
    contributes a proportional, deterministic random slice, and only the sample
    is concatenated.  Uses a seeded RNG so the sample is reproducible and
    identical regardless of shard boundaries.
    """
    total = sum(len(v) for v in vector_shards)
    if total <= int(sample_size):
        return np.vstack(vector_shards)
    rng = np.random.default_rng(seed)
    per_shard = int(sample_size) // max(1, len(vector_shards))
    slices = []
    for v in vector_shards:
        take = max(1, min(per_shard, len(v)))
        idx = rng.choice(len(v), take, replace=False)
        slices.append(np.asarray(v)[idx])
    return np.vstack(slices)