"""Embedding model abstractions and reference implementations.

The *embedding* stage turns the parsed record (serialized to text by
:mod:`vectorer.records`) into a dense vector.  The framework only depends on
the :class:`EmbeddingModel` interface, so the underlying model can be swapped
without touching the blocking or scoring stages.

* :class:`SentenceTransformerEmbedding` wraps a Hugging Face / torch
  sentence-transformers model (pinned to a revision for reproducibility).
* :class:`CharacterHashingEmbedding` is a deterministic, dependency-free
  embedding used by the test-suite and demos.  It hashes character n-grams of
  the input text into a fixed-width vector, which is fast and reproducible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

import numpy as np

Vector = Sequence[float]


class EmbeddingModel:
    """Interface implemented by every embedding model.

    Implementations must provide :meth:`embed` (single record) and
    :meth:`embed_many` (batched records).  :attr:`dimension` reports the fixed
    output dimensionality, defaulting to a lazy probe on first embed.
    """

    dimension: Optional[int] = None

    def embed(self, text: str) -> Vector:
        raise NotImplementedError

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dimension={self.dimension})"


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence-transformers embedding with a pinned model revision.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier (e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``).
    revision:
        Exact hub commit to load; keeps embedding output reproducible across
        runs.  ``None`` uses the hub default revision.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ...).
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device} if device else {}
        self._model = SentenceTransformer(model_id, revision=revision, **kwargs)
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_embedding_dimension
        self.dimension = int(get_dim())
        self.model_id = model_id
        self.revision = revision

    def embed(self, text: str) -> Vector:
        return [float(x) for x in self._model.encode([text])[0]]

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        encoded = self._model.encode(list(texts), batch_size=64)
        return [[float(x) for x in row] for row in encoded]


class CharacterHashingEmbedding(EmbeddingModel):
    """Deterministic hashed character n-gram embedding (no external deps).

    Emits a fixed-width, normalized vector whose entries are the counts of
    character 2/3-grams hashed into contiguous bins (a ``text -> vector``
    "feature hashing" scheme).  Spelling-similar texts share many n-grams, so
    cosine similarity under :class:`vectorer.vectorstores.FlatIndex` behaves
    like a cheap fuzzy-similarity blocker -- sufficient for tests and demos.

    Parameters
    ----------
    dimension:
        Output dimensionality (default 256).
    ngrams:
        Size of character n-grams fed to the hasher (default ``(2, 3)``).
    """

    _TOKENS = re.compile(r"[a-z0-9]+")

    def __init__(
        self,
        dimension: int = 256,
        ngrams: tuple[int, ...] = (2, 3),
    ) -> None:
        self.dimension = int(dimension)
        self.ngrams = tuple(int(n) for n in ngrams)

    def _digest(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float64")
        lowered = text.lower()
        tokens = self._TOKENS.findall(lowered) or [lowered]
        for token in tokens:
            for n in self.ngrams:
                if len(token) < n:
                    continue
                for i in range(len(token) - n + 1):
                    gram = token[i : i + n]
                    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    slot = int.from_bytes(digest, "little") % self.dimension
                    vector[slot] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, text: str) -> Vector:
        return self._digest(text).tolist()

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        return [self._digest(text).tolist() for text in texts]