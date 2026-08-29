"""Vector store abstractions: indexing strategies and databases.

The *vector search blocking* stage needs three cooperating pieces:

* an :class:`IndexingStrategy` - the ANN/exact nearest-neighbour index over
  the stored vectors (:class:`FlatIndex` wraps a FAISS flat inner-product
  index over L2-normalized vectors, i.e. cosine similarity);
* an :class:`EmbeddingModel` (:mod:`vectorer.embeddings`) that produces the
  vectors; and
* a :class:`VectorDatabase` (e.g. :class:`InMemoryVectorDatabase`) that owns
  the index, the embedding model and the record payloads the index entries map
  back to.

Stores support incremental ``add/update/delete`` so the incremental pipeline
can grow the reference population without rebuilding the whole index.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from .embeddings import EmbeddingModel, Vector

T = TypeVar("T")


class IndexingStrategy:
    """Approximate or exact nearest-neighbour index over stored vectors."""

    def add(self, vectors: Sequence[Vector]) -> None:
        """Add vectors to the index (positional order is preserved)."""
        raise NotImplementedError

    def search(self, query: Vector, k: int) -> Tuple[list[int], list[float]]:
        """Return ``(indices, scores)`` of the ``k`` nearest vectors."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def save(self, path: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot be saved")

    @classmethod
    def load(cls, path: Any, **kwargs: Any) -> "IndexingStrategy":
        raise NotImplementedError(f"{cls.__name__} cannot be loaded")


class FlatIndex(IndexingStrategy):
    """Exact flat inner-product index over L2-normalized vectors (cosine).

    FAISS is imported lazily so the framework stays importable in
    environments where FAISS is not installed.
    """

    def __init__(self, normalize: bool = True) -> None:
        import faiss

        self._faiss = faiss
        self.normalize = normalize
        self._index: Optional[Any] = None

    def _ensure(self, dimension: int) -> None:
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(int(dimension))

    @staticmethod
    def _as_float32(vectors: Sequence[Vector]) -> np.ndarray:
        return np.asarray(list(vectors), dtype="float32")

    def add(self, vectors: Sequence[Vector]) -> None:
        array = self._as_float32(vectors)
        if array.size == 0:
            return
        self._ensure(int(array.shape[1]))
        if self.normalize:
            self._faiss.normalize_L2(array)
        self._index.add(array)  # type: ignore[union-attr]

    def search(self, query: Vector, k: int) -> Tuple[list[int], list[float]]:
        self._ensure(len(query))
        q = np.asarray([query], dtype="float32")
        if self.normalize:
            self._faiss.normalize_L2(q)
        kk = min(int(k), len(self))
        scores, indices = self._index.search(q, kk)  # type: ignore[union-attr]
        return list(indices[0]), list(scores[0])

    def clear(self) -> None:
        if self._index is not None:
            self._index.reset()  # type: ignore[union-attr]

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)  # type: ignore[union-attr]

    def save(self, path: Any) -> None:
        if self._index is None:
            raise RuntimeError("index is empty; nothing to save")
        self._faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Any, normalize: bool = True) -> "FlatIndex":
        import faiss

        instance = cls(normalize=normalize)
        instance._index = faiss.read_index(str(path))
        return instance

    def reconstruct(self, indices: Sequence[int]) -> np.ndarray:
        """Return the stored vectors at ``indices`` (used by canopy blocking)."""
        if self._index is None:
            raise RuntimeError("index is empty")
        n = len(self)
        result = np.vstack([self._index.reconstruct(int(i)) for i in indices])
        return np.asarray(result, dtype="float32")


class VectorDatabase(Generic[T]):
    """A store of reference records indexed by position.

    The database owns the embedding model, the indexing strategy, and the
    record payloads.  Positions returned by the index map back to records via
    :meth:`record_at`.
    """

    @property
    def embedding(self) -> EmbeddingModel:
        raise NotImplementedError

    @property
    def index(self) -> IndexingStrategy:
        raise NotImplementedError

    def add(self, records: Sequence[T]) -> None:
        raise NotImplementedError

    def update(self, records: Sequence[T], positions: Sequence[int]) -> None:
        raise NotImplementedError

    def delete(self, positions: Sequence[int]) -> None:
        raise NotImplementedError

    def record_at(self, position: int) -> T:
        raise NotImplementedError

    def records(self) -> list[T]:
        return [self.record_at(i) for i in range(len(self))]

    def __len__(self) -> int:
        raise NotImplementedError


class InMemoryVectorDatabase(VectorDatabase[T]):
    """In-memory vector database composing an embedding model and an index.

    Supports persistence (``save``/``load``) so a resolved reference
    population can be checkpointed without re-embedding on restart.
    Updating/deleting rebuilds the underlying index from the surviving
    records.
    """

    VECTOR_FILE = "index.faiss"
    RECORDS_FILE = "records.pkl"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        embedding: EmbeddingModel,
        index: Optional[IndexingStrategy] = None,
    ) -> None:
        self._embedding = embedding
        self._index = index or FlatIndex()
        self._records: list[T] = []

    @property
    def embedding(self) -> EmbeddingModel:
        return self._embedding

    @property
    def index(self) -> IndexingStrategy:
        return self._index

    def add(self, records: Sequence[T]) -> None:
        from .records import to_record_dict

        texts = [embed_text_of(to_record_dict(r)) for r in records]
        self._index.add(self._embedding.embed_many(texts))
        self._records.extend(records)

    def update(self, records: Sequence[T], positions: Sequence[int]) -> None:
        positions = [int(p) for p in positions]
        if len(records) != len(positions):
            raise ValueError("update requires one position per record")
        for record, position in zip(records, positions):
            if not 0 <= position < len(self._records):
                raise IndexError(f"position {position} out of range")
            self._records[position] = record
        self._reindex()

    def delete(self, positions: Sequence[int]) -> None:
        positions = sorted({int(p) for p in positions})
        if not positions:
            return
        if positions[0] < 0 or positions[-1] >= len(self._records):
            raise IndexError("position out of range")
        for position in reversed(positions):
            del self._records[position]
        self._reindex()

    def _reindex(self) -> None:
        self._index.clear()
        if self._records:
            self.add(self._records)

    def record_at(self, position: int) -> T:
        return self._records[position]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, directory: Any) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._index.save(directory / self.VECTOR_FILE)
        with (directory / self.RECORDS_FILE).open("wb") as handle:
            pickle.dump(self._records, handle)
        metadata = {
            "records": len(self._records),
            "index_class": type(self._index).__name__,
            "embedding_class": type(self._embedding).__name__,
            "dimension": getattr(self._embedding, "dimension", None),
        }
        (directory / self.METADATA_FILE).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Any,
        embedding: Optional[EmbeddingModel] = None,
        index: Optional[IndexingStrategy] = None,
    ) -> "InMemoryVectorDatabase[T]":
        directory = Path(directory)
        if index is None:
            index = FlatIndex.load(directory / cls.VECTOR_FILE)
        if embedding is None:
            raise ValueError(
                "an embedding model is required to rebuild an InMemoryVectorDatabase; "
                "pass the same embedding used to build the store"
            )
        with (directory / cls.RECORDS_FILE).open("rb") as handle:
            records = pickle.load(handle)
        database = cls(embedding, index)
        database._records = list(records)
        return database

    def vectors(self) -> np.ndarray:
        """Return the stored vectors (used by canopy blocking on the dataset)."""
        return self._index.reconstruct(list(range(len(self))))


def embed_text_of(record: dict) -> str:
    """Serialize a record dict for embedding (schema order, no id column)."""
    return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)