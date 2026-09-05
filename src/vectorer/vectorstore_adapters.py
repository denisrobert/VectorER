"""External distributed vector DB adapters.

The incremental pipeline's reference store is the ``VectorDatabase`` interface,
which this module adapts to an external, **distributed** vector database --
Qdrant by default.  This is the multi-node serving path for incremental
resolution (the expensive ANN index and the record payloads live in the cluster;
the embedding model and the FS scorer stay local).

``QdrantVectorDatabase`` implements the interface against a Qdrant collection:

* ``add(records)``     -- embed each record and upsert ``(vector, position,
  payload=record)`` into the collection.
* ``record_at(pos)``   -- fetch the stored payload by the position-as-id.
* ``index.search(q)``  -- HNSW cosine search over the collection's vectors,
  returning ``(positions, scores)``.
* ``update/delete``    -- upsert/replace by position id.
* ``__len__``          -- collection point count.

The adapter requires the ``qdrant-client`` package (optional dependency) and a
reachable Qdrant server.  It preserves cosine semantics by using Qdrant's
``Distance.COSINE``; the embedding model runs locally (only index + payloads go
remote).

Example::

    from qdrant_client import QdrantClient
    from vectorer.vectorstore_adapters import QdrantVectorDatabase

    qclient = QdrantClient(host="localhost", port=6333)
    db = QdrantVectorDatabase(embedder=embedder, client=qclient,
                              collection="people", distance=Distance.COSINE)
    db.add(reference_records)
    pipeline = IncrementalPipeline.from_store(db, scorer, k=20, tau=0.85)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .embeddings import EmbeddingModel
from .records import to_record_dict
from .vectorstores import IndexingStrategy, VectorDatabase


class _QdrantIndex(IndexingStrategy):
    """A thin IndexingStrategy view over a Qdrant collection.

    Positions are the framework's record index, stored as Qdrant point ids, so
    ``search`` returns (position, score) rows exactly like the local
    ``FlatIndex``.
    """

    def __init__(self, db: "QdrantVectorDatabase") -> None:
        self._db = db

    def search(self, query: Any, k: int) -> tuple[list[int], list[float]]:
        return self._db._search(query, k)

    def add(self, vectors: Sequence[Any]) -> None:
        raise NotImplementedError(
            "QdrantVectorDatabase.add(records) embeds + upserts; do not call "
            "index.add directly"
        )

    def clear(self) -> None:
        raise NotImplementedError(
            "QdrantVectorDatabase uses an external collection; drop the "
            "collection instead"
        )

    def __len__(self) -> int:
        return len(self._db)


class QdrantVectorDatabase(VectorDatabase[dict]):
    """``VectorDatabase`` backed by a distributed Qdrant collection.

    Parameters
    ----------
    embedder:
        Local embedding model (stays on this machine; only index + payloads
        go remote).
    client:
        A configured ``qdrant_client.QdrantClient``.
    collection:
        Collection name.  Created with the given ``distance``/``vector_size``
        if absent.
    vector_size:
        Embedding dimensionality (used to create the collection).
    distance:
        Qdrant ``Distance`` (default COSINE).  ``index.search`` returns cosine
        scores comparable with the local ``FlatIndex``.
    embed_text:
        Optional per-record serializer (defaults to ``field: value`` lines).
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        client: Any,
        collection: str,
        vector_size: Optional[int] = None,
        distance: Any = None,
        embed_text: Optional[Any] = None,
    ) -> None:
        self._embedder = embedder
        self._client = client
        self._collection = collection
        self._embed_text = embed_text or (
            lambda r: "\n".join(f"{k}: {v}" for k, v in r.items() if v is not None)
        )
        self._index = _QdrantIndex(self)

        from qdrant_client.http import models  # type: ignore

        from qdrant_client.http.models import Distance  # type: ignore[import-not-found]

        size = int(vector_size) if vector_size is not None else int(embedder.dimension or 0)
        if size <= 0:
            raise ValueError(
                "vector_size is required when the embedder's dimension is unknown"
            )
        self._distance = distance or Distance.COSINE
        _ensure_collection(self._client, self._collection, size, self._distance)

    # -- VectorDatabase -----------------------------------------------------

    @property
    def embedding(self) -> EmbeddingModel:
        return self._embedder

    @property
    def index(self) -> IndexingStrategy:
        return self._index

    def add(self, records: Sequence[dict]) -> None:
        """Embed the records and upsert them at positions ``add_offset..``.

        New positions are ``len(self)..len(self)+len(records)-1`` (the global
        record index), matching what ``record_at`` expects.
        """
        from qdrant_client.http import models

        start = len(self)
        points = []
        for i, record in enumerate(records):
            rec = to_record_dict(record)
            text = self._embed_text(rec)
            vec = list(self._embedder.embed(text))
            points.append(
                models.PointStruct(
                    id=start + i,
                    vector=vec,
                    payload={"record": rec},
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def update(self, records: Sequence[dict], positions: Sequence[int]) -> None:
        """Replace the payload/vector at each given position."""
        from qdrant_client.http import models

        points = []
        for pos, record in zip(positions, records):
            rec = to_record_dict(record)
            text = self._embed_text(rec)
            vec = list(self._embedder.embed(text))
            points.append(models.PointStruct(id=int(pos), vector=vec, payload={"record": rec}))
        self._client.upsert(collection_name=self._collection, points=points)

    def delete(self, positions: Sequence[int]) -> None:
        """Delete the points at ``positions``."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=[int(p) for p in positions],
        )

    def record_at(self, position: int) -> dict:
        res = self._client.retrieve(
            collection_name=self._collection, ids=[int(position)], with_payload=True
        )
        if not res or not getattr(res[0], "payload", None):
            raise IndexError(f"no record at position {position}")
        return dict(res[0].payload.get("record", {}))

    def _search(self, query: Any, k: int) -> tuple[list[int], list[float]]:
        # Qdrant >= 1.15 uses query_points; older clients use search.
        search = getattr(self._client, "query_points", None)
        if search is not None:
            res = search(
                collection_name=self._collection,
                query=query,
                limit=int(k),
                with_payload=False,
            )
        else:
            res = self._client.search(
                collection_name=self._collection,
                query_vector=query,
                limit=int(k),
                with_payload=False,
            )
        # Both forms return a list of hit objects with .id and .score.
        hits = getattr(res, "points", None) or res
        indices = [int(h.id) for h in hits]
        scores = [float(h.score) for h in hits]
        return indices, scores

    def __len__(self) -> int:
        return int(self._client.count(collection_name=self._collection).count)


def _ensure_collection(client, collection: str, size: int, distance) -> None:
    """Create/reuse the Qdrant collection (idempotent across clients/nodes)."""
    try:
        existing = client.get_collection(collection_name=collection)
        if existing is not None:
            return
    except Exception as exc:  # noqa: BLE001  (collection may not exist)
        _ = exc
    from qdrant_client.http import models

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=size, distance=distance),
    )