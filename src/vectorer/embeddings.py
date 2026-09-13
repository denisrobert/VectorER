"""Embedding model abstractions and reference implementations.

The *embedding* stage turns the parsed record (serialized to text by
:mod:`vectorer.records`) into a dense vector.  The framework only depends on
the :class:`EmbeddingModel` interface, so the underlying model can be swapped
without touching the blocking or scoring stages.

* :class:`SentenceTransformerEmbedding` wraps a Hugging Face / torch
  sentence-transformers model (pinned to a revision for reproducibility).
* :class:`OpenAIEmbedding` calls the OpenAI Embeddings API
  (``text-embedding-3-*``, ``text-embedding-ada-002``, etc.).
* :class:`CharacterHashingEmbedding` is a deterministic, dependency-free
  embedding used by the test-suite and demos.  It hashes character n-grams of
  the input text into a fixed-width vector, which is fast and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

import numpy as np

Vector = Sequence[float]

# Number of tokens sent per API request when batching.
_OPENAI_BATCH_SIZE = 16

# The default vendor endpoint.  Custom ``base_url`` values (e.g. a local
# OpenAI-compatible server such as Ollama / LM Studio / vLLM) may be queried
# without an API key.
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


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

    def to_settings(self) -> dict:
        """Serializable constructor settings for per-worker re-instantiation.

        Distributed batch ER rebuilds the embedding model inside each worker
        (a heavy model is loaded once per worker, e.g. onto a GPU), so
        concrete models must be reconstructible from plain constructor
        settings via :func:`embedder_from_settings`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_settings(); it cannot "
            "be re-instantiated in distributed workers"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dimension={self.dimension})"


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence-transformers embedding — load by identifier, or wrap a model
    you already instantiated.

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
    backend:
        Compute backend passed through to ``SentenceTransformer``
        (``"torch"``, ``"onnx"``, ``"openvino"``, ...) when supported by the
        installed sentence-transformers; ``None`` uses its default.  Kept in
        :meth:`to_settings` so distributed workers re-load the model with the
        same backend (e.g. ONNX/OpenVINO servers on plain CPUs).
    model:
        A **preconfigured** embedding model instead of loading one from
        ``model_id``/``revision``/``device``.  Any object with an ``encode``
        method (e.g. an already-instantiated ``SentenceTransformer``) is
        wrapped as-is into the :class:`EmbeddingModel` interface; pass the
        throughput/batch/device settings it was configured with and the
        framework does not touch them.  Use this to reuse a model already
        loaded on a GPU, with a different ``device_map``/quantization/pooling
        or other features that cannot be expressed as ``model_id`` +
        ``device``.  When given, ``model_id``/``revision``/``device`` are
        ignored.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        if model is not None:
            self._model = model
            self.model_id = getattr(model, "model_id", None) or model_id
            self.revision = revision
        else:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if device is not None:
                kwargs["device"] = device
            if backend is not None:
                kwargs["backend"] = backend
            self._model = SentenceTransformer(model_id, revision=revision, **kwargs)
            self.model_id = model_id
            self.revision = revision
        self.device = device
        self.backend = backend
        self._wrapped = model is not None
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        if get_dim is None:
            get_dim = getattr(self._model, "get_embedding_dimension", None)
        try:
            self.dimension = int(get_dim()) if get_dim is not None else None
        except Exception:
            self.dimension = None

    def to_settings(self) -> dict:
        if self._wrapped:
            raise ValueError(
                "SentenceTransformerEmbedding was given a pre-loaded model object "
                "and cannot be re-instantiated in workers; load it from "
                "model_id/device/backend (e.g. model_id='sentence-transformers/"
                "all-MiniLM-L6-v2', device='cuda') so each worker can rebuild it"
            )
        return {
            "type": "sentence_transformer",
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "backend": self.backend,
        }

    def embed(self, text: str) -> Vector:
        return [float(x) for x in self._model.encode([text])[0]]

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        encoded = self._model.encode(list(texts), batch_size=64)
        return [[float(x) for x in row] for row in encoded]


class OpenAIEmbedding(EmbeddingModel):
    """Embedder backed by the OpenAI Embeddings API.

    Uses the ``openai`` Python package when installed (the preferred client),
    otherwise falls back to a minimal ``urllib`` client so the framework stays
    usable without extra dependencies.  The API key is read from the
    ``OPENAI_API_KEY`` environment variable (or passed explicitly); local
    OpenAI-compatible servers can be used without any key (pass ``base_url=``
    and leave ``api_key=None``).

    Parameters
    ----------
    api_key:
        OpenAI API key.  Defaults to the ``OPENAI_API_KEY`` environment
        variable.  May be ``None`` (with no env var set) **when ``base_url``
        points at a local OpenAI-compatible server** (Ollama, LM Studio, vLLM,
        local proxies) that does not require authentication -- the request is
        then sent without an ``Authorization`` header.  Keyless access to the
        default ``api.openai.com`` endpoint is still rejected.
    model:
        Embedding model id, e.g. ``"text-embedding-3-large"``,
        ``"text-embedding-3-small"``, or ``"text-embedding-ada-002"``.
    dimensions:
        Optional truncation for ``text-embedding-3-*`` models (the API supports
        trimming the output dimension).  ``None`` keeps the model's native
        dimensionality.
    base_url:
        Optional override for the API endpoint (e.g. a local OpenAI-compatible
        server such as ``http://localhost:11434/v1`` for Ollama).  Defaults to
        OpenAI's standard endpoint.
    batch_size:
        Number of texts sent per API request.
    timeout:
        Request timeout in seconds (used by both client backends).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        batch_size: int = _OPENAI_BATCH_SIZE,
        timeout: float = 60.0,
    ) -> None:
        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or _DEFAULT_OPENAI_BASE_URL).rstrip("/")
        # Keyless access is permitted only to custom (local) endpoints: the
        # vendor endpoint always requires a key.
        if not self._key and self.base_url == _DEFAULT_OPENAI_BASE_URL:
            raise ValueError(
                "OpenAIEmbedding requires an api key: pass api_key= or set the "
                "OPENAI_API_KEY environment variable (a key is required for the "
                "default api.openai.com endpoint); custom base_url endpoints may "
                "be used without a key"
            )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        # Prefer the official SDK when present; fall back to urllib.
        self._client = _make_openai_client(self._key, self.base_url, self.timeout)
        # Probe the response model to discover the embedding dimension.
        sample = self._call_api(["probe"])
        self.dimension = len(sample[0]["embedding"])

    def to_settings(self) -> dict:
        return {
            "type": "openai",
            "api_key": self._key,
            "model": self.model,
            "dimensions": self.dimensions,
            "base_url": self.base_url,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
        }

    # -- API ---------------------------------------------------------------

    def _call_api(self, inputs: list[str]) -> list[dict]:
        kwargs = {"model": self.model, "input": inputs}
        if self.dimensions is not None:
            kwargs["dimensions"] = int(self.dimensions)
        if self._client is not None:
            try:
                resp = self._client.embeddings.create(**kwargs)
            except Exception as exc:  # noqa: BLE001  (surface the SDK error)
                raise RuntimeError(f"OpenAI embeddings API error: {exc}") from exc
            return [
                {
                    "embedding": item.embedding,
                    "index": item.index,
                }
                for item in resp.data
            ]
        return self._call_api_urllib(inputs)

    def _call_api_urllib(self, inputs: list[str]) -> list[dict]:
        body = {"model": self.model, "input": inputs}
        if self.dimensions is not None:
            body["dimensions"] = int(self.dimensions)
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"OpenAI embeddings API error {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings API unreachable: {exc.reason}") from exc
        return payload.get("data", [])

    # -- EmbeddingModel -------------------------------------------------------

    def embed(self, text: str) -> Vector:
        return [_as_float(v) for v in self._call_api([text])[0]["embedding"]]

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        texts = list(texts)
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            for item in self._call_api(chunk):
                out.append([_as_float(v) for v in item["embedding"]])
        # The API preserves input order, but guard alignment explicitly.
        assert len(out) == len(texts)
        return out


def _make_openai_client(
    api_key: Optional[str], base_url: str, timeout: float
) -> Optional[Any]:
    """Return an OpenAI SDK client when ``openai`` is installed, else ``None``.

    ``None`` signals the urllib fallback path.  When ``api_key`` is ``None``
    (keyless local server) a placeholder is passed so the SDK constructor does
    not raise; the local server ignores it.
    """
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key if api_key else "sk-local-keyless",
            base_url=base_url,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001  (no openai package, or SDK init fails)
        return None


def _as_float(value) -> float:
    return float(value)


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

    def to_settings(self) -> dict:
        return {
            "type": "character_hashing",
            "dimension": self.dimension,
            "ngrams": list(self.ngrams),
        }

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


def embedder_from_settings(state: dict) -> EmbeddingModel:
    """Rebuild an :class:`EmbeddingModel` from :meth:`EmbeddingModel.to_settings`.

    Used by the distributed executors so every worker can re-load the model
    (e.g. a sentence-transformer onto a GPU) without shipping a model object.
    """
    kind = state.get("type")
    if kind == "character_hashing":
        return CharacterHashingEmbedding(
            dimension=state["dimension"], ngrams=tuple(state["ngrams"])
        )
    if kind == "sentence_transformer":
        return SentenceTransformerEmbedding(
            model_id=state["model_id"],
            revision=state.get("revision"),
            device=state.get("device"),
            backend=state.get("backend"),
        )
    if kind == "openai":
        return OpenAIEmbedding(
            api_key=state.get("api_key"),
            model=state["model"],
            dimensions=state.get("dimensions"),
            base_url=state.get("base_url"),
            batch_size=state.get("batch_size", _OPENAI_BATCH_SIZE),
            timeout=state.get("timeout", 60.0),
        )
    raise ValueError(f"unknown embedding model settings: {state!r}")