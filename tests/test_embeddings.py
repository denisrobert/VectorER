"""Tests for the embedding model wrappers."""

import numpy as np

from vectorer.embeddings import CharacterHashingEmbedding, SentenceTransformerEmbedding


class _StubTransformersModel:
    """Duck-typed stand-in for a preconfigured sentence-transformers model."""

    def __init__(self, dimension=8, model_id="stub/model"):
        self.dimension = dimension
        self.model_id = model_id

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, batch_size=None):
        out = []
        for text in texts:
            rng = np.zeros(self.dimension)
            for i, ch in enumerate(text):
                rng[i % self.dimension] += ord(ch)
            norm = float(np.linalg.norm(rng))
            out.append((rng / norm if norm else rng).tolist())
        return np.asarray(out)


def test_sentence_transformer_embedding_wraps_preconfigured_model():
    model = _StubTransformersModel(dimension=8)
    embedder = SentenceTransformerEmbedding(model=model)

    assert embedder.dimension == 8
    assert embedder.model_id == "stub/model"
    vector = embedder.embed("hello")
    assert len(vector) == 8
    vectors = embedder.embed_many(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


def test_preconfigured_model_is_used_verbatim():
    model = _StubTransformersModel(dimension=4)
    embedder = SentenceTransformerEmbedding(
        model_id="ignored/id", revision="ignored", device="cuda:0", model=model
    )
    assert embedder._model is model
    assert embedder.dimension == 4


def test_model_id_falls_back_when_preconfigured_model_has_no_id():
    class AnonModel(_StubTransformersModel):
        def __init__(self):
            super().__init__(dimension=6)
            del self.model_id

    embedder = SentenceTransformerEmbedding(model=AnonModel())
    assert embedder.model_id  # default id kept, normalized metadata stays present


def test_dimension_may_be_unknown_for_external_models():
    class NoDimension(_StubTransformersModel):
        def get_sentence_embedding_dimension(self):
            raise AttributeError

    embedder = SentenceTransformerEmbedding(model=NoDimension())
    assert embedder.dimension is None
    # FlatIndex/vector DB derive the dimension lazily from embedded vectors.
    assert len(embedder.embed("x")) == 8


def test_character_hashing_is_deterministic():
    embedder = CharacterHashingEmbedding(dimension=32)
    assert embedder.embed("alpha") == embedder.embed("alpha")
    assert embedder.embed("alpha") != embedder.embed("bravo")
    assert len(embedder.embed_many(["x", "y"])) == 2