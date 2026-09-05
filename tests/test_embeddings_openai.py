"""Tests for the OpenAI API-backed embedder (mock HTTP, no network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vectorer.embeddings import OpenAIEmbedding


class _Handler(BaseHTTPRequestHandler):
    """Serves a fake /embeddings endpoint whose vector depends on the text."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = {
            "data": [
                {
                    "embedding": [_ord_sum(t) + k for k in range(1, 4)],
                    "index": i,
                }
                for i, t in enumerate(inputs)
            ],
            "model": body.get("model", "fake"),
        }
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


def _ord_sum(text: str) -> float:
    return float(sum(ord(c) for c in text))


@pytest.fixture
def fake_api():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_openai_embedding_dimension_and_embed(fake_api):
    emb = OpenAIEmbedding(api_key="test-key", model="fake-model", base_url=fake_api)
    assert emb.dimension == 3
    vec = emb.embed("hello")
    assert len(vec) == 3
    s = sum(ord(c) for c in "hello")
    assert vec == pytest.approx([s + 1, s + 2, s + 3])


def test_openai_embedding_many_aligns(fake_api):
    emb = OpenAIEmbedding(api_key="test-key", model="fake-model", base_url=fake_api,
                          batch_size=2)
    vecs = emb.embed_many(["a", "b", "c"])
    assert len(vecs) == 3
    for text, got in zip(["a", "b", "c"], vecs):
        s = sum(ord(c) for c in text)
        assert got == pytest.approx([s + 1, s + 2, s + 3])


def test_openai_requires_key():
    import os

    old = os.environ.get("OPENAI_API_KEY")
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="api key"):
            OpenAIEmbedding()
    finally:
        if old:
            os.environ["OPENAI_API_KEY"] = old


def test_openai_prefers_sdk_client_when_available(fake_api):
    """When the ``openai`` package is installed, the SDK client is used."""
    from vectorer.embeddings import _make_openai_client

    emb = OpenAIEmbedding(api_key="test-key", model="fake-model", base_url=fake_api)
    try:
        import openai  # noqa: F401

        # SDK present: the client object should be an OpenAI instance (or at
        # least truthy), not the urllib fallback path.
        client = _make_openai_client("k", fake_api, 1.0)
        assert client is not None
        assert emb._client is not None
    except ImportError:
        # openai not installed: fall back to urllib.
        assert emb._client is None
    # The mock still returns the ord-based vector regardless of backend.
    s = sum(ord(c) for c in "zz")
    assert emb.embed("zz") == pytest.approx([s + 1, s + 2, s + 3])