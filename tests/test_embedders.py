"""The offline fake embedder (used by smoke/quick runs) and the embedder switch."""
import numpy as np
import pytest

from semplan.embedders import embed_texts, fake_embed


def test_fake_embedder_deterministic_normalised():
    a = fake_embed(["red lipstick", "red lipstick", "blue tent"], dim=64).numpy()
    b = fake_embed(["red lipstick"], dim=64).numpy()
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert a[0] @ a[1] == pytest.approx(1.0, abs=1e-5)


def test_fake_embedder_shares_words():
    e = fake_embed(["red lipstick matte", "red lipstick gloss", "camping tent"], dim=256).numpy()
    assert e[0] @ e[1] > e[0] @ e[2] + 0.3


def test_embed_texts_switch_and_empty_text():
    e = embed_texts(["", "x"], embedder="fake", dim=32)
    assert tuple(e.shape) == (2, 32)
    with pytest.raises(ValueError):
        embed_texts(["x"], embedder="nope")


def test_openai_requires_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        embed_texts(["x"], embedder="openai")
