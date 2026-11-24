"""Optional checks against the real prepared data (skipped unless $TT_DATA_DIR is set).

They pin semplan.enrich to the paper's artifacts: the Beauty inner-world enriched texts
must equal the archived i25 text dump byte-for-byte, and the C-site input tables rebuilt
from the enriched / plain catalogs must equal the files the paper's towers used.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from semplan import enrich as E

DATA = os.environ.get("TT_DATA_DIR")
pytestmark = pytest.mark.skipif(not DATA, reason="set TT_DATA_DIR to run real-data checks")


def _p(rel):
    p = Path(DATA) / rel
    if not p.exists():
        pytest.skip(f"{rel} not present")
    return p


def test_beauty_inner_enriched_texts_match_paper():
    meta = json.load(open(_p("amazon_meta/beauty_id2meta.json")))
    base = E.base_texts_from_meta(meta, 12101)
    texts, _ = E.enriched_texts(base, E.titles_from_texts(base),
                                E.pred_counts(_p("amazon_beauty_backbone/Beauty_inner.txt")))
    assert texts == json.load(open(_p("amazon_beauty_backbone/i25_texts_inner_pred3.json")))


@pytest.mark.parametrize("bb,prefix,seq,emb,inp", [
    ("amazon_beauty_backbone", "Beauty", "Beauty.txt", "i22real_embeddings", "i24real_input"),
    ("amazon_beauty_backbone", "Beauty", "Beauty.txt", "embeddings", "i44plain_input"),
    ("amazon_toys_backbone", "Toys", "Toys_and_Games.txt", "i22real_embeddings", "i24real_input"),
])
def test_centroid_tables_match_paper(bb, prefix, seq, emb, inp):
    torch = pytest.importorskip("torch")
    e = torch.load(_p(f"{bb}/{prefix}_openai-3-large_{emb}.pt"), map_location="cpu").float().numpy()
    out, _ = E.centroid_inputs(e, E.pred_counts(_p(f"{bb}/{seq}")))
    ref = torch.load(_p(f"{bb}/{prefix}_openai-3-large_{inp}.pt"), map_location="cpu").numpy()
    assert np.array_equal(out, ref)
