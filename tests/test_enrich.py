"""Predecessor mining (leak-safety), verbalization templates, centroid tokens, inner split."""
import numpy as np
import pytest

from semplan import enrich as E

# user 1: items 1 -> 2 -> 3 -> 4 (4 is the held-out final event)
# user 2: items 5 -> 2 -> 3      (3 is held out)
SEQ = ["1 1 2 3 4", "2 5 2 3"]


def test_mining_drops_user_id_and_final_event():
    pred = E.pred_counts(SEQ)
    # counted: user1 1->2, 2->3 ; user2 5->2 ; NOT 3->4 (final event), NOT 2->3 of user 2
    assert dict(pred[2]) == {1: 1, 5: 1}
    assert dict(pred[3]) == {2: 1}
    assert 4 not in pred, "held-out final event leaked into the predecessor counts"
    # the user id (first token) is never treated as an item
    assert all(p != 2 or i in (3,) for i, c in pred.items() for p in c if p == 2)


def test_mining_is_leak_safe_for_every_final_target():
    rng = np.random.default_rng(0)
    lines = []
    for u in range(50):
        items = rng.integers(1, 30, size=rng.integers(3, 9)).tolist()
        lines.append(" ".join(map(str, [u + 1] + items)))
    pred = E.pred_counts(lines)
    total = sum(sum(c.values()) for c in pred.values())
    expected = sum(len(l.split()) - 3 for l in lines)  # (n_items - 1) pairs after dropping last
    assert total == expected


def test_mining_counts_and_ranking():
    pred = E.pred_counts(["1 7 9 0", "2 7 9 0", "3 8 9 0", "4 6 9 0", "5 6 9 0", "6 6 9 0"])
    assert pred[9].most_common(3) == [(6, 3), (7, 2), (8, 1)]


META = {"1": "This item is called 'Red Lipstick'. It is priced at 9.0. ",
        "2": "This item is called 'Blue Mascara'. ",
        "3": "This item is called 'Nail File'. ",
        "4": "No title here."}


def test_titles_and_base_texts():
    base = E.base_texts_from_meta(META, 5)
    assert base[0].startswith("This item is called 'Red Lipstick'") and base[4] == ""
    assert E.titles_from_texts(base) == ["Red Lipstick", "Blue Mascara", "Nail File", "", ""]


def test_verbalization_template_exact():
    base = E.base_texts_from_meta(META, 5)
    titles = E.titles_from_texts(base)
    # item 3 preceded by 1 (x2) and 2 (x1); item 2 preceded by 4 (no title) only
    pred = E.pred_counts(["1 1 3 0", "2 1 3 0", "3 2 3 0", "4 4 2 0"])
    texts, n = E.enriched_texts(base, titles, pred)
    assert texts[2] == ("This item is called 'Nail File'.  Often bought after: "
                        "'Red Lipstick'; 'Blue Mascara'.")
    assert texts[1] == base[1], "predecessors without a title must not be verbalized"
    assert texts[4] == " ", "empty text becomes a single space (API requirement)"
    assert n == 1
    yelp, _ = E.enriched_texts(base, titles, pred, after_phrase=E.AFTER_VISITED)
    assert "Often visited after: 'Red Lipstick'; 'Blue Mascara'." in yelp[2]


def test_verbalization_keeps_top3_only():
    base = [f"This item is called 'T{i}'. " for i in range(1, 7)]
    titles = E.titles_from_texts(base)
    lines = [f"{u} {p} 6 0" for u, p in enumerate([1, 1, 1, 1, 2, 2, 2, 3, 3, 4], start=1)]
    texts, _ = E.enriched_texts(base, titles, E.pred_counts(lines))
    assert texts[5].endswith("Often bought after: 'T1'; 'T2'; 'T3'.")


def test_centroid_tokens_and_empty_fallback():
    emb = np.array([[3.0, 0, 0], [0, 2.0, 0], [0, 0, 5.0]], dtype=np.float32)
    pred = E.pred_counts(["1 1 3 0", "2 2 3 0"])   # item 3 <- {1, 2}; items 1, 2 have none
    out, n_aug = E.centroid_inputs(emb, pred)
    assert out.shape == (3, 6) and out.dtype == np.float32 and n_aug == 1
    own = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.testing.assert_allclose(out[:, :3], own, atol=1e-6)
    np.testing.assert_allclose(out[2, 3:], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], atol=1e-6)
    np.testing.assert_allclose(out[0, 3:], own[0], atol=1e-6)   # fallback = own embedding
    np.testing.assert_allclose(out[1, 3:], own[1], atol=1e-6)


def test_centroid_uses_l2_normalised_predecessors():
    emb = np.array([[10.0, 0], [0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out, _ = E.centroid_inputs(emb, E.pred_counts(["1 1 3 0", "2 2 3 0"]))
    # mean of unit vectors (1,0),(0,1) -> direction (1,1)/sqrt2, regardless of raw norms
    np.testing.assert_allclose(out[2, 2:], [2 ** -0.5, 2 ** -0.5], atol=1e-6)


def test_inner_split_drops_last_event():
    assert E.drop_last_event(["1 4 5 6 7", "2 8 9 10", ""]) == ["1 4 5 6", "2 8 9"]
