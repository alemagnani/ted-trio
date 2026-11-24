"""LOO split construction, HR@k / NDCG@k, masked vs unmasked evaluation."""
import math

import pytest

from semplan.enrich import drop_last_event
from semplan.eval import protocol


def write(tmp_path, lines, name="seq.txt"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_split_construction(tmp_path):
    split = protocol.build(write(tmp_path, ["1 1 2 3 4 5", "2 6 7 8"]), n_items=10)
    assert split.test == [([0, 1, 2, 3], 4), ([5, 6], 7)]          # rows are id - 1
    assert split.val == [([0, 1, 2], 3), ([5], 6)]
    assert ([0], 1) in split.train_pairs and all(t not in (3, 4) for _, t in split.train_pairs[:3])
    assert split.train_item_rows == {0, 1, 2, 5}
    assert split.cold_test_idx == {0, 1}     # targets 5 and 8 never in a training history


def test_inner_world_pseudo_test_is_original_minus_two(tmp_path):
    lines = ["1 1 2 3 4 5", "2 6 7 8 9"]
    real = protocol.build(write(tmp_path, lines, "real.txt"), n_items=10)
    inner = protocol.build(write(tmp_path, drop_last_event(lines), "inner.txt"), n_items=10)
    assert [t for _, t in inner.test] == [t for _, t in real.val]  # pseudo-test = event -2
    real_targets = [t for _, t in real.test]
    assert all(t not in h for (h, _), t in zip(inner.test, real_targets)), \
        "the real test event must not appear anywhere in the inner world"


def test_max_len_truncation(tmp_path):
    split = protocol.build(write(tmp_path, ["1 " + " ".join(str(i) for i in range(1, 31))]),
                           n_items=40, max_len=20)
    hist, tgt = split.test[0]
    assert len(hist) == 19 and tgt == 29


@pytest.mark.parametrize("rank,hr,ndcg", [(0, 1.0, 1.0), (1, 1.0, 1 / math.log2(3)),
                                          (4, 1.0, 1 / math.log2(6)), (5, 0.0, 0.0)])
def test_dcg_hit_hand_computed(rank, hr, ndcg):
    ranked = [9, 8, 7, 6, 5, 4, 3]
    ranked.insert(rank, 42)
    h, g = protocol.dcg_hit(ranked, 42, 5)
    assert h == hr and g == pytest.approx(ndcg)


def test_evaluate_overall_seen_cold(tmp_path):
    split = protocol.build(write(tmp_path, ["1 1 2 3", "2 1 2 4", "3 2 1 3"]), n_items=6)
    # targets rows: 2, 3, 2 ; training rows {0, 1} -> all targets cold except none
    lists = {0: [2, 0, 1, 3, 4], 1: [0, 1, 3, 2, 4], 2: [5, 4, 0, 1, 3]}
    it = iter(range(3))
    res = protocol.evaluate(lambda h, e: lists[next(it)], split, ks=(1, 5))
    assert res["overall"]["HR@1"] == round(1 / 3, 4)
    assert res["overall"]["HR@5"] == round(2 / 3, 4)
    ndcg5 = (1.0 + 1 / math.log2(4) + 0.0) / 3
    assert res["overall"]["NDCG@5"] == round(ndcg5, 4)
    assert res["cold"]["n"] + res["seen"]["n"] == 3


def test_masked_convention_and_repeat_targets(tmp_path):
    # user 1's target (item 2) also appears in its history -> a repeat target
    split = protocol.build(write(tmp_path, ["1 2 1 2", "2 3 4 5"]), n_items=6)
    full = [1, 0, 2, 3, 4, 5]                    # target row 1 ranked first

    def rank_fn(hist, excl):
        return [i for i in full if i not in excl]

    unmasked = protocol.evaluate(rank_fn, split, ks=(1,), exclude_history=False)
    masked = protocol.evaluate(rank_fn, split, ks=(1,), exclude_history=True)
    assert unmasked["overall"]["HR@1"] == 0.5   # user 1 hits, user 2 misses
    # masking removes the user's own history, which here DELETES the repeat target
    assert masked["overall"]["HR@1"] == 0.0


def test_masked_guard_rejects_leaking_rank_fn(tmp_path):
    split = protocol.build(write(tmp_path, ["1 1 2 3"]), n_items=4)
    with pytest.raises(AssertionError):
        protocol.evaluate(lambda h, e: [0, 1, 2, 3], split, ks=(2,), exclude_history=True)
