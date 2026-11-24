"""Sign-flip test, bootstrap, per-query z fusion, training-free arm, transition heuristic."""
import numpy as np
import pytest
import torch

from semplan import spaces
from semplan.eval import protocol, significance


def test_signflip_p_floor_is_one_over_nperm_plus_one():
    a = np.ones(500)
    b = np.zeros(500)
    r = significance.paired_randomization(a, b, n_perm=999, seed=0)
    assert r["p_two_sided"] == pytest.approx(1 / 1000)
    assert r["mean_delta"] == 1.0


def test_signflip_null_is_not_significant_and_deterministic():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 400).astype(float)
    b = a[rng.permutation(400)]
    r1 = significance.paired_randomization(a, b, n_perm=2000, seed=0)
    r2 = significance.paired_randomization(a, b, n_perm=2000, seed=0)
    assert r1 == r2 and r1["p_two_sided"] > 0.05


def test_identical_systems_give_p_one():
    a = np.array([1.0, 0, 1, 0, 1])
    assert significance.paired_randomization(a, a, n_perm=100)["p_two_sided"] == 1.0


def test_bootstrap_ci_contains_delta():
    rng = np.random.default_rng(2)
    a = rng.random(300)
    b = a - 0.1 + 0.01 * rng.standard_normal(300)
    r = significance.paired_bootstrap(a, b, n_boot=500, seed=0)
    lo, hi = r["ci95"]
    assert lo < r["mean_delta"] < hi and lo < 0.1 < hi and hi - lo < 0.01
    assert r["p_gt_zero"] == 1.0


def test_factorial_effects_contrasts():
    import i44_factorial_inference as F
    n = 200
    rng = np.random.default_rng(0)
    pp = rng.integers(0, 2, n).astype(float)
    cells = {"pp": pp, "Ep": pp.copy(), "pC": pp.copy(), "EC": pp.copy()}
    cells["Ep"][:20] = 1 - cells["Ep"][:20]
    eff = F.effects_from_cells(cells)
    want_E = float(((cells["Ep"] + cells["EC"]) - (cells["pp"] + cells["pC"])).mean() / 2)
    assert eff["E_main"]["estimate"] == pytest.approx(want_E)
    assert eff["C_main"]["estimate"] == pytest.approx(
        float(((cells["pC"] + cells["EC"]) - (cells["pp"] + cells["Ep"])).mean() / 2))
    assert eff["interaction"]["estimate"] == pytest.approx(
        float((cells["EC"] - cells["Ep"] - cells["pC"] + cells["pp"]).mean()))
    assert 0 < eff["E_main"]["p_signflip"] <= 1


def test_per_query_zscore_trio_fusion():
    import i41_common as C
    torch.manual_seed(0)
    items = torch.randn(30, 8)
    qs = {f"s{s}": [torch.randn(4, 8) * (s + 1)] for s in range(3)}   # different scales
    fused = C.trio_z(qs, 0, items)
    per = [C.zscore(torch.einsum("kd,nd->kn", qs[n][0], items).max(dim=0).values.numpy())
           for n in qs]
    np.testing.assert_allclose(fused, np.mean(per, axis=0), rtol=1e-5, atol=1e-6)
    for z in per:   # each seed contributes a unit-variance, zero-mean score vector
        assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-4
    # scaling one seed's scores leaves the fusion unchanged (scale invariance)
    qs2 = dict(qs); qs2["s0"] = [qs["s0"][0] * 100]
    np.testing.assert_allclose(C.trio_z(qs2, 0, items), fused, rtol=1e-4, atol=1e-5)


def test_spaces_zscore_constant_vector():
    np.testing.assert_allclose(spaces.zscore(np.ones(5)), np.zeros(5))


def test_training_free_arm_scoring():
    rng = np.random.default_rng(0)
    n, d = 40, 6
    content = rng.standard_normal((n, d)).astype(np.float32)
    seqs = [list(rng.integers(0, 20, 8)) for _ in range(60)]   # items 20..39 never co-occur
    collab, covered = spaces.build_collab_embeddings(seqs, n, d=4, window=5)
    assert collab.shape == (n, 4) and covered[:20].any() and not covered[20:].any()
    cw, lw = spaces.whiten_blend_spaces(content, collab, covered)
    assert np.allclose(lw[~covered], 0.0)
    hist = [0, 3, 5]
    s = spaces.blended_scores_whitened(cw, lw, covered, hist)
    zc = spaces.zscore(cw @ spaces.user_vec(cw, hist, spaces.WHITEN_RECENCY))
    zl = spaces.zscore(lw @ spaces.user_vec(lw, hist, spaces.WHITEN_RECENCY))
    # covered items: content z + collab z ; uncovered: content z counted twice (fallback)
    np.testing.assert_allclose(s[covered], (zc + zl)[covered], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s[~covered], 2 * zc[~covered], rtol=1e-5, atol=1e-5)
    # a history with no collab signal scores content-only (x2)
    s_cold = spaces.blended_scores_whitened(cw, lw, covered, [25, 30])
    zc2 = spaces.zscore(cw @ spaces.user_vec(cw, [25, 30], spaces.WHITEN_RECENCY))
    np.testing.assert_allclose(s_cold, 2 * zc2, rtol=1e-5, atol=1e-5)


def test_user_vec_recency_weighting():
    space = np.eye(3, dtype=np.float32)
    v = spaces.user_vec(space, [0, 1, 2], recency=0.5)
    np.testing.assert_allclose(v, np.array([0.25, 0.5, 1.0]) / np.linalg.norm([0.25, 0.5, 1]),
                               rtol=1e-6)


def test_transition_count_heuristic(tmp_path):
    import i42_transition_only as H
    p = tmp_path / "seq.txt"
    # rows: 0->1 three times, 0->2 once; user 4's history ends with item row 0
    p.write_text("1 1 2 9\n2 1 2 9\n3 1 2 9\n4 1 3 9\n5 4 1 9\n")
    split = protocol.build(str(p), n_items=20)
    C, pop = H.mine(split)
    assert C[0, 1] == 3 and C[0, 2] == 1
    assert C[0, 8] == 0, "the held-out final event must not be mined"
    rank = H.make_rank_fn(C, pop, gamma=0.0)
    assert rank([3, 0], set())[:2] == [1, 2]
    assert 1 not in rank([3, 0], {1})[:5]                  # masked convention honoured
    rank_d = H.make_rank_fn(C, pop, gamma=0.5)
    assert rank_d([0], set())[0] == 1
