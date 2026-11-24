"""CLOSE-2 — score TIGER beam dumps (j1_dump.py BeamRecord JSONL) under OUR protocol.

Converts a `{ds}_tiger_beams_test.jsonl` dump (produced on the GPU box by
scripts/j1/j1_dump.py or its Toys/Sports adaptation — see close2_tiger_runner.md)
into protocol metrics (HR/NDCG @5/10, overall + seen/cold) and a per-user npz
aligned to `protocol.build(...).test` order, so paired sign-flip tests against
our trio per-user arrays (i24_test_per_user.npz etc.) need no re-alignment.

Alignment: the LIGER eval order is the SEEN test users first, then the UNSEEN
(cold) test users, each in original user order — verified exactly against
outputs/artifacts/j1/liger_alignment_test.json for Beauty (offset=1). The same
mapping is derived here from the protocol split itself; if --alignment is given,
it is cross-checked instead of trusted.

Ranking semantics: within a user, beams are deduped in beam order over VALID
mapped items only (invalid semantic ids produce no candidate — pure generative
cannot back off), and the target's rank is its position in that deduped list.
This reproduces J1's TIGER Beauty numbers (HR@5 0.0453 / HR@10 0.0709 /
NDCG@10 0.0383, cold all 0; NDCG@5 0.0301 vs J1's undeduped 0.0300 — dedup
promotes the target by one slot for a handful of users; dedup is the correct
ranked-list semantics and is what we report).

Usage:
  python scripts/close2_tiger_score.py --dataset beauty \
      --beams outputs/artifacts/j1/dumps/beauty_tiger_beams_test.jsonl \
      --alignment outputs/artifacts/j1/liger_alignment_test.json \
      --out results/close2_tiger_beauty_s42
Writes {out}.json and {out}_per_user.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402

DATASETS = {
    "beauty": dict(seq=ROOT / "data/amazon_beauty_backbone/Beauty.txt", n_items=12101),
    "toys": dict(seq=ROOT / "data/amazon_toys_backbone/Toys_and_Games.txt", n_items=11924),
    "sports": dict(seq=ROOT / "data/amazon_sports_backbone/Sports_and_Outdoors.txt",
                   n_items=18357),
    # EXPAND (next paper): same protocol, same evaluator, new datasets
    "instruments": dict(seq=ROOT / "data/amazon_instruments_backbone/Musical_Instruments.txt",
                        n_items=9922),
    "yelp": dict(seq=ROOT / "data/yelp_backbone/Yelp.txt", n_items=20033),
}
KS = (5, 10)


def liger_row_to_user(split: protocol.LOOSplit) -> list[int]:
    """LIGER eval row -> our test user index: seen users first, then cold users."""
    seen = [i for i in range(len(split.test)) if i not in split.cold_test_idx]
    cold = [i for i in range(len(split.test)) if i in split.cold_test_idx]
    return seen + cold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--beams", required=True, help="BeamRecord JSONL (test split)")
    ap.add_argument("--alignment", default=None,
                    help="optional liger_alignment_test.json to cross-check")
    ap.add_argument("--out", required=True, help="output prefix (writes .json / _per_user.npz)")
    ap.add_argument("--seed", type=int, default=42, help="recorded in the JSON only")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    split = protocol.build(str(ds["seq"]), n_items=ds["n_items"], max_len=20)
    n = len(split.test)
    row2user = liger_row_to_user(split)

    if args.alignment:
        a = json.load(open(args.alignment))
        assert a.get("item_offset", 1) == 1, "expected 1-indexed LIGER item ids"
        assert a["liger_row_to_our_user_test"] == row2user, \
            "alignment file disagrees with derived seen-then-cold mapping"
        print("[close2-tiger] alignment file cross-check OK")

    # gather valid mapped items per LIGER row, beam order preserved
    beams: dict[int, list[tuple[int, int]]] = defaultdict(list)  # row -> [(rank, item_row)]
    dump_targets: dict[int, int] = {}
    n_rec = n_invalid = 0
    for line in open(args.beams):
        r = json.loads(line)
        n_rec += 1
        row = int(r["query_id"].split(":")[1])
        if "target_item" in r:
            dump_targets[row] = int(r["target_item"]) - 1
        if not r["valid"]:
            n_invalid += 1
            continue
        beams[row].append((int(r["beam_rank"]), int(r["mapped_item_id"]) - 1))
    n_rows = max(beams.keys() | dump_targets.keys()) + 1 if (beams or dump_targets) else 0
    assert n_rows == n, f"dump has {n_rows} eval rows but protocol test has {n}"

    hit = {k: np.zeros(n, dtype=np.float32) for k in KS}
    dcg = {k: np.zeros(n, dtype=np.float32) for k in KS}
    n_target_mismatch = 0
    for row in range(n):
        u = row2user[row]
        tgt = split.test[u][1]
        if row in dump_targets and dump_targets[row] != tgt:
            n_target_mismatch += 1
        ranked = []
        seen_items = set()
        for _r, it in sorted(beams.get(row, [])):
            if it not in seen_items:
                seen_items.add(it)
                ranked.append(it)
        for k in KS:
            h, g = protocol.dcg_hit(ranked, tgt, k)
            hit[k][u] = h
            dcg[k][u] = g
    assert n_target_mismatch == 0, (
        f"{n_target_mismatch} rows: dump target_item != protocol target under the "
        "seen-then-cold alignment — dump was NOT produced from the same sequence file")

    cold_mask = np.array([i in split.cold_test_idx for i in range(n)], dtype=bool)

    def rep(mask):
        m = int(mask.sum())
        return {"n": m, **{f"HR@{k}": round(float(hit[k][mask].sum()) / max(m, 1), 4)
                           for k in KS},
                **{f"NDCG@{k}": round(float(dcg[k][mask].sum()) / max(m, 1), 4)
                   for k in KS}}

    out = {
        "experiment": "CLOSE2_tiger_score",
        "dataset": args.dataset, "seed": args.seed,
        "beams": args.beams,
        "n": n, "n_beam_records": n_rec,
        "invalid_beam_frac": round(n_invalid / max(n_rec, 1), 4),
        "overall": rep(np.ones(n, dtype=bool)),
        "seen": rep(~cold_mask),
        "cold": rep(cold_mask),
    }
    out_json = Path(f"{args.out}.json")
    out_npz = Path(f"{args.out}_per_user.npz")
    json.dump(out, open(out_json, "w"), indent=2)
    np.savez(out_npz, cold_mask=cold_mask,
             **{f"hit{k}": hit[k] for k in KS}, **{f"dcg{k}": dcg[k] for k in KS})
    print(json.dumps({kk: out[kk] for kk in ("overall", "seen", "cold",
                                             "invalid_beam_frac")}, indent=2))
    print(f"[close2-tiger] wrote {out_json} and {out_npz}")


if __name__ == "__main__":
    main()
