"""Validate the prepared datasets BEFORE any training (release addition).

For every requested dataset checks: file presence (sequence file, inner split, item text,
and -- unless --no-embeddings -- the item-side matrices), the user / item / interaction
counts printed in the paper, the 5-core property (every user >= 5 events, every item >= 5
occurrences), contiguous 1-indexed ids, that the inner split is exactly "drop each user's
last event", that id2meta covers every item, embedding-matrix shapes, and the sha256 of the
processed sequence files behind the paper.

Data directory: --data-dir, else $TT_DATA_DIR, else <repo>/data.
Exit code 0 iff every check passes.

Usage: python scripts/check_data.py [--dataset beauty toys ...] [--data-dir DIR]
                                    [--no-embeddings] [--no-checksums]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "beauty": dict(bb="amazon_beauty_backbone", seq="Beauty.txt", inner="Beauty_inner.txt",
                   prefix="Beauty", meta="amazon_meta/beauty_id2meta.json",
                   users=22363, items=12101, inter=198502,
                   sha256="226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8"),
    "toys": dict(bb="amazon_toys_backbone", seq="Toys_and_Games.txt", inner="Toys_inner.txt",
                 prefix="Toys", meta="amazon_meta/toys_id2meta.json",
                 users=19412, items=11924, inter=167597,
                 sha256="176c680c688d497598c4a89a03b402c0fe78d795e5db335c2e78dd5f2d778a3f"),
    "sports": dict(bb="amazon_sports_backbone", seq="Sports_and_Outdoors.txt",
                   inner="Sports_inner.txt", prefix="Sports",
                   meta="amazon_meta/sports_id2meta.json",
                   users=35598, items=18357, inter=296337,
                   sha256="2a095e0648872beb9982408958290f5d655db5be2911c3756b5446de9f0b8de2"),
    "instruments": dict(bb="amazon_instruments_backbone", seq="Musical_Instruments.txt",
                        inner="Instruments_inner.txt", prefix="Instruments",
                        meta="amazon_meta/instruments_id2meta.json",
                        users=24772, items=9922, inter=206153,
                        sha256="b45a59d9a0ee22ffaafa896546adfee067e883c856e833633d4b89b2a61f13e7"),
    "yelp": dict(bb="yelp_backbone", seq="Yelp.txt", inner="Yelp_inner.txt", prefix="Yelp",
                 meta="yelp_meta/yelp_id2meta.json",
                 users=30431, items=20033, inter=316354,
                 sha256="724b219106b81dd2349027da997334ddb6da9fe2f152ad6e0d97a93170f7caed"),
}
EMB = {"_openai-3-large_embeddings.pt": 3072, "_openai-3-large_i22inner_embeddings.pt": 3072,
       "_openai-3-large_i22real_embeddings.pt": 3072, "_openai-3-large_i24inner_input.pt": 6144,
       "_openai-3-large_i24real_input.pt": 6144, "_openai-3-large_i44plain_input.pt": 6144,
       "_sentence-t5-xl_embeddings.pt": 768}


class Report:
    def __init__(self):
        self.n_ok = self.n_bad = 0

    def check(self, ok: bool, what: str) -> bool:
        print(f"  [{'OK' if ok else 'FAIL'}] {what}")
        self.n_ok += ok
        self.n_bad += not ok
        return ok


def check_dataset(ds: str, data: Path, rep: Report, embeddings: bool, checksums: bool) -> None:
    cfg = DATASETS[ds]
    print(f"== {ds}")
    seq = data / cfg["bb"] / cfg["seq"]
    if not rep.check(seq.exists(), f"sequence file present: {seq}"):
        return
    raw = seq.read_bytes()
    lines = [l for l in raw.decode().splitlines() if l.strip()]
    uids, user_len, item_cnt = [], [], Counter()
    for l in lines:
        toks = [int(t) for t in l.split()]
        uids.append(toks[0])
        user_len.append(len(toks) - 1)
        item_cnt.update(toks[1:])
    n_inter = sum(user_len)
    rep.check(len(lines) == cfg["users"], f"users {len(lines)} == {cfg['users']}")
    rep.check(len(item_cnt) == cfg["items"], f"items {len(item_cnt)} == {cfg['items']}")
    rep.check(n_inter == cfg["inter"], f"interactions {n_inter} == {cfg['inter']}")
    rep.check(min(user_len) >= 5, f"5-core users: min sequence length {min(user_len)} >= 5")
    rep.check(min(item_cnt.values()) >= 5,
              f"5-core items: min item frequency {min(item_cnt.values())} >= 5")
    rep.check(sorted(item_cnt) == list(range(1, cfg["items"] + 1)),
              "item ids contiguous 1..N")
    rep.check(uids == list(range(1, len(uids) + 1)), "user ids contiguous 1..U in file order")
    if checksums:
        got = hashlib.sha256(raw).hexdigest()
        rep.check(got == cfg["sha256"], f"sha256 {got[:16]}... == paper snapshot")
    inner = data / cfg["bb"] / cfg["inner"]
    if rep.check(inner.exists(), f"inner split present: {inner.name}"):
        want = "\n".join(" ".join(l.split()[:-1]) for l in lines) + "\n"
        rep.check(inner.read_text() == want, "inner split == each user's last event dropped")
    meta = data / cfg["meta"]
    if rep.check(meta.exists(), f"item text present: {cfg['meta']}"):
        m = json.load(open(meta))
        rep.check(sorted(int(k) for k in m) == list(range(1, cfg["items"] + 1))
                  and all(m.values()), f"id2meta covers all {cfg['items']} items, no empty text")
    if embeddings:
        import torch
        for suf, dim in EMB.items():
            p = data / cfg["bb"] / f"{cfg['prefix']}{suf}"
            if rep.check(p.exists(), f"present: {p.name}"):
                t = torch.load(p, map_location="cpu", weights_only=True)
                rep.check(tuple(t.shape) == (cfg["items"], dim),
                          f"shape {tuple(t.shape)} == ({cfg['items']}, {dim})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    ap.add_argument("--data-dir", default=os.environ.get("TT_DATA_DIR", str(ROOT / "data")))
    ap.add_argument("--no-embeddings", action="store_true",
                    help="only check sequence/text files (before the embedding step)")
    ap.add_argument("--no-checksums", action="store_true")
    args = ap.parse_args()
    rep = Report()
    for ds in args.dataset:
        check_dataset(ds, Path(args.data_dir), rep, not args.no_embeddings, not args.no_checksums)
    print(f"\n{rep.n_ok} checks passed, {rep.n_bad} failed")
    sys.exit(1 if rep.n_bad else 0)


if __name__ == "__main__":
    main()
