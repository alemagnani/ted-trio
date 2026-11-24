"""Inner-world sequence file: drop each user's LAST event (the real test target).

The inner world shifts the leave-one-out split one event earlier: on `{X}_inner.txt` the
canonical protocol split gives inner-val = original event -3 and inner pseudo-test =
original event -2, so all selection happens without ever reading the real test event.
All users are kept (5-core => inner length >= 4).

Toys/Sports/Instruments/Yelp inner files are also written by their prep scripts
(i30_prep_toys.py, i35_prep_sports.py, i50_prep_expand.py) with this exact rule; Beauty's
is produced here. Output is checked against docs/data_checksums.md5.

Usage: python scripts/make_inner_split.py --dataset beauty [toys sports instruments yelp]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "beauty": ("data/amazon_beauty_backbone/Beauty.txt",
               "data/amazon_beauty_backbone/Beauty_inner.txt"),
    "toys": ("data/amazon_toys_backbone/Toys_and_Games.txt",
             "data/amazon_toys_backbone/Toys_inner.txt"),
    "sports": ("data/amazon_sports_backbone/Sports_and_Outdoors.txt",
               "data/amazon_sports_backbone/Sports_inner.txt"),
    "instruments": ("data/amazon_instruments_backbone/Musical_Instruments.txt",
                    "data/amazon_instruments_backbone/Instruments_inner.txt"),
    "yelp": ("data/yelp_backbone/Yelp.txt", "data/yelp_backbone/Yelp_inner.txt"),
}


def expected_md5() -> dict[str, str]:
    out = {}
    for line in open(ROOT / "docs/data_checksums.md5"):
        line = line.strip()
        if line and not line.startswith("#"):
            h, p = line.split(maxsplit=1)
            out[p.strip()] = h
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(FILES), required=True)
    args = ap.parse_args()
    exp = expected_md5()
    for ds in args.dataset:
        src, dst = FILES[ds]
        lines = []
        for line in open(ROOT / src):
            toks = line.split()
            if toks:
                lines.append(" ".join(toks[:-1]))
        data = ("\n".join(lines) + "\n").encode()
        out = ROOT / dst
        if out.exists() and out.read_bytes() == data:
            print(f"[{ds}] {dst} already up to date")
        else:
            if out.is_symlink():  # never write through a link
                out.unlink()
            out.write_bytes(data)
            print(f"[{ds}] wrote {dst} ({len(lines)} users)")
        got = hashlib.md5(data).hexdigest()
        status = "OK" if got == exp.get(dst) else "MISMATCH"
        print(f"[{ds}] md5 {got} expected {exp.get(dst)} -> {status}")


if __name__ == "__main__":
    main()
