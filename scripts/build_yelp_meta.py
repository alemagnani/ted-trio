"""Yelp item text: data/yelp_meta/yelp_id2meta.json from LETTER's in-tree Yelp release.

Our Yelp sequences (scripts/i51_prep_yelp_data.py) are the S3Rec/DuoRec/BSARec snapshot
(30,431 users / 20,033 items / 316,354 interactions). That file is ids-only. LETTER
(HonghuiBao2000/LETTER @ 8d0154e) ships the SAME snapshot in-tree as data/Yelp/
Yelp.inter.json + Yelp.item.json (0-indexed; all 30,431 sequences equal Yelp.txt with
item id + 1), and Yelp.item.json carries the business name ("title") and category list
("description") for every item. This script downloads Yelp.item.json at the pinned commit,
archives it as data/yelp_meta/LETTER_Yelp.item.json, and writes the 1-indexed id2meta file
in the same sentence template as the Amazon id2meta files:

    "This item is called '<title>'. It belongs to the categories of <categories>."

Both files are checked against docs/data_checksums.md5 (the output is byte-identical to the
file used in the paper).

Usage: python scripts/build_yelp_meta.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTER_COMMIT = "8d0154e28de37dbb6e24871c508ad8ddb1921cda"
URL = (f"https://raw.githubusercontent.com/HonghuiBao2000/LETTER/{LETTER_COMMIT}/"
       "data/Yelp/Yelp.item.json")
RAW = ROOT / "data/yelp_meta/LETTER_Yelp.item.json"
OUT = ROOT / "data/yelp_meta/yelp_id2meta.json"
RAW_MD5 = "189b749d1e16dcba08094911ed152334"
OUT_MD5 = "069d831b3dec51bc5d26c524ff74c442"
N_ITEMS = 20033


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not (RAW.exists() and md5_bytes(RAW.read_bytes()) == RAW_MD5):
        print(f"downloading {URL}", flush=True)
        if RAW.is_symlink():
            RAW.unlink()
        urllib.request.urlretrieve(URL, RAW)
    got = md5_bytes(RAW.read_bytes())
    if got != RAW_MD5:
        sys.exit(f"{RAW} md5 {got} != expected {RAW_MD5}")
    item = json.load(open(RAW))
    assert len(item) == N_ITEMS, len(item)
    out = {}
    for k, v in item.items():
        out[str(int(k) + 1)] = (f"This item is called '{v['title'].strip()}'. "
                                f"It belongs to the categories of {v['description'].strip()}.")
    assert sorted(int(k) for k in out) == list(range(1, N_ITEMS + 1))
    data = json.dumps(out).encode()
    if OUT.is_symlink():  # never write through a link into someone else's data directory
        OUT.unlink()
    OUT.write_bytes(data)
    status = "OK" if md5_bytes(data) == OUT_MD5 else "MISMATCH"
    print(f"wrote {OUT} ({len(out)} items) md5 {md5_bytes(data)} -> {status}", flush=True)
    if status != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
