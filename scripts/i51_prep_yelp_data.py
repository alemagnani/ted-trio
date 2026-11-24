"""I51 — Yelp sequential-rec backbone acquisition (DuoRec/BSARec snapshot).

Acquires the canonical "Yelp" snapshot used by DuoRec (WSDM'22), BSARec
(AAAI'24, arXiv 2312.10325), and many others: Yelp Open Dataset transactions
after 2019-01-01, 5-core, 30,431 users x 20,033 items x 316,354 interactions.

We download the *processed sequence file released by the authors themselves*
(provenance-clean; guarantees the exact snapshot those papers trained on):

  Provenance
  ----------
  Origin : https://github.com/RUCAIBox/CIKM2020-S3Rec  (S3Rec, CIKM'20)
           data/Yelp.txt, added in first commit
           e6c774da92dba847b00ac89e8bfc5c55cc44d378 (2020-08-10)
  Copy   : https://github.com/yehjin-shin/BSARec  (main @
           c80bdc0f1a889b78fe9a27e2058389cc84caea84) src/data/Yelp.txt —
           byte-identical to the S3Rec original
           (md5 da7989002c123e7680afaaa5141fb85b for both).
           BSARec's src/data/process/README.md states the authors could NOT
           reproduce this file from raw Yelp data ("the Yelp raw data used in
           the code base could not be located") and used the S3Rec
           pre-processed file directly; DuoRec's repo (RuihongQiu/DuoRec)
           releases no Yelp data at all (RecBole example datasets only), so
           the S3Rec file is the single source of truth for this snapshot.

  Counts
  ------
  30,431 users / 20,033 items / 316,354 interactions — matches the BSARec
  paper table (and DuoRec) EXACTLY. Ids are 1-indexed and contiguous
  (users 1..30431 in order, items 1..20033); sequences chronological;
  min sequence length 5 (5-core).

  Metadata caveat
  ---------------
  The released file is sequence-only: no mapping from the integer item ids
  back to Yelp business_ids exists anywhere in the S3Rec/DuoRec/BSARec repos,
  and the exact raw snapshot is unrecoverable (per BSARec's own README).
  Therefore no yelp_id2meta.json can be built with clean provenance — the
  business metadata (name/categories/city) cannot be attached to these ids.
  We deliberately do NOT fabricate one.

Outputs (idempotent — skipped if present):
  data/yelp_backbone/Yelp.txt
      one line per user: "user_id item_id item_id ..." (1-indexed,
      chronological; Toys_and_Games.txt convention). The released file
      already follows this convention exactly, so it is copied verbatim
      after md5 verification.

No embedding API is called and no API key is read — data prep only.
"""
import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "data/yelp_backbone"
TMP = ROOT / "data/raw/yelp_release"  # download cache (was a /tmp dir)

# Pin to the exact BSARec commit (byte-identical to the S3Rec original).
URL = ("https://raw.githubusercontent.com/yehjin-shin/BSARec/"
       "c80bdc0f1a889b78fe9a27e2058389cc84caea84/src/data/Yelp.txt")
FALLBACK_URL = ("https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/"
                "e6c774da92dba847b00ac89e8bfc5c55cc44d378/data/Yelp.txt")
TXT_MD5 = "da7989002c123e7680afaaa5141fb85b"

TXT_OUT = BACKBONE / "Yelp.txt"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_release():
    """Download the released Yelp.txt into TMP (idempotent, md5-verified)."""
    TMP.mkdir(parents=True, exist_ok=True)
    tpath = TMP / "Yelp_release.txt"
    if tpath.exists() and md5(tpath) == TXT_MD5:
        print(f"skip download: {tpath} exists with correct md5", flush=True)
        return tpath
    for url in (URL, FALLBACK_URL):
        print(f"downloading {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, tpath)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", flush=True)
            continue
        if md5(tpath) == TXT_MD5:
            return tpath
        print(f"  md5 mismatch: {md5(tpath)}", flush=True)
    raise RuntimeError("could not fetch Yelp.txt with expected md5 "
                       f"{TXT_MD5} from either pinned URL")


def verify():
    users, items, inters, minlen, maxit = 0, set(), 0, 10 ** 9, 0
    uids = []
    for line in open(TXT_OUT):
        toks = line.split()
        uids.append(int(toks[0]))
        seq = [int(t) for t in toks[1:]]
        users += 1
        inters += len(seq)
        items.update(seq)
        minlen = min(minlen, len(seq))
        maxit = max(maxit, max(seq))
    print(f"users={users} items={len(items)} interactions={inters} "
          f"min_seq_len={minlen}", flush=True)
    print(f"md5({TXT_OUT.name})={md5(TXT_OUT)}", flush=True)
    assert (users, len(items), inters) == (30431, 20033, 316354), "count mismatch"
    assert uids == list(range(1, users + 1)), "user ids not contiguous 1..N"
    assert maxit == len(items) and min(items) == 1, "item ids not contiguous 1..M"
    assert minlen >= 5
    assert md5(TXT_OUT) == TXT_MD5
    print("counts match the DuoRec/BSARec paper table exactly "
          "(30,431 / 20,033 / 316,354)", flush=True)


if __name__ == "__main__":
    BACKBONE.mkdir(parents=True, exist_ok=True)
    if TXT_OUT.exists():
        print("output exists; verifying only", flush=True)
    else:
        src = fetch_release()
        shutil.copyfile(src, TXT_OUT)
        print(f"wrote {TXT_OUT}", flush=True)
    verify()

# ---------------------------------------------------------------------------
# ADDENDUM (2026-08-10, post-acquisition): item text for this exact snapshot.
# LETTER (CIKM'24, HonghuiBao2000/LETTER @ 8d0154e) ships data/Yelp/ in-tree with
# Yelp.inter.json + Yelp.item.json on the SAME S3Rec snapshot: verified locally that
# all 30,431 LETTER sequences (0-indexed) equal our Yelp.txt lines with item id +1
# (30431/30431 exact match, 0 mismatches). Yelp.item.json has full coverage
# (0/20,033 missing title/description: business name + category list).
# data/yelp_meta/yelp_id2meta.json is built from it (1-indexed, toys template):
#   "This item is called '<title>'. It belongs to the categories of <cats>."
# Raw source archived at data/yelp_meta/LETTER_Yelp.item.json. This makes the
# BSARec/DuoRec dense bars AND the LETTER-TIGER generative bars directly comparable
# on one snapshot, with item text available for our frozen-embedding towers.
# ---------------------------------------------------------------------------
