"""I50 — Amazon Reviews 2018 Musical_Instruments sequential-rec backbone acquisition.

Acquires the canonical "Instruments" snapshot used by LC-Rec (arXiv 2311.09049),
LETTER (2405.07314), EAGER-LLM, Semantic Convergence, LOHRec. Instead of
re-running the raw-Amazon-2018 pipeline, we download the *processed data released
by the LC-Rec authors themselves* (provenance-clean; guarantees the exact
snapshot those papers trained on):

  Provenance
  ----------
  Repo   : https://github.com/RUCAIBox/LC-Rec  (main @ e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669)
  Script : data_process/amazon18_data_process.py  (raw Amazon-2018 ratings CSV +
           meta filter + interaction dedup per user + iterative 5-core on users
           AND items + per-user timestamp sort; leave-one-out splits downstream)
  Data   : README "Dataset" Google Drive folder
           https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6
           file Instruments.zip, id 19tsIxV_LbErh5FfYhT732M82y0V-3bxR
           (md5 21a8e3b824ddb2c8f587b5d38c732b7f, dated 2023-11-15)

  Counts
  ------
  Released snapshot: 24,772 users x 9,922 items x 206,153 interactions
  (0-indexed, contiguous ids). The LC-Rec paper table prints 24,773 / 9,923 /
  206,153 — users and items are each exactly +1 vs. their own released data
  (pad-inclusive vocab convention); interactions match exactly. Follow-up papers
  (LETTER builds directly on the LC-Rec codebase/data) inherit the same snapshot.

Outputs (idempotent — skipped if present):
  data/amazon_instruments_backbone/Musical_Instruments.txt
      one line per user: "user_id item_id item_id ..." (1-indexed, chronological;
      Toys_and_Games.txt convention)
  data/amazon_meta/instruments_id2meta.json
      {1-indexed item id -> natural-language meta text}, replicating the exact
      toys_id2meta.json sentence template:
        "This item is called 'TITLE'. " (if title)
        "It is priced at PRICE. "       (Amazon-2018 Instruments meta has no
                                         price field -> omitted for all items,
                                         matching the toys behaviour for items
                                         without a price)
        "It is manufactured by 'BRAND'. " (if brand)
        "It belongs to the categories of C1, C2, ... . " (always; root category
         'Musical Instruments' dropped, mirroring toys where the root
         'Toys & Games' never appears)

No embedding API is called and no API key is read — data prep only.
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "data/amazon_instruments_backbone"
META_DIR = ROOT / "data/amazon_meta"
TMP = ROOT / "data/raw/lcrec_instruments"  # download cache (was a /tmp dir)

GDRIVE_FILE_ID = "19tsIxV_LbErh5FfYhT732M82y0V-3bxR"  # Instruments.zip (LC-Rec release)
ZIP_MD5 = "21a8e3b824ddb2c8f587b5d38c732b7f"

TXT_OUT = BACKBONE / "Musical_Instruments.txt"
META_OUT = META_DIR / "instruments_id2meta.json"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_release():
    """Download + unzip the LC-Rec Instruments release into TMP (idempotent)."""
    inter = TMP / "Instruments/Instruments.inter.json"
    item = TMP / "Instruments/Instruments.item.json"
    if inter.exists() and item.exists():
        print(f"skip download: {inter} exists", flush=True)
        return inter, item
    TMP.mkdir(parents=True, exist_ok=True)
    zpath = TMP / "Instruments.zip"
    if not (zpath.exists() and md5(zpath) == ZIP_MD5):
        print("downloading Instruments.zip from LC-Rec Google Drive ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "gdown", GDRIVE_FILE_ID, "-O", str(zpath)],
            check=True)
    got = md5(zpath)
    assert got == ZIP_MD5, f"Instruments.zip md5 {got} != expected {ZIP_MD5}"
    with zipfile.ZipFile(zpath) as z:
        z.extract("Instruments/Instruments.inter.json", TMP)
        z.extract("Instruments/Instruments.item.json", TMP)
    print(f"extracted to {TMP}/Instruments/", flush=True)
    return inter, item


def write_backbone(inter_path):
    inter = json.load(open(inter_path))
    keys = sorted(inter.keys(), key=int)
    assert [int(k) for k in keys] == list(range(len(keys))), "non-contiguous uids"
    with open(TXT_OUT, "w") as f:
        for k in keys:
            seq = inter[k]
            assert len(seq) >= 5, (k, seq)
            f.write(f"{int(k) + 1} " + " ".join(str(i + 1) for i in seq) + "\n")
    print(f"wrote {TXT_OUT}", flush=True)


def build_meta_text(m):
    """Replicate the toys_id2meta.json sentence template exactly."""
    parts = []
    title = (m.get("title") or "").strip()
    if title:
        parts.append(f"This item is called '{title}'. ")
    price = (str(m.get("price") or "")).strip()   # absent in Instruments-2018 meta
    if price:
        parts.append(f"It is priced at {price}. ")
    brand = (m.get("brand") or "").strip()
    if brand:
        parts.append(f"It is manufactured by '{brand}'. ")
    cats = [c.strip() for c in (m.get("categories") or "").split(",") if c.strip()]
    if cats and cats[0] == "Musical Instruments":   # drop root, mirroring toys
        cats = cats[1:]
    parts.append("It belongs to the categories of " + ", ".join(cats) + ". ")
    return "".join(parts)


def write_meta(item_path):
    item = json.load(open(item_path))
    out = {str(int(k) + 1): build_meta_text(v) for k, v in item.items()}
    assert sorted(int(k) for k in out) == list(range(1, len(out) + 1))
    with open(META_OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {META_OUT} ({len(out)} items)", flush=True)


def verify():
    users, items, inters, minlen = 0, set(), 0, 10 ** 9
    for line in open(TXT_OUT):
        toks = line.split()
        users += 1
        seq = toks[1:]
        inters += len(seq)
        items.update(seq)
        minlen = min(minlen, len(seq))
    meta = json.load(open(META_OUT))
    print(f"users={users} items={len(items)} interactions={inters} "
          f"min_seq_len={minlen}", flush=True)
    print(f"meta entries={len(meta)}", flush=True)
    print(f"md5({TXT_OUT.name})={md5(TXT_OUT)}", flush=True)
    assert (users, len(items), inters) == (24772, 9922, 206153), "count mismatch"
    assert minlen >= 5
    print("counts match LC-Rec released snapshot "
          "(paper table prints these +1 on users/items: 24,773 / 9,923)", flush=True)


if __name__ == "__main__":
    BACKBONE.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    if TXT_OUT.exists() and META_OUT.exists():
        print("outputs exist; verifying only", flush=True)
    else:
        inter_path, item_path = fetch_release()
        if not TXT_OUT.exists():
            write_backbone(inter_path)
        else:
            print(f"skip {TXT_OUT}: exists", flush=True)
        if not META_OUT.exists():
            write_meta(item_path)
        else:
            print(f"skip {META_OUT}: exists", flush=True)
    verify()
