"""Amazon-2014 5-core backbones (Beauty / Toys_and_Games / Sports_and_Outdoors).

The paper's Amazon-2014 datasets are the TIGER-lineage canonical 5-core snapshots, built
by the *unmodified* LIGER preprocessing (facebookresearch/liger @ b6ccc37, file
ID_generation/preprocessing/data_process.py). That code downloads the public McAuley
Amazon-2014 "5-core" review files + metadata from SNAP:

  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{Name}_5.json.gz
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{Name}.json.gz

and writes `{Name}.txt` (one line per user: "uid iid iid ..." chronological, 1-indexed),
`{Name}_title_price_brand_categories_amazon_id2meta.json` (item text) and
`{Name}_item2attributes.json`. This driver calls LIGER's own `preprocessing()` (we do not
vendor LIGER: it is CC BY-NC 4.0 — clone it yourself, see README) and copies the outputs
into this repo's data layout:

  data/amazon_{ds}_backbone/{Name}.txt
  data/amazon_{ds}_backbone/{Name}_item2attributes.json
  data/amazon_meta/{ds}_id2meta.json

then checks every file against the md5 of the files behind the paper
(docs/data_checksums.md5). The original Toys/Sports files are byte-identical to LIGER's
preprocessing output; Beauty was produced by the same code inside the LIGER run on the
GPU box (J1) and synced back.

`--t5` additionally builds the Beauty free-arm content matrix
`data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt` exactly the way the paper's
file was made: LIGER's `process_embeddings` (sentence-transformers/sentence-t5-xl over the
sorted id2meta texts, batch 4, then sklearn StandardScaler), saved as a CPU float32 tensor.
(Toys/Sports/Instruments/Yelp t5 matrices come from scripts/i30_embed_t5_toys.py,
i35_embed_t5_sports.py and i50_embed_t5_expand.py instead; those are L2-normalised and
NOT standard-scaled — this asymmetry is historical and faithful to the paper.)

Usage:
  git clone https://github.com/facebookresearch/liger.git third_party/liger
  git -C third_party/liger checkout b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524
  python scripts/prep_amazon2014.py --dataset beauty toys sports
  python scripts/prep_amazon2014.py --dataset beauty --t5          # + Beauty t5-xl matrix
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIGER_COMMIT = "b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524"

DATASETS = {
    "beauty": dict(name="Beauty", backbone="amazon_beauty_backbone", n_items=12101),
    "toys": dict(name="Toys_and_Games", backbone="amazon_toys_backbone", n_items=11924),
    "sports": dict(name="Sports_and_Outdoors", backbone="amazon_sports_backbone",
                   n_items=18357),
}
# LIGER configs/dataset/amazon.yaml
FEATURES = ["title", "price", "brand", "categories"]
PROMPT_FORMAT = "amazon"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_md5() -> dict[str, str]:
    out = {}
    for line in open(ROOT / "docs/data_checksums.md5"):
        line = line.strip()
        if line and not line.startswith("#"):
            h, p = line.split(maxsplit=1)
            out[p.strip()] = h
    return out


def import_liger(liger_root: Path):
    if not (liger_root / "ID_generation/preprocessing/data_process.py").exists():
        sys.exit(f"LIGER clone not found at {liger_root} (see module docstring)")
    try:
        import wget  # noqa: F401  (LIGER imports it; only its Steam path uses it)
    except ImportError:
        sys.modules["wget"] = types.ModuleType("wget")
    sys.path.insert(0, str(liger_root))
    from ID_generation.preprocessing.data_process import preprocessing
    return preprocessing


def build(ds: str, liger_root: Path, raw_dir: Path, work_dir: Path) -> None:
    cfg = DATASETS[ds]
    name = cfg["name"]
    preprocessing = import_liger(liger_root)
    config = {
        "type": "Amazon", "name": name, "features_needed": FEATURES,
        "prompt_format": PROMPT_FORMAT,
        "raw_data_path": str(raw_dir) + "/", "processed_data_path": str(work_dir) + "/",
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_file, id2meta_file, item2attr_file = preprocessing(config)
    bb = ROOT / "data" / cfg["backbone"]
    meta_dir = ROOT / "data/amazon_meta"
    bb.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for dst in (bb / f"{name}.txt", bb / f"{name}_item2attributes.json",
                meta_dir / f"{ds}_id2meta.json"):
        if dst.is_symlink():  # never write through a link
            dst.unlink()
    shutil.copyfile(data_file, bb / f"{name}.txt")
    shutil.copyfile(item2attr_file, bb / f"{name}_item2attributes.json")
    shutil.copyfile(id2meta_file, meta_dir / f"{ds}_id2meta.json")
    print(f"[{ds}] wrote {bb / f'{name}.txt'}, {meta_dir / f'{ds}_id2meta.json'}", flush=True)


def build_t5_beauty(liger_root: Path, work_dir: Path) -> None:
    import torch
    sys.path.insert(0, str(liger_root))
    from ID_generation.utils import process_embeddings

    out = ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"
    tmp = work_dir / "Beauty_sentence-t5-xl_embeddings.pt"
    work_dir.mkdir(parents=True, exist_ok=True)  # --verify-only skips build(), which made it
    if tmp.exists():
        tmp.unlink()  # process_embeddings would otherwise just load a stale file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb = process_embeddings({"dataset": {"content_model": "sentence-t5-xl"}}, device,
                             str(ROOT / "data/amazon_meta/beauty_id2meta.json"), str(tmp))
    emb = emb.detach().float().cpu()
    assert tuple(emb.shape) == (DATASETS["beauty"]["n_items"], 768), emb.shape
    torch.save(emb, out)
    print(f"[beauty] wrote {out} {tuple(emb.shape)} (StandardScaler sentence-t5-xl)",
          flush=True)


def verify(ds: str) -> bool:
    exp = expected_md5()
    cfg = DATASETS[ds]
    files = [f"data/{cfg['backbone']}/{cfg['name']}.txt",
             f"data/{cfg['backbone']}/{cfg['name']}_item2attributes.json",
             f"data/amazon_meta/{ds}_id2meta.json"]
    ok = True
    for rel in files:
        got = md5(ROOT / rel)
        good = got == exp.get(rel)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {rel} md5={got} expected={exp.get(rel)}",
              flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), required=True)
    ap.add_argument("--liger-root", default=str(ROOT / "third_party/liger"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data/raw/amazon2014"),
                    help="download cache for the SNAP files (LIGER writes into <raw>/amazon/)")
    ap.add_argument("--work-dir", default=str(ROOT / "data/raw/liger_processed"))
    ap.add_argument("--t5", action="store_true",
                    help="also build the Beauty StandardScaler sentence-t5-xl matrix")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    liger_root = Path(args.liger_root)
    all_ok = True
    for ds in args.dataset:
        if not args.verify_only:
            build(ds, liger_root, Path(args.raw_dir), Path(args.work_dir))
        all_ok &= verify(ds)
    if args.t5:
        build_t5_beauty(liger_root, Path(args.work_dir))
    if not all_ok:
        sys.exit("md5 mismatch: the processed files differ from the paper's snapshot")
    print("all backbone files match the paper's md5s", flush=True)


if __name__ == "__main__":
    main()
