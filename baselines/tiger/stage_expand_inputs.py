"""Stage Instruments / Yelp inputs into a LIGER clone for the TIGER reproduction.

LIGER's preprocessing only knows how to download Amazon-2014 Beauty/Toys/Sports, so for
Instruments (LC-Rec snapshot) and Yelp (S3Rec/LETTER snapshot) the paper's TIGER runs used
LIGER with `preprocessing()` replaced by existence checks (see run_expand.py) and the
`processed/` directory pre-seeded with OUR backbone files:

  processed/{Name}.txt                                            <- our sequence file (md5-checked)
  processed/{Name}_title_price_brand_categories_amazon_id2meta.json <- our id2meta json
  processed/{Name}_sentence-t5-xl_embeddings.pt                   <- StandardScaler(t5-xl), CUDA tensor

The embedding file must be StandardScaler-transformed (LIGER standard-scales its t5
embeddings before RQ-VAE; seeding raw vectors would silently change the SID family) and
saved as a CUDA tensor (LIGER loads it with torch.load and uses it in place on the GPU).
It is derived from our L2-normalised sentence-t5-xl matrix
(scripts/i50_embed_t5_expand.py). NOTE: this staging step is a reconstruction: the original
on-box staging commands were not archived; the recipe above is the one recorded in the
project log.

Usage (from this repo's root, with a GPU and the LIGER clone at third_party/liger):
  python baselines/tiger/stage_expand_inputs.py --dataset instruments yelp
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "instruments": dict(name="Musical_Instruments",
                        seq="data/amazon_instruments_backbone/Musical_Instruments.txt",
                        meta="data/amazon_meta/instruments_id2meta.json",
                        t5="data/amazon_instruments_backbone/Instruments_sentence-t5-xl_embeddings.pt",
                        seq_md5="0cda39d087a34cf922b835c78d14b7fd", n_items=9922),
    "yelp": dict(name="Yelp", seq="data/yelp_backbone/Yelp.txt",
                 meta="data/yelp_meta/yelp_id2meta.json",
                 t5="data/yelp_backbone/Yelp_sentence-t5-xl_embeddings.pt",
                 seq_md5="da7989002c123e7680afaaa5141fb85b", n_items=20033),
}


def main() -> None:
    import torch
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), required=True)
    ap.add_argument("--liger-root", default=str(ROOT / "third_party/liger"))
    args = ap.parse_args()
    proc = Path(args.liger_root) / "ID_generation/preprocessing/processed"
    proc.mkdir(parents=True, exist_ok=True)
    for ds in args.dataset:
        cfg = DATASETS[ds]
        name = cfg["name"]
        seq = ROOT / cfg["seq"]
        got = hashlib.md5(seq.read_bytes()).hexdigest()
        assert got == cfg["seq_md5"], f"{seq} md5 {got} != {cfg['seq_md5']}"
        shutil.copyfile(seq, proc / f"{name}.txt")
        shutil.copyfile(ROOT / cfg["meta"],
                        proc / f"{name}_title_price_brand_categories_amazon_id2meta.json")
        emb = torch.load(ROOT / cfg["t5"], map_location="cpu", weights_only=True).float()
        assert tuple(emb.shape) == (cfg["n_items"], 768), emb.shape
        scaled = torch.tensor(StandardScaler().fit_transform(emb.numpy()), dtype=torch.float32)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("WARNING: no CUDA device; saving a CPU tensor (LIGER expects CUDA)")
        torch.save(scaled.to(dev), proc / f"{name}_sentence-t5-xl_embeddings.pt")
        print(f"[{ds}] staged {name}.txt, id2meta, t5-xl (StandardScaler, {dev}) in {proc}")


if __name__ == "__main__":
    main()
