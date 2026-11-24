"""I44 — prep for the 2x2 injection ablation: plain-catalog embeddings + plain-centroid inputs.

Cells of the 2x2 (item-side enrichment E on/off x query-side centroid C on/off):
  (E,C)  = i24_real towers        [exists]
  (E,-)  = i22_real towers        [exists for Beauty; trained fresh for Toys/Sports]
  (-,-)  = plain-catalog towers   [needs plain openai embeddings]
  (-,C)  = plain + centroid       [needs plain input matrix]

This script produces, per dataset (REAL world only):
  <X>_openai-3-large_embeddings.pt        (plain catalog; skipped if present)
  <X>_openai-3-large_i44plain_input.pt    (plain-emb P3 centroids, i24 convention)

Embedding texts are identical to the i22 pipeline's plain template (id2meta json);
paper embedder: OpenAI (key from $OPENAI_API_KEY only; --embedder to swap). i24 leak-safety convention:
user id dropped, final held-out event dropped before predecessor counting.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import enrich as E  # noqa: E402
from semplan.embedders import add_embedder_arg, embed_texts  # noqa: E402

CFG = {
    "beauty": dict(n=12101, data="amazon_beauty_backbone", meta="beauty_id2meta.json",
                   seq="Beauty.txt", prefix="Beauty"),
    "toys": dict(n=11924, data="amazon_toys_backbone", meta="toys_id2meta.json",
                 seq="Toys_and_Games.txt", prefix="Toys"),
    "sports": dict(n=18357, data="amazon_sports_backbone", meta="sports_id2meta.json",
                   seq="Sports_and_Outdoors.txt", prefix="Sports"),
    # EXPAND (next paper)
    "instruments": dict(n=9922, data="amazon_instruments_backbone",
                        meta="instruments_id2meta.json",
                        seq="Musical_Instruments.txt", prefix="Instruments"),
    "yelp": dict(n=20033, data="yelp_backbone", meta="yelp_id2meta.json",
                 meta_dir="yelp_meta", seq="Yelp.txt", prefix="Yelp"),
}


def embed_plain(cfg, out: Path, embedder: str) -> None:
    meta = json.load(open(ROOT / "data" / cfg.get("meta_dir", "amazon_meta") / cfg["meta"]))
    texts = E.base_texts_from_meta(meta, cfg["n"])
    print(f"{cfg['n']} items, {sum(1 for t in texts if not t)} without meta text", flush=True)
    t0 = time.time()
    torch.save(embed_texts(texts, embedder), out)
    print(f"wrote {out} in {(time.time()-t0)/60:.1f} min", flush=True)


def build_input(cfg, emb_file: Path, out: Path) -> None:
    emb = torch.load(emb_file, map_location="cpu", weights_only=True).float().numpy()
    mat, n_aug = E.centroid_inputs(emb, E.pred_counts(ROOT / "data" / cfg["data"] / cfg["seq"]))
    torch.save(torch.from_numpy(mat), out)
    print(f"{out.name}: pred3-augmented {n_aug}/{cfg['n']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    add_embedder_arg(ap)
    args = ap.parse_args()
    cfg = CFG[args.dataset]
    d = ROOT / "data" / cfg["data"]
    plain = d / f"{cfg['prefix']}_openai-3-large_embeddings.pt"
    if not plain.exists():
        embed_plain(cfg, plain, args.embedder)
    else:
        print(f"{plain.name} exists, skipping embed", flush=True)
    inp = d / f"{cfg['prefix']}_openai-3-large_i44plain_input.pt"
    if inp.exists():
        print(f"{inp.name} exists, skipping", flush=True)
    else:
        build_input(cfg, plain, inp)


if __name__ == "__main__":
    main()
