"""Backbone prep for Toys / Sports / Instruments / Yelp: inner world + pred3-enriched item
catalogs (E) + I24 query-input matrices (C). The exact Beauty frontier pipeline
(i22_enrich_embed_catalog.py + i24_build_input_emb.py) generalised over the other four
datasets (it was written for Instruments/Yelp as I50/I51; i30_prep_toys.py and
i35_prep_sports.py are thin wrappers that call it for Toys/Sports).

Usage: python scripts/i50_prep_expand.py --dataset {toys,sports,instruments,yelp}
                                         [--embedder openai|fake|st:<model>]

Steps (idempotent -- each output skipped if present):
  1. {Name}_inner.txt        -- each user's LAST item dropped (Beauty_inner convention).
  2. i22-style enrichment    -- "Often bought after: 'p1'; 'p2'; 'p3'." (Yelp: "Often
                               visited after:") from directed consecutive co-occurrence on
                               ids[1:][:-1] (user id + held-out final event dropped; leak-safe),
                               per world (inner/real), embedded (paper: openai
                               text-embedding-3-large) -> {Name}_openai-3-large_
                               i22{inner,real}_embeddings.pt [N, 3072].
  3. I24 input matrices      -- [L2(own emb) ; L2(pred3 centroid, own-emb fallback)]
                               -> {Name}_openai-3-large_i24{inner,real}_input.pt [N, 6144].
OpenAI volume: ~24k-37k short texts per dataset (two worlds), a few cents each.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import enrich as E  # noqa: E402
from semplan.embedders import add_embedder_arg, embed_texts  # noqa: E402

DATASETS = {
    "toys": dict(data=ROOT / "data/amazon_toys_backbone", seq="Toys_and_Games.txt",
                 inner="Toys_inner.txt", prefix="Toys",
                 meta=ROOT / "data/amazon_meta/toys_id2meta.json", n=11924,
                 after_phrase=E.AFTER_BOUGHT),
    "sports": dict(data=ROOT / "data/amazon_sports_backbone", seq="Sports_and_Outdoors.txt",
                   inner="Sports_inner.txt", prefix="Sports",
                   meta=ROOT / "data/amazon_meta/sports_id2meta.json", n=18357,
                   after_phrase=E.AFTER_BOUGHT),
    "instruments": dict(data=ROOT / "data/amazon_instruments_backbone",
                        seq="Musical_Instruments.txt", inner="Instruments_inner.txt",
                        prefix="Instruments",
                        meta=ROOT / "data/amazon_meta/instruments_id2meta.json", n=9922,
                        after_phrase=E.AFTER_BOUGHT),
    "yelp": dict(data=ROOT / "data/yelp_backbone", seq="Yelp.txt", inner="Yelp_inner.txt",
                 prefix="Yelp", meta=ROOT / "data/yelp_meta/yelp_id2meta.json", n=20033,
                 after_phrase=E.AFTER_VISITED),
}


def prep(dataset: str, embedder: str) -> None:
    cfg = DATASETS[dataset]
    DATA, N, P = cfg["data"], cfg["n"], cfg["prefix"]
    meta = json.load(open(cfg["meta"]))
    assert len(meta) == N, (len(meta), N)
    base = E.base_texts_from_meta(meta, N)
    titles = E.titles_from_texts(base)

    real, inner = DATA / cfg["seq"], DATA / cfg["inner"]
    if not inner.exists():
        lines = open(real).read().splitlines()
        assert all(len(l.split()) >= 3 for l in lines if l.strip())
        inner.write_text("\n".join(E.drop_last_event(lines)) + "\n")
        print(f"wrote {inner}", flush=True)
    else:
        print(f"skip {inner}: exists", flush=True)

    for world, seq in (("inner", inner), ("real", real)):
        out_path = DATA / f"{P}_openai-3-large_i22{world}_embeddings.pt"
        if out_path.exists():
            print(f"skip {world} embed: exists", flush=True)
            continue
        texts, n = E.enriched_texts(base, titles, E.pred_counts(seq), cfg["after_phrase"])
        print(f"{seq.name}: {n}/{N} items enriched", flush=True)
        emb = embed_texts(texts, embedder)
        torch.save(emb, out_path)
        print(f"wrote {out_path} shape {tuple(emb.shape)}", flush=True)

    for world, seq in (("inner", inner), ("real", real)):
        out_file = DATA / f"{P}_openai-3-large_i24{world}_input.pt"
        if out_file.exists():
            print(f"skip {out_file.name}: exists", flush=True)
            continue
        emb = torch.load(DATA / f"{P}_openai-3-large_i22{world}_embeddings.pt",
                         map_location="cpu", weights_only=True).float().numpy()
        out, n_aug = E.centroid_inputs(emb, E.pred_counts(seq))
        torch.save(torch.from_numpy(out), out_file)
        print(f"{out_file.name}: shape {out.shape}, pred3-augmented {n_aug}/{N}", flush=True)


def main(default_dataset=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=default_dataset is None, default=default_dataset,
                    choices=sorted(DATASETS))
    add_embedder_arg(ap)
    args = ap.parse_args()
    prep(args.dataset, args.embedder)


if __name__ == "__main__":
    main()
