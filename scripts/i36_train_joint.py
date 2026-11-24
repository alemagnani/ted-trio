"""I36 — multi-dataset JOINT query-tower trunk (Beauty + Toys + Sports inner worlds).

Hypothesis: the I24 recipe transferred to Toys (I30) and Sports (I35) with ZERO tuning,
so a single trunk trained jointly on all three inner worlds may beat the per-dataset
trios via shared sequential structure. Architecture is IDENTICAL to the J5a/I24 tower
(scripts/j5/train_j5a.py: d=256, 2L, 4H, d_ff=512, tau=0.05, K=4, include-val, 6144-d
[own emb ; pred3 centroid] inputs, full-softmax CE + forbidden masking) — the ONLY
change is the data feed: batches are interleaved ROUND-ROBIN over the 3 datasets
(step % 3), each dataset scoring against its OWN frozen L2-normalised item catalog.

Interleaving decision: round-robin, NOT proportional-to-size. Sports has ~2-3x the
training pairs of Beauty/Toys and would dominate a proportional schedule; round-robin
gives each inner world an equal optimisation budget (steps/3 gradient steps each),
which is the cleanest test of "shared structure helps" vs "more data helps". Note the
default --steps 20000 therefore gives each dataset ~6.7k steps vs 20k in the
per-dataset baselines — the comparison is deliberately compute-matched on the trunk.

CHECKPOINT FORMAT (decision): ONE trunk file per seed, saved with the standard
``semplan.dense.frozen_query.save_checkpoint`` -> ``<out-dir>/j5.pt``. This works
because save_checkpoint EXCLUDES the frozen ``item``/``input_item`` buffers and
load_checkpoint DROPS the shape keys (item_dim/n_items/input_dim) from the stored
config and loads with strict=False: the trained trunk (in_proj 6144->256, encoder,
pool, out_proj 256->3072) is shape-compatible with every dataset because all three
share D_in=6144 / D=3072. So the SAME j5.pt is loaded per dataset via
``load_checkpoint(path, item_emb=<that dataset's l2 catalog>, input_emb=<its input>)``
— no per-dataset checkpoint copies needed. The script verifies this round-trip for
all three datasets after training.

Usage (box, one job per seed — co-locate, ~3 GB VRAM each):
  python scripts/i36_train_joint.py --seed 42 \
      --out-dir outputs/artifacts/i36_joint_s42
CPU smoke (synthetic 3-dataset fixture, ~50 steps):
  python scripts/i36_train_joint.py --smoke --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semplan.dense.chn_de import ChnDeDataset  # noqa: E402
from semplan.dense.frozen_query import (  # noqa: E402
    FrozenQueryTower, collate_with_forbidden, encode_split_queries, load_checkpoint,
    save_checkpoint,
)
from semplan.eval import protocol  # noqa: E402

# I24 recipe constants — replicated verbatim from the winning per-dataset runs
# (train_j5a --n-query 4 --steps 20000 --include-val, defaults d256/2L/4H).
D, LAYERS, HEADS, D_FF, TAU, N_QUERY, MAX_LEN = 256, 2, 4, 512, 0.05, 4, 20
KS = (5, 10)

DATASETS = {  # ORDER FIXED — round-robin index = step % 3
    "beauty": {
        "seq": ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt",
        "item_emb": ROOT / "data/amazon_beauty_backbone/"
                           "Beauty_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_beauty_backbone/"
                            "Beauty_openai-3-large_i24inner_input.pt",
        "n_items": 12101},
    "toys": {
        "seq": ROOT / "data/amazon_toys_backbone/Toys_inner.txt",
        "item_emb": ROOT / "data/amazon_toys_backbone/"
                           "Toys_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_toys_backbone/"
                            "Toys_openai-3-large_i24inner_input.pt",
        "n_items": 11924},
    "sports": {
        "seq": ROOT / "data/amazon_sports_backbone/Sports_inner.txt",
        "item_emb": ROOT / "data/amazon_sports_backbone/"
                           "Sports_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_sports_backbone/"
                            "Sports_openai-3-large_i24inner_input.pt",
        "n_items": 18357},
}

SMOKE_ROOT = ROOT / "outputs/tmp/i36_smoke"


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def make_smoke_fixture(root: Path, n_datasets: int = 3, n_items: int = 50,
                       n_users: int = 30, d_item: int = 16, seed: int = 0) -> dict:
    """Deterministic tiny 3-dataset fixture mirroring the real layout (seq txt +
    item/input/content .pt). Input dim = 2*d_item, matching the [own;pred3] doubling."""
    rng = np.random.default_rng(seed)
    fix = root / "fixture"
    datasets = {}
    for di in range(n_datasets):
        name = f"fake{di}"
        dd = fix / name
        dd.mkdir(parents=True, exist_ok=True)
        seq_p = dd / f"{name}_inner.txt"
        if not seq_p.exists():
            lines = []
            for u in range(n_users):
                L = int(rng.integers(5, 13))
                items = rng.choice(n_items, size=L, replace=False) + 1  # 1-indexed
                lines.append(" ".join([str(u + 1)] + [str(i) for i in items]))
            seq_p.write_text("\n".join(lines) + "\n")
            item = rng.standard_normal((n_items, d_item)).astype(np.float32)
            torch.save(torch.from_numpy(item), dd / "item_emb.pt")
            torch.save(torch.from_numpy(np.concatenate([item, item[::-1]], axis=1)),
                       dd / "input_emb.pt")
            torch.save(torch.from_numpy(
                rng.standard_normal((n_items, d_item)).astype(np.float32)),
                dd / "content_emb.pt")
        datasets[name] = {"seq": seq_p, "item_emb": dd / "item_emb.pt",
                          "input_emb": dd / "input_emb.pt", "n_items": n_items}
    return datasets


@torch.no_grad()
def split_metrics(model: FrozenQueryTower, split, which: str, device: str = "cpu",
                  chunk: int = 512) -> dict:
    """Full-corpus max-IP HR/NDCG of ``split.<which>`` (mirrors train_j5a)."""
    q = encode_split_queries(model, split, which=which, device=device)   # [n, K, D]
    tgts = torch.tensor([t for _h, t in getattr(split, which)], dtype=torch.long)
    n = len(tgts)
    hit = {k: np.zeros(n, np.float32) for k in KS}
    dcg = {k: np.zeros(n, np.float32) for k in KS}
    item = model.item.cpu()
    for s in range(0, n, chunk):
        sc = torch.einsum("bkd,nd->bkn", q[s: s + chunk], item).max(dim=1).values
        rank = (sc > sc.gather(1, tgts[s: s + chunk].unsqueeze(1))).sum(dim=1).numpy()
        for k in KS:
            h = rank < k
            hit[k][s: s + chunk] = h
            dcg[k][s: s + chunk] = np.where(h, 1.0 / np.log2(rank + 2), 0.0)
    return {f"HR@{k}": round(float(hit[k].mean()), 4) for k in KS} | \
           {f"NDCG@{k}": round(float(dcg[k].mean()), 4) for k in KS}


def main() -> None:
    ap = argparse.ArgumentParser(description="I36 joint multi-dataset query tower")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/artifacts/i36_joint_s<seed>")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU plumbing test on a synthetic 3-dataset fixture "
                         f"(under {SMOKE_ROOT}); ~50 steps, asserts loss decreases "
                         "and the trunk checkpoint round-trips per dataset")
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        datasets = make_smoke_fixture(SMOKE_ROOT)
        args.steps = min(args.steps, 50)
        args.batch = min(args.batch, 16)
        args.device = "cpu"
        out_dir = Path(args.out_dir) if args.out_dir else SMOKE_ROOT / f"s{args.seed}"
        print(f"[i36] SMOKE — synthetic fixture at {SMOKE_ROOT / 'fixture'}", flush=True)
    else:
        datasets = DATASETS
        out_dir = Path(args.out_dir) if args.out_dir else \
            ROOT / f"outputs/artifacts/i36_joint_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(datasets)

    # per-dataset frozen sides + protocol-safe training pairs (include-val, I24 recipe)
    item_np, input_np, item_t, input_t, splits, ds = {}, {}, {}, {}, {}, {}
    for nm, cfg in datasets.items():
        item_np[nm] = l2(torch.load(cfg["item_emb"], map_location="cpu",
                                    weights_only=True).float().numpy())
        input_np[nm] = torch.load(cfg["input_emb"], map_location="cpu",
                                  weights_only=True).float().numpy()
        assert item_np[nm].shape[0] == cfg["n_items"], (nm, item_np[nm].shape)
        item_t[nm] = torch.from_numpy(item_np[nm]).to(args.device)
        input_t[nm] = torch.from_numpy(input_np[nm]).to(args.device)
        splits[nm] = protocol.build(str(cfg["seq"]), n_items=cfg["n_items"],
                                    max_len=MAX_LEN)
        ds[nm] = ChnDeDataset(splits[nm], max_len=MAX_LEN, include_val=True)
        print(f"[i36] {nm}: n_items={cfg['n_items']} train_pairs={len(ds[nm])} "
              f"D={item_np[nm].shape[1]} D_in={input_np[nm].shape[1]}", flush=True)
    d_in = {input_np[nm].shape[1] for nm in names}
    d_item = {item_np[nm].shape[1] for nm in names}
    assert len(d_in) == 1 and len(d_item) == 1, \
        f"trunk needs shared dims, got D_in={d_in} D={d_item}"

    # one trunk; the frozen item/input buffers are SWAPPED per batch (registered
    # buffers accept plain reassignment, and save_checkpoint never serialises them)
    model = FrozenQueryTower(item_np[names[0]], d=D, n_layers=LAYERS, n_heads=HEADS,
                             d_ff=D_FF, max_len=MAX_LEN, tau=TAU, seed=args.seed,
                             n_query_vectors=N_QUERY, input_emb=input_np[names[0]])
    model.to(args.device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warmup = max(int(args.steps * 0.05), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(args.steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rngs = {nm: np.random.default_rng(args.seed + di) for di, nm in enumerate(names)}
    losses: dict[str, list[float]] = {nm: [] for nm in names}
    ckpt_path = out_dir / "j5.pt"
    for step in range(args.steps):
        nm = names[step % len(names)]                       # round-robin interleave
        model.item = item_t[nm]                             # swap frozen sides
        model.input_item = input_t[nm]
        idxs = rngs[nm].choice(len(ds[nm]), size=min(args.batch, len(ds[nm])),
                               replace=False)
        b = collate_with_forbidden(ds[nm], idxs, MAX_LEN)
        loss = model(b["hist_rows"].to(args.device), b["hist_mask"].to(args.device),
                     b["tgt_rows"].to(args.device), b["forbidden_cols"].to(args.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses[nm].append(float(loss.item()))
        if (step + 1) % 150 == 0:
            avg = {n_: round(sum(l[-50:]) / max(len(l[-50:]), 1), 4)
                   for n_, l in losses.items()}
            print(f"[i36] step {step + 1}/{args.steps} loss {avg} "
                  f"lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]",
                  flush=True)
        if (step + 1) % 1000 == 0:
            save_checkpoint(model, str(ckpt_path), step=step + 1)
    save_checkpoint(model, str(ckpt_path), step=args.steps)

    # verify the SINGLE trunk checkpoint round-trips against EVERY dataset's tables,
    # and report inner-val metrics per dataset (include-val => CONTAMINATED, selection
    # happens in i36_gate.py on the inner pseudo-test; these are descriptive only)
    metrics = {}
    for nm in names:
        m = load_checkpoint(str(ckpt_path), item_np[nm], input_emb=input_np[nm])
        metrics[nm] = {"val_CONTAMINATED": split_metrics(m, splits[nm], "val",
                                                         device=args.device)}
        print(f"[i36] {nm} inner-val (contaminated): {metrics[nm]}", flush=True)

    first = {nm: float(np.mean(losses[nm][:5])) for nm in names}
    last = {nm: float(np.mean(losses[nm][-5:])) for nm in names}
    result = {"job": "I36_joint", "seed": args.seed, "steps": args.steps,
              "batch": args.batch, "lr": args.lr, "smoke": args.smoke,
              "interleave": "round_robin", "datasets": names,
              "arch": {"d": D, "layers": LAYERS, "heads": HEADS, "d_ff": D_FF,
                       "tau": TAU, "n_query": N_QUERY, "include_val": True},
              "loss_first5": {nm: round(first[nm], 4) for nm in names},
              "loss_last5": {nm: round(last[nm], 4) for nm in names},
              "metrics": metrics, "ckpt": str(ckpt_path),
              "wall_min": round((time.time() - t0) / 60, 1)}
    (out_dir / "i36_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[i36] done in {result['wall_min']} min -> {out_dir}", flush=True)
    if args.smoke:
        assert all(last[nm] < first[nm] for nm in names), \
            f"SMOKE FAIL: loss did not decrease ({first} -> {last})"
        print(f"[i36] SMOKE PASS: loss {first} -> {last}; trunk reloads for "
              f"all {len(names)} datasets", flush=True)


if __name__ == "__main__":
    main()
