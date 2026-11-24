"""CLOSE-2 — SASRec same-protocol baseline (RecSys 2027 repair program, hole 2).

Faithful SASRec (Kang & McAuley, ICDM 2018) trained on the SAME sequences the backbone
gives (LIGER-style {Dataset}.txt, capped to protocol max_len=20) and scored FULL-CORPUS
on the real test split through src/semplan/eval/protocol.py — the single enforced
evaluator. No 100-negative sampled eval anywhere.

Architecture per the paper's Amazon defaults: item embedding + learned positional
embedding, 2 self-attention blocks, 1 head, d=50, dropout 0.5, shared item embedding
as the output layer. Training = per-position next-item BCE with 1 uniformly sampled
negative per position (negatives excluded from the user's own item set), Adam lr 1e-3,
batch 128. Per-position targets on s[:-2] are exactly the protocol's train_pairs
(predict s[j-1] from s[:j-1], j=2..n-2) — asserted in --smoke.

Model selection: protocol VAL HR@10 (checked every --eval_every epochs, patience-based);
TEST is scored once, at the end, with the best-val checkpoint.

Outputs (no ledger / RESULTS.md writes — CLOSE-2 scoring is handled by the prereg):
  outputs/artifacts/close2_sasrec_{ds}_s{seed}/sasrec_best.pt
  results/close2_sasrec_{ds}_s{seed}.json           (protocol.evaluate dict + config)
  results/close2_sasrec_{ds}_s{seed}_per_user.npz   (hit5,dcg5,hit10,dcg10,cold_mask
                                                              in split.test order — pairs with
                                                              i24/i30 per-user arrays)

CLI:
  python scripts/close2_sasrec.py --dataset beauty --seed 42 --device cuda:0
  python scripts/close2_sasrec.py --smoke          # CPU fixture, ~1 min
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402

DATASETS = {
    "beauty": dict(seq=ROOT / "data/amazon_beauty_backbone/Beauty.txt", n_items=12101),
    "toys": dict(seq=ROOT / "data/amazon_toys_backbone/Toys_and_Games.txt", n_items=11924),
    "sports": dict(seq=ROOT / "data/amazon_sports_backbone/Sports_and_Outdoors.txt",
                   n_items=18357),
    # EXPAND (next paper): same protocol, same evaluator, new datasets
    "instruments": dict(seq=ROOT / "data/amazon_instruments_backbone/Musical_Instruments.txt",
                        n_items=9922),
    "yelp": dict(seq=ROOT / "data/yelp_backbone/Yelp.txt", n_items=20033),
}
KS = (5, 10)


# --------------------------------------------------------------------------- model
class PointWiseFFN(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, kernel_size=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):  # [B, L, d]
        h = x.transpose(-1, -2)
        h = self.drop2(self.conv2(self.relu(self.drop1(self.conv1(h)))))
        return h.transpose(-1, -2) + x  # residual


class SASRec(nn.Module):
    """Standard SASRec; item ids are 1-indexed internally, 0 = pad."""

    def __init__(self, n_items: int, maxlen: int, d: int = 50, n_blocks: int = 2,
                 n_heads: int = 1, dropout: float = 0.5):
        super().__init__()
        self.n_items = n_items
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                                         batch_first=True)
                                   for _ in range(n_blocks)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks)])
        self.ffn = nn.ModuleList([PointWiseFFN(d, dropout) for _ in range(n_blocks)])
        self.last_ln = nn.LayerNorm(d, eps=1e-8)
        nn.init.xavier_normal_(self.item_emb.weight.data)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.xavier_normal_(self.pos_emb.weight.data)

    def seq2feats(self, seq):  # seq: [B, L] int64 (0 = pad, left-padded)
        L = seq.shape[1]
        x = self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5)
        x = x + self.pos_emb.weight[:L].unsqueeze(0)
        x = self.emb_drop(x)
        pad_mask = seq == 0                              # [B, L]
        x = x * (~pad_mask).unsqueeze(-1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device),
                            diagonal=1)
        for ln_a, att, ln_f, ffn in zip(self.attn_ln, self.attn, self.ffn_ln, self.ffn):
            q = ln_a(x)
            a, _ = att(q, x, x, attn_mask=causal, need_weights=False)
            x = q + a
            x = ffn(ln_f(x))
            x = x * (~pad_mask).unsqueeze(-1)
        return self.last_ln(x)                           # [B, L, d]

    def forward(self, seq, pos, neg):
        """BCE logits: seq/pos/neg all [B, L], 1-indexed, 0 = pad."""
        feats = self.seq2feats(seq)
        pos_logits = (feats * self.item_emb(pos)).sum(-1)
        neg_logits = (feats * self.item_emb(neg)).sum(-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def score_full(self, seq):
        """Final-position scores over the FULL corpus: [B, n_items] (col = protocol row)."""
        feats = self.seq2feats(seq)[:, -1]               # [B, d]
        return feats @ self.item_emb.weight[1:].T        # rows 0..n_items-1


# --------------------------------------------------------------------------- data
def training_matrix(split: protocol.LOOSplit, maxlen: int):
    """Per-user SASRec training tensors from the protocol's train side (s[:-2]).

    For each user sequence s the training sub-sequence is t = s[:-2]; input t[:-1],
    label t[1:] (right-aligned into maxlen). This is exactly the multiset of
    protocol.train_pairs (predict s[j-1] from s[:j-1], j=2..n-2).
    """
    train_seqs = [list(h) for h, _t in split.val]        # s[:-2] per user with n>=3
    train_seqs = [s for s in train_seqs if len(s) >= 2]  # need >=1 (input, label) pair
    n = len(train_seqs)
    seq = np.zeros((n, maxlen), dtype=np.int64)
    pos = np.zeros((n, maxlen), dtype=np.int64)
    user_items = []
    for i, s in enumerate(train_seqs):
        inp, lab = s[:-1][-maxlen:], s[1:][-maxlen:]
        seq[i, maxlen - len(inp):] = np.asarray(inp) + 1   # 1-indexed, 0 = pad
        pos[i, maxlen - len(lab):] = np.asarray(lab) + 1
        user_items.append(np.asarray(sorted(set(s)), dtype=np.int64) + 1)
    n_pairs = int((pos > 0).sum())
    return seq, pos, user_items, n_pairs


def sample_negatives(pos: np.ndarray, user_items: list, n_items: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Uniform negatives excluding the user's own item set (paper's random_neq)."""
    neg = rng.integers(1, n_items + 1, size=pos.shape)
    neg[pos == 0] = 0
    for i, own in enumerate(user_items):
        row = neg[i]
        mask = (row > 0) & np.isin(row, own)
        while mask.any():
            row[mask] = rng.integers(1, n_items + 1, size=int(mask.sum()))
            mask = (row > 0) & np.isin(row, own)
    return neg


def pad_histories(events, maxlen: int) -> np.ndarray:
    out = np.zeros((len(events), maxlen), dtype=np.int64)
    for i, (h, _t) in enumerate(events):
        h = h[-maxlen:]
        out[i, maxlen - len(h):] = np.asarray(h) + 1
    return out


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def ranked_lists(model: SASRec, events, maxlen: int, device, topk: int,
                 batch: int = 256) -> list:
    """Full-corpus top-k ranked item rows per event, batched."""
    model.eval()
    seqs = torch.from_numpy(pad_histories(events, maxlen))
    out = []
    for i in range(0, len(seqs), batch):
        scores = model.score_full(seqs[i:i + batch].to(device))
        _, idx = torch.topk(scores, k=topk, dim=1)
        out.extend(idx.cpu().numpy().tolist())
    return out


def protocol_eval(model: SASRec, split: protocol.LOOSplit, which: str, device):
    """Score through protocol.evaluate (THE evaluator) using precomputed full-corpus lists."""
    events = getattr(split, which)
    lists = ranked_lists(model, events, model.maxlen, device, topk=max(KS))
    it = iter(lists)
    metrics = protocol.evaluate(lambda h, e: next(it), split, ks=KS, which=which)
    return metrics, lists


def per_user_arrays(lists, events, cold_idx):
    n = len(events)
    arr = {f"hit{k}": np.zeros(n, dtype=np.float32) for k in KS}
    arr.update({f"dcg{k}": np.zeros(n, dtype=np.float32) for k in KS})
    for i, (_h, tgt) in enumerate(events):
        for k in KS:
            h, g = protocol.dcg_hit(lists[i], tgt, k)
            arr[f"hit{k}"][i] = h
            arr[f"dcg{k}"][i] = g
    arr["cold_mask"] = np.array([i in cold_idx for i in range(n)], dtype=bool)
    return arr


# --------------------------------------------------------------------------- train
def train(args, seq_file: str, n_items: int, out_ckpt: Path | None):
    t0 = time.time()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    split = protocol.build(seq_file, n_items=n_items, max_len=args.maxlen)
    seq_np, pos_np, user_items, n_pairs = training_matrix(split, args.maxlen)
    assert n_pairs == len(split.train_pairs), (
        f"training targets ({n_pairs}) must equal protocol.train_pairs "
        f"({len(split.train_pairs)})")
    print(f"[close2-sasrec] users(train)={len(seq_np)} train_pairs={n_pairs} "
          f"val={len(split.val)} test={len(split.test)} cold={len(split.cold_test_idx)} "
          f"n_items={n_items} [{time.time()-t0:.0f}s]", flush=True)

    model = SASRec(n_items, args.maxlen, d=args.d, n_blocks=args.blocks,
                   n_heads=args.heads, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    use_ce = getattr(args, "loss", "bce") == "ce"
    seq_t = torch.from_numpy(seq_np)
    pos_t = torch.from_numpy(pos_np)

    best = {"val_hr10": -1.0, "epoch": -1, "state": None}
    n_users = len(seq_np)
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not use_ce:
            neg_np = sample_negatives(pos_np, user_items, n_items, rng)
            neg_t = torch.from_numpy(neg_np)
        perm = rng.permutation(n_users)
        tot, nb = 0.0, 0
        for i in range(0, n_users, args.batch):
            idx = perm[i:i + args.batch]
            s = seq_t[idx].to(device)
            p = pos_t[idx].to(device)
            if use_ce:
                # full cross-entropy over all N items per position (Klenitskiy
                # loss swap; same architecture/data/selection as the BCE run)
                feats = model.seq2feats(s)                          # [B, L, d]
                logits = feats @ model.item_emb.weight[1:].T        # [B, L, N]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, n_items), (p - 1).reshape(-1),
                    ignore_index=-1)
            else:
                ng = neg_t[idx].to(device)
                pos_logits, neg_logits = model(s, p, ng)
                mask = (p > 0).float()
                loss = (bce(pos_logits, torch.ones_like(pos_logits)) * mask).sum()
                loss = loss + (bce(neg_logits, torch.zeros_like(neg_logits)) * mask).sum()
                loss = loss / mask.sum().clamp(min=1.0)
            if args.l2_emb > 0:
                loss = loss + args.l2_emb * model.item_emb.weight.norm(2) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, _ = protocol_eval(model, split, "val", device)
            hr10 = val_metrics["overall"]["HR@10"]
            tag = ""
            if hr10 > best["val_hr10"]:
                best = {"val_hr10": hr10, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                epochs_no_improve = 0
                tag = "  *best*"
            else:
                epochs_no_improve += args.eval_every
            print(f"[close2-sasrec] epoch {epoch:4d} loss {tot/max(nb,1):.4f} "
                  f"val HR@10 {hr10:.4f}{tag} [{time.time()-t0:.0f}s]", flush=True)
            if epochs_no_improve >= args.patience:
                print(f"[close2-sasrec] early stop at epoch {epoch} "
                      f"(best val HR@10 {best['val_hr10']:.4f} @ {best['epoch']})",
                      flush=True)
                break

    assert best["state"] is not None
    model.load_state_dict(best["state"])
    if out_ckpt is not None:
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state"], "config": vars(args),
                    "n_items": n_items, "best_val_hr10": best["val_hr10"],
                    "best_epoch": best["epoch"]}, out_ckpt)
        print(f"[close2-sasrec] saved {out_ckpt}")

    test_metrics, test_lists = protocol_eval(model, split, "test", device)
    arrays = per_user_arrays(test_lists, split.test, split.cold_test_idx)
    # cross-check: per-user means reproduce protocol.evaluate aggregates
    for k in KS:
        assert round(float(arrays[f"hit{k}"].mean()), 4) == test_metrics["overall"][f"HR@{k}"]
        assert round(float(arrays[f"dcg{k}"].mean()), 4) == test_metrics["overall"][f"NDCG@{k}"]
    return split, model, best, test_metrics, arrays, time.time() - t0


# --------------------------------------------------------------------------- smoke
def smoke(args):
    """Tiny synthetic fixture: chain transitions i -> i+1 (mod n) with noise.

    Asserts: (a) per-position training targets == protocol.train_pairs count,
    (b) protocol_eval / per-user arrays are internally consistent (aggregates match),
    (c) the model actually learns the chain (HR@10 well above the 10/n_items floor).
    """
    import tempfile
    n_items, n_users, slen = 60, 300, 10
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seq_file = Path(td) / "Smoke.txt"
        with open(seq_file, "w") as f:
            for u in range(n_users):
                start = int(rng.integers(0, n_items))
                items = [(start + j) % n_items for j in range(slen)]
                # 10% noise swaps
                for j in range(slen):
                    if rng.random() < 0.10:
                        items[j] = int(rng.integers(0, n_items))
                f.write(f"{u+1} " + " ".join(str(i + 1) for i in items) + "\n")
        args.epochs, args.eval_every, args.patience = 60, 10, 60
        args.d, args.batch, args.maxlen, args.dropout = 32, 64, 20, 0.2
        split, model, best, test_metrics, arrays, dt = train(
            args, str(seq_file), n_items, out_ckpt=None)

        assert len(split.test) == n_users and len(split.val) == n_users
        for key in ("hit5", "dcg5", "hit10", "dcg10", "cold_mask"):
            assert arrays[key].shape == (n_users,), key
        hr10 = test_metrics["overall"]["HR@10"]
        floor = 10.0 / n_items
        assert hr10 > 3 * floor, f"smoke model failed to learn: HR@10={hr10} floor={floor}"
        print(f"[close2-sasrec] SMOKE PASS  test={json.dumps(test_metrics['overall'])} "
              f"(chance floor HR@10={floor:.3f})  [{dt:.0f}s]")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    # paper defaults for Amazon (d=50, 2 blocks, 1 head, dropout 0.5)
    ap.add_argument("--d", type=int, default=50)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--maxlen", type=int, default=20)  # = protocol max_len
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2_emb", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--loss", choices=["bce", "ce"], default="bce",
                    help="bce = faithful 2018 recipe (the FROZEN prereg config); "
                         "ce = full cross-entropy over all items (Klenitskiy loss swap; "
                         "descriptive row only, NOT part of the frozen plan)")
    args = ap.parse_args()
    from semplan.quick import max_epochs  # release: TT_MAX_EPOCHS cap (quick mode only)
    args.epochs = max_epochs(args.epochs)
    args.eval_every = min(args.eval_every, args.epochs)

    if args.smoke:
        smoke(args)
        return
    assert args.dataset, "--dataset required (or --smoke)"
    ds = DATASETS[args.dataset]
    suffix = "" if args.loss == "bce" else "_ce"
    tag = f"close2_sasrec{suffix}_{args.dataset}_s{args.seed}"
    ckpt = ROOT / "outputs/artifacts" / tag / "sasrec_best.pt"
    split, model, best, test_metrics, arrays, dt = train(
        args, str(ds["seq"]), ds["n_items"], out_ckpt=ckpt)

    out = {
        "experiment": "CLOSE2_sasrec" if args.loss == "bce" else "CLOSE2_sasrec_ce_descriptive",
        "dataset": args.dataset, "seed": args.seed,
        "config": {k: v for k, v in vars(args).items() if k not in ("smoke",)},
        "selection": {"criterion": "protocol val HR@10",
                      "best_val_hr10": best["val_hr10"], "best_epoch": best["epoch"]},
        "test": test_metrics,
        "checkpoint": str(ckpt),
        "runtime_sec": round(dt, 1),
    }
    res = ROOT / "results" / f"{tag}.json"
    json.dump(out, open(res, "w"), indent=2)
    npz = ROOT / "results" / f"{tag}_per_user.npz"
    np.savez(npz, **arrays)
    print(f"[close2-sasrec] test={json.dumps(test_metrics['overall'])}")
    print(f"[close2-sasrec] wrote {res} and {npz} [{dt:.0f}s]")


if __name__ == "__main__":
    main()
