"""I37 — pretrained-LM query encoder in the J5a frozen-item recipe (Beauty inner).

Hypothesis (backlog "I37 pretrained-LM query encoder", licensed by RESULTS #57):
the I18 size-sweep showed naive from-scratch scale COLLAPSES (d256 0.075 -> 60M
0.026) at 22k sequences, while the LLM-scale winners (URM-7B 0.0929) use pretrained
backbones. I37 tests the cheapest version of that class: keep the ENTIRE winning
J5a/I24 recipe (6144-d [own emb ; pred3 centroid] frozen input vectors, K=4 heads,
full-softmax CE over the frozen L2-normalised I22 item catalog, tau=0.05, forbidden
masking, include-val) and swap ONLY the trunk: the 2L/d256 transformer becomes a
pretrained HF text encoder (default sentence-transformers/all-MiniLM-L6-v2,
--lm <larger open encoder> to scale) consuming the SAME per-position item vectors
via a learned linear projection into the LM hidden size, fed as ``inputs_embeds``
with the history attention mask. The LM never sees tokens — it is used as a
pretrained sequence encoder over projected item vectors (its own absolute position
embeddings still apply on top of our J5a recency embedding).

READOUT (decision): J5a does NOT read the last position — it attention-pools with
K=4 learned pooling heads over ALL unmasked positions (frozen_query.FrozenQueryTower
.encode_queries: pool logits -> masked softmax over L -> K weighted sums -> shared
out_proj -> L2-normalise). I37 replicates THAT readout verbatim on the LM's
last_hidden_state, so the only changed component is the trunk. Serving stays
two-tower/ANN (K independent probes, max-merge) per DIR-2.

CHECKPOINT FORMAT (decision): ``<out-dir>/lmq.pt`` with the i36/save_checkpoint
pattern — state_dict EXCLUDES the frozen ``item``/``input_item`` buffers; the config
dict stores the full HF lm config (plain-JSON dict, so torch.load(weights_only=True)
still works) and load_lm_checkpoint rebuilds the LM with AutoModel.from_config
(NO hub access needed at load time: the fine-tuned LM weights are in the state_dict)
then loads with strict=False, exactly like frozen_query.load_checkpoint.

Usage (box, one job per seed):
  python scripts/i37_train_lmq.py --seed 42 \
      --out-dir outputs/artifacts/i37_lmq_s42
CPU smoke (synthetic fixture + tiny random-weight BERT config, no downloads):
  python scripts/i37_train_lmq.py --smoke --seed 42
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
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semplan.dense.chn_de import ChnDeDataset  # noqa: E402
from semplan.dense.frozen_query import (  # noqa: E402
    collate_with_forbidden, encode_split_queries,
)
from semplan.dense.tricks_query import (  # noqa: E402  (I48: T2 TED soft labels)
    build_ted_teachers, collate_ted, mine_forward_adjacency,
)
from semplan.eval import protocol  # noqa: E402

# I24/J5a recipe constants (unchanged): tau, K heads, history cap, eval Ks
TAU, N_QUERY, MAX_LEN = 0.05, 4, 20
KS = (5, 10)
DEFAULT_LM = "sentence-transformers/all-MiniLM-L6-v2"

BEAUTY = {
    "seq": ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt",
    "item_emb": ROOT / "data/amazon_beauty_backbone/"
                       "Beauty_openai-3-large_i22inner_embeddings.pt",
    "input_emb": ROOT / "data/amazon_beauty_backbone/"
                        "Beauty_openai-3-large_i24inner_input.pt",
    "n_items": 12101,
}
SMOKE_ROOT = ROOT / "outputs/tmp/i37_smoke"


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def tiny_lm_config():
    """Random-weight BERT-family config for --smoke (no hub access, seconds on CPU)."""
    from transformers import BertConfig
    return BertConfig(vocab_size=16, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=64)


class LMQueryTower(nn.Module):
    """J5a frozen-item query tower with a pretrained HF encoder as the trunk.

    Same contract as frozen_query.FrozenQueryTower (encode_queries / full_logits /
    forward(CE) / .item / .max_len), so encode_split_queries and the gate sweeps
    work unchanged. Trunk: in_proj(D_in -> H) + J5a recency embedding -> HF encoder
    via inputs_embeds -> K attention-pool heads -> out_proj(H -> D) -> L2-norm.
    """

    def __init__(self, item_emb: np.ndarray, lm, max_len: int = MAX_LEN,
                 tau: float = TAU, seed: int = 42, n_query_vectors: int = N_QUERY,
                 input_emb: np.ndarray | None = None, freeze_lm: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        item = F.normalize(torch.from_numpy(np.asarray(item_emb, np.float32)), dim=-1)
        self.register_buffer("item", item)                  # [n_items, D] FROZEN
        n_items, D = item.shape
        inp = item if input_emb is None else \
            torch.from_numpy(np.asarray(input_emb, np.float32))
        self.register_buffer("input_item", inp)             # [n_items, D_in] FROZEN
        D_in = inp.shape[1]
        self.tau, self.k, self.max_len = tau, n_query_vectors, max_len
        self.freeze_lm = freeze_lm
        self.lm = lm
        H = lm.config.hidden_size
        # plain-JSON config only (torch.load(weights_only=True) compatibility)
        self.config = {"max_len": max_len, "tau": tau, "seed": seed,
                       "n_query_vectors": n_query_vectors, "freeze_lm": freeze_lm,
                       "lm_config": lm.config.to_dict(),
                       "item_dim": D, "n_items": n_items, "input_dim": D_in}
        self.in_proj = nn.Linear(D_in, H)
        self.rec_emb = nn.Embedding(max_len, H)             # J5a recency positions
        self.pool = nn.Linear(H, n_query_vectors)           # K attention-pool heads
        self.out_proj = nn.Linear(H, D)
        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_lm:
            self.lm.eval()                                  # probe-style: no LM dropout
        return self

    def encode_queries(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor
                       ) -> torch.Tensor:
        """[B, L] histories -> [B, K, D] L2-normalised queries (J5a readout on LM)."""
        h = self.in_proj(self.input_item[hist_rows])                    # [B, L, H]
        lengths = hist_mask.sum(1, keepdim=True)
        pos = torch.arange(hist_rows.shape[1], device=hist_rows.device).unsqueeze(0)
        recency = (lengths - 1 - pos).clamp(min=0, max=self.max_len - 1)
        h = h + self.rec_emb(recency)
        h = self.lm(inputs_embeds=h,
                    attention_mask=hist_mask.long()).last_hidden_state  # [B, L, H]
        att = self.pool(h).masked_fill(~hist_mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(att, dim=1)                                   # [B, L, K]
        q = torch.einsum("blk,bld->bkd", w, h)                          # [B, K, H]
        return F.normalize(self.out_proj(q), dim=-1)                    # [B, K, D]

    def full_logits(self, q: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bkd,nd->bkn", q, self.item).max(dim=1).values

    def forward(self, hist_rows, hist_mask, tgt_rows, forbidden_cols=None,
                ted_cols=None, ted_probs=None, ted_beta: float = 0.0):
        """Full-softmax CE with forbidden masking (identical to FrozenQueryTower).
        I48: optional TED soft-label mix — code-exact copy of the tricks_query T2
        path (target = (1-beta)*onehot + beta*teacher, beta renormalised to 0 for
        examples whose teacher support is empty/fully masked)."""
        logits = self.full_logits(self.encode_queries(hist_rows, hist_mask)) / self.tau
        if forbidden_cols is not None:
            rows = torch.arange(len(logits), device=logits.device
                                ).unsqueeze(1).expand_as(forbidden_cols)
            valid = forbidden_cols >= 0
            logits[rows[valid], forbidden_cols[valid]] = float("-inf")
        if ted_cols is None or ted_beta <= 0:
            return F.cross_entropy(logits, tgt_rows)
        logp = F.log_softmax(logits, dim=-1)
        lp_tgt = logp.gather(1, tgt_rows.unsqueeze(1)).squeeze(1)
        lp_c = logp.gather(1, ted_cols.clamp(min=0))
        fin = (ted_cols >= 0) & torch.isfinite(lp_c)
        p = ted_probs * fin
        psum = p.sum(1)
        soft = (p * lp_c.masked_fill(~fin, 0.0)).sum(1) / psum.clamp(min=1e-12)
        beta = ted_beta * (psum > 0)                                 # renormalised
        return (-((1.0 - beta) * lp_tgt + beta * soft)).mean()


def save_lm_checkpoint(model: LMQueryTower, path: str, step: int = 0) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if k not in ("item", "input_item")}  # frozen, external (i36 pattern)
    torch.save({"state_dict": sd, "config": model.config, "step": step}, path)


def load_lm_checkpoint(path: str, item_emb: np.ndarray,
                       input_emb: np.ndarray | None = None) -> LMQueryTower:
    from transformers import AutoConfig, AutoModel
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = {k: v for k, v in ckpt["config"].items()
           if k not in ("item_dim", "n_items", "input_dim")}
    lm_cfg = dict(cfg.pop("lm_config"))
    lm = AutoModel.from_config(AutoConfig.for_model(lm_cfg.pop("model_type"), **lm_cfg))
    model = LMQueryTower(item_emb, lm, input_emb=input_emb, **cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.eval()


@torch.no_grad()
def split_metrics(model: LMQueryTower, split, which: str, device: str = "cpu",
                  chunk: int = 512) -> dict:
    """Full-corpus max-IP HR/NDCG of ``split.<which>`` (mirrors i36_train_joint)."""
    q = encode_split_queries(model, split, which=which, device=device)
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
    ap = argparse.ArgumentParser(description="I37 pretrained-LM query tower (Beauty)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lm", default=DEFAULT_LM,
                    help="HF encoder id (any Hugging Face encoder)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head lr (in_proj/rec_emb/pool/out_proj) — J5a value")
    ap.add_argument("--lm-lr", type=float, default=3e-5,
                    help="fine-tune lr for the pretrained LM parameter group")
    ap.add_argument("--freeze-lm", action="store_true",
                    help="probe-style: LM frozen (eval mode), only heads train")
    ap.add_argument("--ted_beta", type=float, default=0.0,
                    help="I48: TED soft-label weight (0 = plain CE; i40 semantics)")
    ap.add_argument("--ted_tau", type=float, default=0.1,
                    help="I48: TED teacher temperature (i40 semantics)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/artifacts/i37_lmq_s<seed>")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU plumbing test: synthetic 1-dataset fixture (under "
                         f"{SMOKE_ROOT}) + tiny random-weight BERT config, ~40 steps; "
                         "asserts loss decreases and the checkpoint round-trips")
    args = ap.parse_args()
    t0 = time.time()

    from i36_train_joint import make_smoke_fixture  # shared fixture builder
    if args.smoke:
        cfg = make_smoke_fixture(SMOKE_ROOT, n_datasets=1)["fake0"]
        args.steps, args.batch, args.device = min(args.steps, 40), 8, "cpu"
        out_dir = Path(args.out_dir) if args.out_dir else SMOKE_ROOT / f"s{args.seed}"
        from transformers import AutoModel
        lm = AutoModel.from_config(tiny_lm_config())
        lm_name = "tiny-random-bert (smoke)"
        print(f"[i37] SMOKE — fixture at {SMOKE_ROOT / 'fixture'}, tiny LM", flush=True)
    else:
        cfg = BEAUTY
        out_dir = Path(args.out_dir) if args.out_dir else \
            ROOT / f"outputs/artifacts/i37_lmq_s{args.seed}"
        from transformers import AutoModel
        lm = AutoModel.from_pretrained(args.lm)
        lm_name = args.lm
    out_dir.mkdir(parents=True, exist_ok=True)

    item_np = l2(torch.load(cfg["item_emb"], map_location="cpu",
                            weights_only=True).float().numpy())
    input_np = torch.load(cfg["input_emb"], map_location="cpu",
                          weights_only=True).float().numpy()
    assert item_np.shape[0] == cfg["n_items"], item_np.shape
    split = protocol.build(str(cfg["seq"]), n_items=cfg["n_items"], max_len=MAX_LEN)
    ds = ChnDeDataset(split, max_len=MAX_LEN, include_val=True)   # I24 recipe
    print(f"[i37] lm={lm_name} n_items={cfg['n_items']} train_pairs={len(ds)} "
          f"D={item_np.shape[1]} D_in={input_np.shape[1]} "
          f"H={lm.config.hidden_size}", flush=True)

    ted_teachers = None
    if args.ted_beta > 0:                                            # I48 TED
        adj = mine_forward_adjacency(str(cfg["seq"]), cfg["n_items"])
        ted_teachers = build_ted_teachers(adj, cfg["n_items"], ted_tau=args.ted_tau)
        n_rows = sum(1 for c, _p in ted_teachers if len(c))
        print(f"[i37] TED teachers: beta={args.ted_beta} tau={args.ted_tau} "
              f"nonzero rows {n_rows}/{cfg['n_items']}", flush=True)

    model = LMQueryTower(item_np, lm, max_len=MAX_LEN, tau=TAU, seed=args.seed,
                         n_query_vectors=N_QUERY, input_emb=input_np,
                         freeze_lm=args.freeze_lm)
    model.to(args.device).train()
    lm_params = [p for p in model.lm.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("lm.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if lm_params:
        groups.append({"params": lm_params, "lr": args.lm_lr})
    n_lm_train = sum(p.numel() for p in lm_params)
    n_head = sum(p.numel() for p in head_params)
    n_lm_frozen = sum(p.numel() for p in model.lm.parameters()) - n_lm_train
    print(f"[i37] params: heads {n_head:,} + LM trainable {n_lm_train:,} "
          f"(LM frozen {n_lm_frozen:,})", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warmup = max(int(args.steps * 0.05), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(args.steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    ckpt_path = out_dir / "lmq.pt"
    for step in range(args.steps):
        idxs = rng.choice(len(ds), size=min(args.batch, len(ds)), replace=False)
        b = collate_with_forbidden(ds, idxs, MAX_LEN)
        tc = tp = None
        if ted_teachers is not None:                                 # I48 TED
            tc, tp = collate_ted(ted_teachers, b["hist_rows"], b["hist_mask"])
            tc, tp = tc.to(args.device), tp.to(args.device)
        loss = model(b["hist_rows"].to(args.device), b["hist_mask"].to(args.device),
                     b["tgt_rows"].to(args.device), b["forbidden_cols"].to(args.device),
                     ted_cols=tc, ted_probs=tp, ted_beta=args.ted_beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            print(f"[i37] step {step + 1}/{args.steps} loss {avg:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]",
                  flush=True)
        if (step + 1) % 1000 == 0:
            save_lm_checkpoint(model, str(ckpt_path), step=step + 1)
    save_lm_checkpoint(model, str(ckpt_path), step=args.steps)

    # round-trip check: reloaded model must reproduce the trained queries exactly
    m2 = load_lm_checkpoint(str(ckpt_path), item_np, input_emb=input_np)
    model.eval()
    with torch.no_grad():
        b = collate_with_forbidden(ds, np.arange(min(8, len(ds))), MAX_LEN)
        q1 = model.cpu().encode_queries(b["hist_rows"], b["hist_mask"])
        q2 = m2.encode_queries(b["hist_rows"], b["hist_mask"])
    assert torch.allclose(q1, q2, atol=1e-5), "checkpoint round-trip mismatch"
    print("[i37] checkpoint round-trip OK", flush=True)

    # inner-val metrics (include-val => CONTAMINATED; selection lives in i37_gate.py)
    val_m = split_metrics(m2, split, "val", device="cpu")
    print(f"[i37] inner-val (contaminated): {val_m}", flush=True)

    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    result = {"job": "I37_lmq", "seed": args.seed, "steps": args.steps,
              "batch": args.batch, "lr": args.lr, "lm_lr": args.lm_lr,
              "lm": lm_name, "freeze_lm": args.freeze_lm, "smoke": args.smoke,
              "ted_beta": args.ted_beta, "ted_tau": args.ted_tau,
              "arch": {"trunk": "pretrained_lm", "hidden": lm.config.hidden_size,
                       "tau": TAU, "n_query": N_QUERY, "max_len": MAX_LEN,
                       "include_val": True, "readout": "J5a_k_head_attention_pool"},
              "params_trainable": n_head + n_lm_train,      # I19 size axis
              "params_head": n_head, "params_lm_trainable": n_lm_train,
              "params_lm_frozen": n_lm_frozen,
              "loss_first5": round(first, 4), "loss_last5": round(last, 4),
              "val_CONTAMINATED": val_m, "ckpt": str(ckpt_path),
              "wall_min": round((time.time() - t0) / 60, 1)}
    (out_dir / "i37_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[i37] done in {result['wall_min']} min -> {out_dir}", flush=True)
    if args.smoke:
        assert last < first, f"SMOKE FAIL: loss did not decrease ({first} -> {last})"
        print(f"[i37] SMOKE PASS: loss {first:.4f} -> {last:.4f}; "
              "checkpoint round-trips", flush=True)


if __name__ == "__main__":
    main()
