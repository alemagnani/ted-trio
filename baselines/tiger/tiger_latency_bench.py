#!/usr/bin/env python
"""TIGER batch-1 decode latency bench — RECONSTRUCTION of the ad-hoc on-box bench
behind results/close2_tiger_latency.json (P1 §4 efficiency: 739 ms/query
GPU, 1.7 s CPU). The original script was never kept; this one is rebuilt from the
artifact's decode_config + env block and exercises the SAME decode path as the
reported eval (baselines/tiger/j1_dump.py -> vendor third_party/liger TIGER.generate):

  unconstrained beam-20 (num_beams=20, num_return_sequences=20, max_new_tokens=4,
  use_cache=True) with post-hoc SID->item validity mapping, batch_size=1,
  GPU: fp16 autocast (as in j1_dump.dump_tiger); CPU: fp32, torch threads=1.

Data pipeline (splits, SID expansion, sid2item map) is byte-identical to
j1_dump.build_data (vendor process_data_split + load_data, seed 42), just with
explicit file paths instead of the on-box LIGER_ROOT layout. Queries are the
test split, seen part first, in original order (the artifact's sanity_decodes
targets 5/11/19/24/32 for query_idx 0..4 match this ordering).

Output: JSON in the same shape as close2_tiger_latency.json, with
"reconstructed": true. Validated 2026-08-13 on an idle RTX 4090 (torch 2.5.1+cu124,
transformers 4.39.3): sanity_decodes match the booked artifact exactly; the booked
739 ms GPU / 1.7 s CPU were co-tenant upper bounds (idle reruns:
results/close2_tiger_latency_reconstructed_{gpu,cpu}.json).

Usage:
  python baselines/tiger/tiger_latency_bench.py --device cpu --n-queries 100
  python baselines/tiger/tiger_latency_bench.py --device cuda --n-queries 500
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIGER_ROOT = ROOT / "third_party" / "liger"
sys.path.insert(0, str(ROOT / "src"))       # semplan.*
sys.path.insert(0, str(LIGER_ROOT))         # utils, src.*, ID_generation.*

from semplan.common.seed import set_seed  # noqa: E402

# ---------------------------------------------------------------- constants
# Same values as baselines/tiger/j1_dump.py (and the artifact's decode_config).
SEED = 42
CODEBOOK_SIZE = 256
N_POSITIONS = 102
MAX_ITEMS_PER_SEQ = 20
NUM_BEAMS = 20

DATASETS = {
    "beauty": dict(name="Beauty", n_items=12101),
    "toys": dict(name="Toys_and_Games", n_items=11924),
    "sports": dict(name="Sports_and_Outdoors", n_items=18357),
    "instruments": dict(name="Musical_Instruments", n_items=9922),
    "yelp": dict(name="Yelp", n_items=20033),
}

# T5 section of third_party/liger/configs/dataset/amazon.yaml (as in j1_dump.py)
T5_CFG = dict(
    encoder_layers=6, decoder_layers=6, d_model=128, d_ff=1024, num_heads=6,
    d_kv=64, dropout_rate=0.2, feed_forward_proj="relu", initializer_factor=0.02,
)

# configs/method/base.yaml flattened as run.py does (TIGER = method base)
METHOD_BASE = dict(
    include_user_id=False, use_id="sid",
    flag_add_input_embedding=False, flag_use_output_embedding=False,
    embedding_loss_weight=0, sid_loss_weight=1, evaluation_method="default",
    embedding_head_dict=dict(
        use_new_init=False, embed_target="ground_truth", embed_proj_type="mlp",
    ),
)


# ---------------------------------------------------------------- data / model
def build_data(args):
    """Byte-identical to j1_dump.build_data: vendor split + SID expansion."""
    import torch
    from ID_generation.utils import process_data_split
    from src.load_data import load_data
    from utils import set_seed as vendor_set_seed

    vendor_set_seed(SEED)
    n_items = DATASETS[args.dataset]["n_items"]
    cfg = {"dataset": {"max_items_per_seq": MAX_ITEMS_PER_SEQ}}
    id_split, user_sequence = process_data_split(
        cfg, args.seq, args.id2meta, is_steam=False)
    item_embedding = torch.load(
        args.item_emb, map_location="cpu", weights_only=False).float()
    assert item_embedding.shape == (n_items, 768), \
        f"bad embedding shape {tuple(item_embedding.shape)}"
    if args.lean_mem:
        # Memory-lean substitute: vendor load_data_helper materialises a
        # [20, 768] fp32 input_embeddings row per TRAINING example (~10 GB on
        # Beauty), but the TIGER decode path never reads them (method base:
        # flag_add_input_embedding=False -> model_forward uses only
        # input_sids/attention_mask_sids/labels_*). A [n_items, 1] zero stand-in
        # leaves splits, SID expansion and the decode 100% unchanged.
        item_embedding = torch.zeros((n_items, 1))

    (_train, val_data, test_data, unseen_val_data, unseen_test_data,
     _seen, _vu, _tu, max_last_semantic_ids,
     n_semantic_codebook, n_codebook, item2sid) = load_data(
        args.sids, user_sequence,
        id_split["unseen_val"], id_split["unseen_test"], id_split["seen"],
        item_embedding, METHOD_BASE,
        max_length=N_POSITIONS, codebook_size=CODEBOOK_SIZE,
        max_items_per_seq=MAX_ITEMS_PER_SEQ,
    )
    assert n_codebook == 4 and n_semantic_codebook == 3
    assert item2sid.shape == (n_items, 4)
    last_codebook_size = max(max_last_semantic_ids, CODEBOOK_SIZE)
    vocab_size = CODEBOOK_SIZE * n_semantic_codebook + last_codebook_size + 2

    sid2item = {tuple(int(t) for t in item2sid[i]): i + 1 for i in range(n_items)}
    assert len(sid2item) == n_items, "expanded SIDs must be unique"

    return dict(
        sid2item=sid2item, vocab_size=int(vocab_size), n_codebook=n_codebook,
        n_semantic_codebook=n_semantic_codebook,
        splits={"val": (val_data, unseen_val_data),
                "test": (test_data, unseen_test_data)},
    )


def build_model(args, data, device):
    """Byte-identical to j1_dump.build_model(which='tiger')."""
    import torch
    from transformers import T5Config
    from src.tiger import TIGER

    vocab = data["vocab_size"]
    model_config = T5Config(
        num_layers=T5_CFG["encoder_layers"],
        num_decoder_layers=T5_CFG["decoder_layers"],
        d_model=T5_CFG["d_model"], d_ff=T5_CFG["d_ff"],
        num_heads=T5_CFG["num_heads"], d_kv=T5_CFG["d_kv"],
        dropout_rate=T5_CFG["dropout_rate"], vocab_size=vocab,
        pad_token_id=0, eos_token_id=int(vocab - 1), decoder_start_token_id=0,
        feed_forward_proj=T5_CFG["feed_forward_proj"], n_positions=N_POSITIONS,
        layer_norm_epsilon=1e-8,
        initializer_factor=T5_CFG["initializer_factor"],
    )
    model = TIGER(
        config=model_config, n_semantic_codebook=data["n_semantic_codebook"],
        max_items_per_seq=MAX_ITEMS_PER_SEQ,
        flag_use_output_embedding=METHOD_BASE["flag_use_output_embedding"],
        flag_use_learnable_text_embed=METHOD_BASE["flag_add_input_embedding"],
        embedding_head_dict=METHOD_BASE["embedding_head_dict"],
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:  # ckpt.pt-style wrapper
        state = state["model_state_dict"]
    assert state["shared.weight"].shape[0] == vocab, (
        f"vocab mismatch: ckpt {state['shared.weight'].shape[0]} vs data {vocab}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    n_params = count_unique_params(state)
    print(f"[bench] loaded {args.ckpt} (vocab={vocab}, params={n_params})")
    return model, n_params


def count_unique_params(state) -> int:
    """Unique-tensor param count (T5 weight tying), as in close3_serving_bench."""
    seen, tot = set(), 0
    for v in state.values():
        if not hasattr(v, "data_ptr") or v.data_ptr() in seen:
            continue
        seen.add(v.data_ptr())
        tot += v.numel()
    return int(tot)


# ---------------------------------------------------------------- bench core
def batch1_queries(data, split, limit):
    """First `limit` batch-1 queries: seen part first, then unseen, each in
    original order — the exact query ordering of j1_dump.split_loaders."""
    from torch.utils.data import DataLoader
    from utils import CustomDataset

    seen_d, unseen_d = data["splits"][split]
    out = []
    for d, cold in [(seen_d, False), (unseen_d, True)]:
        if len(d["input_ids"]) == 0:
            continue
        for batch in DataLoader(CustomDataset(d), batch_size=1, shuffle=False):
            out.append((batch, cold))
            if len(out) >= limit:
                return out
    return out


def decode_one(model, batch, device, data, use_fp16_autocast):
    """One batch-1 unconstrained beam-20 decode + post-hoc SID->item mapping.
    Mirrors j1_dump.dump_tiger's generate call (minus the logprob dump)."""
    import contextlib

    import torch
    import torch.nn.functional as F
    from src.evaluation import model_forward

    n_codebook = data["n_codebook"]
    _, input_kwargs = model_forward(
        model, batch, device, n_codebook, METHOD_BASE, skip_forward=True)
    amp = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
           if use_fp16_autocast else contextlib.nullcontext())
    with amp:
        out = model.generate(
            **input_kwargs, num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS, max_new_tokens=n_codebook,
            use_cache=True,
        )
    seqs = out[:, 1:1 + n_codebook]
    if seqs.shape[1] < n_codebook:  # early eos
        seqs = F.pad(seqs, (0, n_codebook - seqs.shape[1]), value=0)
    seqs = seqs.cpu()
    items, valid = [], []
    for r in range(NUM_BEAMS):
        item = data["sid2item"].get(tuple(int(t) for t in seqs[r]))
        items.append(item)
        valid.append(item is not None)
    target = int(batch["labels_ids"][0, 0])
    return items, valid, target


def pct(xs, p):
    import numpy as np
    return float(np.percentile(np.asarray(xs), p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="beauty")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/artifacts/j1/tiger_ckpt_best.pt"),
                    help="TIGER checkpoint (raw state dict or {'model_state_dict': ...})")
    ap.add_argument("--sids", default=str(ROOT / "outputs/artifacts/j1/Beauty_sentence-t5-xl_42.pkl"),
                    help="semantic-ID pkl (LIGER ID_generation output)")
    ap.add_argument("--seq", default=str(ROOT / "outputs/artifacts/j1/Beauty.txt"),
                    help="LIGER-preprocessed user-sequence txt")
    ap.add_argument("--id2meta", default=str(
        ROOT / "outputs/artifacts/j1/Beauty_title_price_brand_categories_amazon_id2meta.json"))
    ap.add_argument("--item-emb", default=str(
        ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"),
        help="sentence-t5-xl item embeddings [n_items, 768]")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--n-queries", "--n_queries", dest="n_queries", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads for --device cpu (original bench: 1)")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--lean-mem", action=argparse.BooleanOptionalAction, default=True,
                    help="replace item embeddings with a zero stand-in during data "
                         "staging (TIGER decode never reads them; saves ~10 GB RAM; "
                         "decode-path-neutral). --no-lean-mem for the original "
                         "on-box behavior.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(
        ROOT / "results/close2_tiger_latency_reconstructed.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers

    set_seed(args.seed)
    on_gpu = args.device == "cuda"
    if on_gpu:
        assert torch.cuda.is_available(), "--device cuda but CUDA is unavailable"
    else:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    t0 = time.time()
    data = build_data(args)
    model, n_params = build_model(args, data, device)
    queries = batch1_queries(data, args.split,
                             max(args.n_queries, args.warmup))
    assert len(queries) >= args.n_queries, \
        f"split has only {len(queries)} queries"
    print(f"[bench] data+model ready in {time.time()-t0:.0f}s; "
          f"{len(queries)} batch-1 queries staged ({args.split} split)")

    def sync():
        if on_gpu:
            torch.cuda.synchronize()

    # warmup (untimed): first `warmup` queries
    for batch, _cold in queries[:args.warmup]:
        decode_one(model, batch, device, data, use_fp16_autocast=on_gpu)
    sync()

    # timed: queries[0:n] from the start, so query_idx 0 matches the artifact's
    # sanity_decodes (original bench: seen-test query 0 -> target_item 5 on Beauty)
    lat_ms, ranks, sanity = [], [], []
    gpu_samples = []
    for qidx, (batch, _cold) in enumerate(queries[:args.n_queries]):
        sync()
        t = time.perf_counter()
        items, valid, target = decode_one(
            model, batch, device, data, use_fp16_autocast=on_gpu)
        sync()
        lat_ms.append((time.perf_counter() - t) * 1e3)
        rank = next((r for r, it in enumerate(items) if it == target), None)
        ranks.append(rank)
        if qidx < 5:
            sanity.append(dict(query_idx=qidx, target_item=target,
                               top5_items=items[:5], top5_valid=valid[:5]))
        if on_gpu and qidx % max(1, args.n_queries // 5) == 0:
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                u, m = r.stdout.strip().split("\n")[0].split(",")
                gpu_samples.append({"util_pct": int(u), "mem_used_mib": int(m)})
            except Exception:
                pass

    def hr(k):
        return float(np.mean([1.0 if (r is not None and r < k) else 0.0
                              for r in ranks]))

    stats = {
        "n_queries": args.n_queries,
        "warmup_queries": args.warmup,
        "mean_ms": float(np.mean(lat_ms)),
        "median_ms": float(np.median(lat_ms)),
        "p90_ms": pct(lat_ms, 90),
        "p99_ms": pct(lat_ms, 99),
        "min_ms": float(np.min(lat_ms)),
        "max_ms": float(np.max(lat_ms)),
        "HR@5_timed_subset": hr(5),
        "HR@10_timed_subset": hr(10),
        "gpu_util_samples_during": gpu_samples if on_gpu else None,
    }
    block_key = "gpu_batch1" if on_gpu else f"cpu_batch1_threads{args.threads}"

    out = {
        "experiment": "close2_tiger_latency",
        "reconstructed": True,
        "reconstruction_note": (
            "script rebuilt 2026-08 from close2_tiger_latency.json's "
            "decode_config (original ad-hoc on-box bench never kept); decode "
            "path identical to baselines/tiger/j1_dump.py dump_tiger; NOT yet "
            "validated against the booked 739 ms GPU / 1.7 s CPU on the "
            "original RTX-4090 box (torch 2.5.1+cu124, transformers 4.39.3, "
            "co-tenant load)"),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": (f"Amazon {DATASETS[args.dataset]['name']} "
                    f"(LIGER preprocessing, sentence-t5-xl SIDs, seed {args.seed})"),
        "checkpoint": {
            "path": os.path.relpath(args.ckpt, ROOT),
            "n_params": n_params,
        },
        "decode_config": {
            "num_beams": NUM_BEAMS,
            "num_return_sequences": NUM_BEAMS,
            "max_new_tokens_sid_len": data["n_codebook"],
            "n_positions": N_POSITIONS,
            "max_items_per_seq": MAX_ITEMS_PER_SEQ,
            "batch_size": 1,
            "constrained": ("unconstrained beam-20 with post-hoc SID->item "
                            "validity mapping, identical to the reported eval "
                            "(j1_dump.py)"),
            "gpu_precision": "fp16 autocast (same as eval)",
            "cpu_precision": f"fp32, torch threads={args.threads}",
            "timed_section": ("model_forward input prep + generate + "
                              "SID->item mapping, per batch-1 request"),
            "lean_mem_staging": args.lean_mem,
        },
        "environment": {
            "gpu": (torch.cuda.get_device_name(0) if on_gpu else None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "hardware": platform.platform(),
            "loadavg_at_start": list(os.getloadavg()),
        },
        block_key: stats,
        "sanity_decodes": sanity,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({block_key: stats, "sanity_decodes": sanity}, indent=2))
    print(f"[bench] wrote {args.out} [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
