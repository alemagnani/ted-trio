#!/usr/bin/env python
"""J1/CLOSE-2 dump script. Runs on the GPU box inside the LIGER clone AFTER training.

Dumps, for {Beauty,Toys_and_Games,Sports_and_Outdoors} / sentence-t5-xl / seed 42:
  LIGER (method=setting): projected item embeddings, per-query user reps (dense
    predicted_embedding), and dense top-100 runs (RunRecord-style JSONL).
  TIGER (method=base): unconstrained beam-20 hypotheses with per-token logprobs
    (BeamRecord-style JSONL) via compute_transition_scores (transformers 4.39.3).

Usage (from $LIGER_ROOT):
  python j1_dump.py --which liger --split val                   # Beauty (J1 default)
  python j1_dump.py --dataset toys   --which tiger --split test
  python j1_dump.py --dataset sports --which tiger --split test
  python j1_dump.py --all

Default (--dataset beauty) is bit-identical in behavior to the original J1
Beauty script: same constants, same j1_* checkpoint dirs, same dump filenames
(close2_tiger_score.py's Beauty verification, HR@5 0.0453, depends on this).
"""

import argparse
import json
import math
import os
import sys

LIGER_ROOT = os.environ.get("LIGER_ROOT", os.getcwd())  # run from inside the LIGER clone
sys.path.insert(0, LIGER_ROOT)

# ----------------------------------------------------------------------------- config
SEED = 42
CODEBOOK_SIZE = 256
N_POSITIONS = 102
MAX_ITEMS_PER_SEQ = 20
EVAL_BATCH_SIZE = 32
NUM_BEAMS = 20
TOPK_DENSE = 100

# Per-dataset constants (baselines/tiger/TIGER_RUNBOOK.md, section 0).
# `experiment`: Beauty checkpoints were trained by J1 under experiment_id=j1_{method};
# Toys/Sports train under experiment_id=close2_{method} (runner doc, section 3).
DATASETS = {
    "beauty": dict(name="Beauty", liger_dir="Amazon_Beauty",
                   n_items=12101, experiment="j1"),
    "toys": dict(name="Toys_and_Games", liger_dir="Amazon_Toys_and_Games",
                 n_items=11924, experiment="close2"),
    "sports": dict(name="Sports_and_Outdoors", liger_dir="Amazon_Sports_and_Outdoors",
                   n_items=18357, experiment="close2"),
    # EXPAND (next paper): trained on-box via run_expand.py, experiment_id=close2_tiger
    "instruments": dict(name="Musical_Instruments", liger_dir="Amazon_Musical_Instruments",
                        n_items=9922, experiment="close2"),
    "yelp": dict(name="Yelp", liger_dir="Amazon_Yelp",
                 n_items=20033, experiment="close2"),
}


def configure(dataset_key, seed=42):
    """Set per-dataset globals. Module import defaults to beauty so the
    original J1 Beauty behavior is preserved exactly.

    `seed` (release addition) selects the per-seed SID pickle and checkpoint dir
    (`*_sentence-t5-xl_{seed}.pkl`, `*_seed_{seed}`); the Instruments/Yelp seed-43/44
    dumps were made with per-seed copies of this script that changed exactly these
    constants. Default 42 = the original behavior."""
    global DATASET_KEY, DATASET, N_ITEMS, SEED
    global DATA_FILE, ID2META_FILE, EMBEDDING_FILE, SID_FILE, CKPT_DIR
    d = DATASETS[dataset_key]
    name = d["name"]
    DATASET_KEY = dataset_key
    DATASET = d["liger_dir"]
    N_ITEMS = d["n_items"]
    DATA_FILE = f"{LIGER_ROOT}/ID_generation/preprocessing/processed/{name}.txt"
    ID2META_FILE = (
        f"{LIGER_ROOT}/ID_generation/preprocessing/processed/"
        f"{name}_title_price_brand_categories_amazon_id2meta.json"
    )
    EMBEDDING_FILE = (
        f"{LIGER_ROOT}/ID_generation/preprocessing/processed/"
        f"{name}_sentence-t5-xl_embeddings.pt"
    )
    SEED = seed
    SID_FILE = f"{LIGER_ROOT}/ID_generation/ID/{name}_sentence-t5-xl_{seed}.pkl"
    exp = d["experiment"]
    CKPT_DIR = {
        "liger": f"{LIGER_ROOT}/results/liger/{DATASET}/{exp}_liger_seed_{seed}",
        "tiger": f"{LIGER_ROOT}/results/tiger/{DATASET}/{exp}_tiger_seed_{seed}",
    }


configure("beauty")
DUMP_DIR = os.environ.get("J1_DUMP_DIR", os.path.join(LIGER_ROOT, "dumps"))


def _import_deps():
    """Heavy imports (torch + vendor LIGER modules), deferred so that
    `python j1_dump.py --help` works off-box. Called from main() before any
    work, so runtime behavior is unchanged."""
    global np, torch, F, DataLoader, tqdm, T5Config
    global CustomDataset, set_seed, process_data_split, load_data
    global model_forward, get_target_embed, TIGER
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import T5Config

    from utils import CustomDataset, set_seed  # $LIGER_ROOT/utils.py
    from ID_generation.utils import process_data_split
    from src.load_data import load_data
    from src.evaluation import model_forward, get_target_embed
    from src.tiger import TIGER


def _no_grad(fn):
    """Lazy @torch.no_grad(): torch is only imported in _import_deps()."""
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper

# T5 section of configs/dataset/amazon.yaml
T5_CFG = dict(
    encoder_layers=6, decoder_layers=6, d_model=128, d_ff=1024, num_heads=6,
    d_kv=64, dropout_rate=0.2, feed_forward_proj="relu", initializer_factor=0.02,
)

# configs/method/base.yaml + top part of configs/main.yaml (flattened as run.py does)
METHOD_BASE = dict(
    include_user_id=False, use_id="sid",
    flag_add_input_embedding=False, flag_use_output_embedding=False,
    embedding_loss_weight=0, sid_loss_weight=1, evaluation_method="default",
    embedding_head_dict=dict(
        use_new_init=False, embed_target="ground_truth", embed_proj_type="mlp",
    ),
)
# configs/method/setting.yaml (inherits base)
METHOD_SETTING = dict(
    METHOD_BASE,
    flag_add_input_embedding=True, flag_use_output_embedding=True,
    embedding_loss_weight=1, sid_loss_weight=1, evaluation_method="dense",
    embedding_head_dict=dict(
        METHOD_BASE["embedding_head_dict"],
        normalize_logits=True, logits_temperature=0.07,
        text_embedding_dim=768, hidden_sizes=[256, 512, 768],
        embd_proj_dropout_rate=0.5, embd_proj_in_dropout_rate=0.2,
    ),
)
METHOD_CFG = {"liger": METHOD_SETTING, "tiger": METHOD_BASE}


# ----------------------------------------------------------------------------- data
_DATA_CACHE = None


def build_data():
    """Rebuild data exactly as run.py -> train_tiger does (include_user_id False,
    so splits are identical for both methods)."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    set_seed(SEED)
    cfg = {"dataset": {"max_items_per_seq": MAX_ITEMS_PER_SEQ}}
    id_split, user_sequence = process_data_split(cfg, DATA_FILE, ID2META_FILE, is_steam=False)
    item_embedding = torch.load(EMBEDDING_FILE, map_location="cpu", weights_only=False).float()
    assert item_embedding.shape == (N_ITEMS, 768), f"bad embedding shape {item_embedding.shape}"

    (training_data, val_data, test_data, unseen_val_data, unseen_test_data,
     _seen_sids, _val_unseen_sids, _test_unseen_sids, max_last_semantic_ids,
     n_semantic_codebook, n_codebook, item2sid) = load_data(
        SID_FILE, user_sequence,
        id_split["unseen_val"], id_split["unseen_test"], id_split["seen"],
        item_embedding, METHOD_BASE,
        max_length=N_POSITIONS, codebook_size=CODEBOOK_SIZE,
        max_items_per_seq=MAX_ITEMS_PER_SEQ,
    )
    assert n_codebook == 4 and n_semantic_codebook == 3
    assert item2sid.shape == (N_ITEMS, 4)
    last_codebook_size = max(max_last_semantic_ids, CODEBOOK_SIZE)
    vocab_size = CODEBOOK_SIZE * n_semantic_codebook + last_codebook_size + 2

    sid2item = {tuple(int(t) for t in item2sid[i]): i + 1 for i in range(N_ITEMS)}
    assert len(sid2item) == N_ITEMS, "expanded SIDs must be unique"

    _DATA_CACHE = dict(
        item_embedding=item_embedding, item2sid=item2sid, sid2item=sid2item,
        vocab_size=int(vocab_size), n_codebook=n_codebook,
        n_semantic_codebook=n_semantic_codebook,
        splits={
            "val": (val_data, unseen_val_data),
            "test": (test_data, unseen_test_data),
        },
    )
    return _DATA_CACHE


def split_loaders(data, split):
    """(loader, cold_flag) pairs: seen part first, then unseen — each in its
    original order, with a per-query cold flag."""
    seen_d, unseen_d = data["splits"][split]
    out = []
    for d, cold in [(seen_d, False), (unseen_d, True)]:
        if len(d["input_ids"]) > 0:
            out.append((DataLoader(CustomDataset(d), batch_size=EVAL_BATCH_SIZE,
                                   shuffle=False), cold))
    return out


# ----------------------------------------------------------------------------- model
def build_model(which, data, device):
    mc = METHOD_CFG[which]
    vocab = data["vocab_size"]
    model_config = T5Config(
        num_layers=T5_CFG["encoder_layers"], num_decoder_layers=T5_CFG["decoder_layers"],
        d_model=T5_CFG["d_model"], d_ff=T5_CFG["d_ff"], num_heads=T5_CFG["num_heads"],
        d_kv=T5_CFG["d_kv"], dropout_rate=T5_CFG["dropout_rate"], vocab_size=vocab,
        pad_token_id=0, eos_token_id=int(vocab - 1), decoder_start_token_id=0,
        feed_forward_proj=T5_CFG["feed_forward_proj"], n_positions=N_POSITIONS,
        layer_norm_epsilon=1e-8, initializer_factor=T5_CFG["initializer_factor"],
    )
    model = TIGER(
        config=model_config, n_semantic_codebook=data["n_semantic_codebook"],
        max_items_per_seq=MAX_ITEMS_PER_SEQ,
        flag_use_output_embedding=mc["flag_use_output_embedding"],
        flag_use_learnable_text_embed=mc["flag_add_input_embedding"],
        embedding_head_dict=mc["embedding_head_dict"],
    )
    best = os.path.join(CKPT_DIR[which], "results", "ckpt_best.pt")
    last = os.path.join(CKPT_DIR[which], "ckpt.pt")
    if os.path.exists(best):
        state, src = torch.load(best, map_location="cpu", weights_only=False), best
    else:
        assert os.path.exists(last), f"no checkpoint for {which} at {last}"
        state, src = torch.load(last, map_location="cpu", weights_only=False)["model_state_dict"], last
    assert state["shared.weight"].shape[0] == vocab, (
        f"vocab mismatch: ckpt {state['shared.weight'].shape[0]} vs data {vocab}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"[{which}] loaded {src} (vocab={vocab})")
    return model, mc, src


# ----------------------------------------------------------------------------- metrics
def hr_ndcg(ranks, ks=(5, 10)):
    """ranks: list of 0-based hit rank or None. Binary relevance =>
    NDCG@k == 1/log2(rank+2), matching src.evaluation.ndcg_at_k_torch."""
    out = {}
    for k in ks:
        hits = [1.0 if (r is not None and r < k) else 0.0 for r in ranks]
        ndcg = [1.0 / math.log2(r + 2) if (r is not None and r < k) else 0.0 for r in ranks]
        out[f"HR@{k}"] = float(np.mean(hits)) if ranks else float("nan")
        out[f"NDCG@{k}"] = float(np.mean(ndcg)) if ranks else float("nan")
    return out


def print_metrics(name, ranks, colds):
    groups = {"all": ranks,
              "seen": [r for r, c in zip(ranks, colds) if not c],
              "cold": [r for r, c in zip(ranks, colds) if c]}
    for g, rs in groups.items():
        m = hr_ndcg(rs)
        print(f"[{name}] {g:5s} n={len(rs):6d} " +
              " ".join(f"{k}={v:.4f}" for k, v in m.items()))


# ----------------------------------------------------------------------------- liger
@_no_grad
def dump_liger_items(model, data, device):
    """3a: get_target_embed-style projected item embeddings, fp32 normalized."""
    proj = model.emb_proj(data["item_embedding"].to(device))
    proj = F.normalize(proj.float(), dim=-1)
    assert proj.shape[0] == N_ITEMS
    path = os.path.join(DUMP_DIR, f"liger_{DATASET_KEY}_item_embeddings.pt")
    torch.save(proj.cpu(), path)
    print(f"saved {path} {list(proj.shape)}")


@_no_grad
def dump_liger(model, mc, ckpt_src, data, split, device):
    item_embedding_dev = data["item_embedding"].to(device)
    reps, ranks, colds = [], [], []
    meta_path = os.path.join(DUMP_DIR, f"liger_{DATASET_KEY}_user_reps_{split}.meta.jsonl")
    run_path = os.path.join(DUMP_DIR, f"liger_{DATASET_KEY}_dense_top100_{split}.jsonl")
    qidx = 0
    with open(meta_path, "w") as f_meta, open(run_path, "w") as f_run:
        for loader, cold in split_loaders(data, split):
            for batch in tqdm(loader, desc=f"liger {split} cold={cold}"):
                # full forward exactly as evaluate_dense_sids (amp inside model_forward)
                _, _ = model_forward(model, batch, device, data["n_codebook"], mc)
                pred = model.predicted_embedding  # [B, d_model], fp16
                _, logits = get_target_embed(pred, model, mc, item_embedding_dev)
                scores, idx = logits.float().topk(TOPK_DENSE, dim=1, largest=True)
                targets = batch["labels_ids"][:, 0].tolist()
                reps.append(pred.float().cpu())
                for b in range(len(targets)):
                    tgt = int(targets[b])
                    retrieved = (idx[b] + 1).tolist()  # 1-indexed item ids
                    rank = retrieved.index(tgt) if tgt in retrieved else None
                    ranks.append(rank)
                    colds.append(cold)
                    qid = f"{split}:{qidx}"
                    f_meta.write(json.dumps({
                        "query_id": qid, "row": qidx, "target_item": tgt,
                        "cold": cold}) + "\n")
                    f_run.write(json.dumps({
                        "query_id": qid, "query_text": "",
                        "relevant_item_ids": [str(tgt)],
                        "retrieved_item_ids": [str(i) for i in retrieved],
                        "retrieved_scores": [round(float(s), 6) for s in scores[b]],
                        "system_name": "liger_dense", "dataset": DATASET,
                        "checkpoint": ckpt_src, "seed": SEED,
                        "latency_ms": 0.0, "relevance_grades": {},
                        "cold": cold}) + "\n")
                    qidx += 1
    reps = torch.cat(reps, dim=0)
    rep_path = os.path.join(DUMP_DIR, f"liger_{DATASET_KEY}_user_reps_{split}.pt")
    torch.save(reps, rep_path)
    print(f"saved {rep_path} {list(reps.shape)}, {run_path}, {meta_path}")
    print_metrics(f"liger dense {split}", ranks, colds)


# ----------------------------------------------------------------------------- tiger
@_no_grad
def dump_tiger(model, mc, ckpt_src, data, split, device):
    n_codebook = data["n_codebook"]
    sid2item = data["sid2item"]
    beam_path = os.path.join(DUMP_DIR, f"{DATASET_KEY}_tiger_beams_{split}.jsonl")
    ranks, colds = [], []
    qidx = 0
    with open(beam_path, "w") as f_beam:
        for loader, cold in split_loaders(data, split):
            for batch in tqdm(loader, desc=f"tiger {split} cold={cold}"):
                _, input_kwargs = model_forward(
                    model, batch, device, n_codebook, mc, skip_forward=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(
                        **input_kwargs, num_beams=NUM_BEAMS,
                        num_return_sequences=NUM_BEAMS, max_new_tokens=n_codebook,
                        use_cache=True, output_scores=True,
                        return_dict_in_generate=True,
                    )
                # transformers 4.39.3: normalize_logits re-applies log_softmax
                trans = model.compute_transition_scores(
                    out.sequences, tuple(s.float() for s in out.scores),
                    out.beam_indices, normalize_logits=True,
                )  # [B*num_beams, gen_len]; padded steps are 0
                bsz = batch["labels_sids"].shape[0]
                seqs = out.sequences[:, 1:1 + n_codebook]
                if seqs.shape[1] < n_codebook:  # early eos across whole batch
                    pad = n_codebook - seqs.shape[1]
                    seqs = F.pad(seqs, (0, pad), value=0)
                    trans = F.pad(trans, (0, pad), value=0.0)
                trans = trans[:, :n_codebook]
                seqs = seqs.reshape(bsz, NUM_BEAMS, n_codebook).cpu()
                trans = trans.reshape(bsz, NUM_BEAMS, -1).float().cpu()
                seq_scores = out.sequences_scores.reshape(bsz, NUM_BEAMS).float().cpu()
                targets = batch["labels_ids"][:, 0].tolist()

                for b in range(bsz):
                    tgt = int(targets[b])
                    beams = []
                    for r in range(NUM_BEAMS):
                        toks = [int(t) for t in seqs[b, r]]
                        item = sid2item.get(tuple(toks))
                        lps = [float(x) for x in trans[b, r]]
                        beams.append(dict(
                            query_id=f"{split}:{qidx}", beam_rank=r,
                            semantic_tokens=toks, token_logprobs=lps,
                            sequence_logprob=float(seq_scores[b, r]),
                            probability=0.0,
                            prefixes=[toks[:d + 1] for d in range(len(toks))],
                            mapped_item_id=(str(item) if item is not None else None),
                            valid=item is not None,
                            target_item=tgt, cold=cold,
                        ))
                    # probability: normalized over the query's valid beams
                    joint = [math.exp(sum(bm["token_logprobs"])) for bm in beams]
                    z = sum(j for bm, j in zip(beams, joint) if bm["valid"])
                    for bm, j in zip(beams, joint):
                        bm["probability"] = (j / z) if (bm["valid"] and z > 0) else 0.0
                        f_beam.write(json.dumps(bm) + "\n")
                    rank = next((r for r, bm in enumerate(beams)
                                 if bm["mapped_item_id"] == str(tgt)), None)
                    ranks.append(rank)
                    colds.append(cold)
                    qidx += 1
    print(f"saved {beam_path}")
    print_metrics(f"tiger beams {split}", ranks, colds)
    if DATASET_KEY == "beauty":
        print("(parity check: TIGER seen HR@5 should be roughly ~0.045 on Beauty)")
    elif DATASET_KEY == "toys":
        print("(sanity bar: published TIGER Toys HR@5 ~0.052; flag if far above)")
    elif DATASET_KEY == "sports":
        print("(sanity bar: published TIGER Sports HR@5 ~0.026, LIGER reprod ~0.019)")


# ----------------------------------------------------------------------------- main
def run(which, split, device):
    data = build_data()
    model, mc, ckpt_src = build_model(which, data, device)
    if which == "liger":
        if not getattr(run, "_liger_items_dumped", False):
            dump_liger_items(model, data, device)
            run._liger_items_dumped = True
        dump_liger(model, mc, ckpt_src, data, split, device)
    else:
        dump_tiger(model, mc, ckpt_src, data, split, device)
    del model
    torch.cuda.empty_cache()
    print(f"J1_DUMP_DONE {which} {split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="beauty",
                    help="dataset constants to use (default: beauty, the original J1 behavior)")
    ap.add_argument("--which", choices=["liger", "tiger"])
    ap.add_argument("--split", choices=["val", "test"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="TIGER/SID seed to dump (release addition; default 42 = original)")
    args = ap.parse_args()
    configure(args.dataset, args.seed)
    _import_deps()
    assert torch.cuda.is_available(), "this script requires a CUDA device"
    device = torch.device("cuda:0")
    os.makedirs(DUMP_DIR, exist_ok=True)
    if args.all:
        combos = [(w, s) for w in ["liger", "tiger"] for s in ["val", "test"]]
    else:
        assert args.which and args.split, "need --which and --split, or --all"
        combos = [(args.which, args.split)]
    for which, split in combos:
        run(which, split, device)


if __name__ == "__main__":
    main()
