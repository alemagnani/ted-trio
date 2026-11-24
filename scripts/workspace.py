"""Create an isolated run workspace for scripts/reproduce.sh (release addition).

Every runner resolves paths from its own location (ROOT = parent of scripts/), so a
workspace is simply a directory holding a copy of scripts/ (+ docs/, a link to src/) and
its own data/, outputs/ and results/. Nothing a workspace run writes can touch this
repository's shipped results/.

Modes
  --mode full    data/ -> symlink to the full dataset directory (TT_DATA_DIR); the real
                 pipeline then fills data/ and writes results into <ws>/results, which
                 scripts/compare_results.py diffs against the shipped results/.
  --mode quick   data/ holds the FIRST --users users of every sequence file (real data),
                 the inner file derived from that subset, and read-only symlinks to the
                 existing item-side files (id2meta, item embeddings, input tables, t5).
                 Item ids / catalog size are unchanged, so every script runs unmodified.
  --mode prep    like quick, but WITHOUT links to any embedding/input-table file, so the
                 prep/embedding scripts can be exercised (with --embedder fake) and every
                 file they write is a new file inside the workspace. The full-size sequence
                 files are linked under their own names so the download scripts take their
                 "already present -> verify" path.

Usage:
  python scripts/workspace.py --mode quick --dataset beauty --users 200 \
      --data-src /path/to/data --ws outputs/quick/beauty
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "beauty": dict(bb="amazon_beauty_backbone", seq="Beauty.txt", inner="Beauty_inner.txt",
                   prefix="Beauty", meta=["amazon_meta/beauty_id2meta.json"],
                   extra=["Beauty_item2attributes.json"]),
    "toys": dict(bb="amazon_toys_backbone", seq="Toys_and_Games.txt", inner="Toys_inner.txt",
                 prefix="Toys", meta=["amazon_meta/toys_id2meta.json"],
                 extra=["Toys_and_Games_item2attributes.json"]),
    "sports": dict(bb="amazon_sports_backbone", seq="Sports_and_Outdoors.txt",
                   inner="Sports_inner.txt", prefix="Sports",
                   meta=["amazon_meta/sports_id2meta.json"],
                   extra=["Sports_and_Outdoors_item2attributes.json"]),
    "instruments": dict(bb="amazon_instruments_backbone", seq="Musical_Instruments.txt",
                        inner="Instruments_inner.txt", prefix="Instruments",
                        meta=["amazon_meta/instruments_id2meta.json"], extra=[]),
    "yelp": dict(bb="yelp_backbone", seq="Yelp.txt", inner="Yelp_inner.txt", prefix="Yelp",
                 meta=["yelp_meta/yelp_id2meta.json", "yelp_meta/LETTER_Yelp.item.json"],
                 extra=[]),
}
EMB_SUFFIXES = ["_openai-3-large_embeddings.pt", "_openai-3-large_i22inner_embeddings.pt",
                "_openai-3-large_i22real_embeddings.pt", "_openai-3-large_i24inner_input.pt",
                "_openai-3-large_i24real_input.pt", "_openai-3-large_i44plain_input.pt",
                "_sentence-t5-xl_embeddings.pt"]


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  (missing in data source, not linked: {src.name})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def code_tree(ws: Path) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    if (ws / "scripts").exists():
        shutil.rmtree(ws / "scripts")
    shutil.copytree(ROOT / "scripts", ws / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if (ws / "baselines").exists():
        shutil.rmtree(ws / "baselines")
    shutil.copytree(ROOT / "baselines", ws / "baselines",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (ws / "docs").mkdir(exist_ok=True)
    shutil.copy(ROOT / "docs/data_checksums.md5", ws / "docs/data_checksums.md5")
    s = ws / "src"
    if s.is_symlink() or s.exists():
        s.unlink() if s.is_symlink() else shutil.rmtree(s)
    s.symlink_to((ROOT / "src").resolve())
    tp = ws / "third_party"
    if not (tp.is_symlink() or tp.exists()):
        (ROOT / "third_party").mkdir(exist_ok=True)
        tp.symlink_to((ROOT / "third_party").resolve())
    for d in ("results", "outputs/artifacts", "ledger"):
        (ws / d).mkdir(parents=True, exist_ok=True)


def subset_data(ds: str, src: Path, ws: Path, users: int, mode: str) -> None:
    cfg = DATASETS[ds]
    bb = ws / "data" / cfg["bb"]
    bb.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(src / cfg["bb"] / cfg["seq"]) as f:
        for line in f:
            if line.strip():
                lines.append(line.rstrip("\n"))
            if len(lines) >= users:
                break
    if mode == "prep":
        # full-size sequence file under its own name -> download scripts verify, never write
        link(src / cfg["bb"] / cfg["seq"], bb / cfg["seq"])
    else:
        (bb / cfg["seq"]).write_text("\n".join(lines) + "\n")
        (bb / cfg["inner"]).write_text(
            "\n".join(" ".join(l.split()[:-1]) for l in lines) + "\n")
    for m in cfg["meta"]:
        if mode == "prep" and m.startswith("yelp_meta/"):
            continue  # built inside the workspace by build_yelp_meta.py
        link(src / m, ws / "data" / m)
    for e in cfg["extra"]:
        link(src / cfg["bb"] / e, bb / e)
    if mode == "quick":
        for suf in EMB_SUFFIXES:
            link(src / cfg["bb"] / f"{cfg['prefix']}{suf}", bb / f"{cfg['prefix']}{suf}")
    print(f"[workspace] {ds}: {len(lines)} users -> {bb}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("full", "quick", "prep"), required=True)
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=[])
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--data-src", default=os.environ.get("TT_DATA_DIR", str(ROOT / "data")))
    ap.add_argument("--ws", required=True)
    args = ap.parse_args()
    ws, src = Path(args.ws), Path(args.data_src)
    code_tree(ws)
    if args.mode == "full":
        d = ws / "data"
        if d.is_symlink() or d.exists():
            d.unlink()
        src.mkdir(parents=True, exist_ok=True)
        d.symlink_to(src.resolve())
        # booked per-user vectors of EARLIER looks that later looks compare against
        for f in ("i23_test_per_user.npz", "j7_test_per_user.npz"):
            if not (ws / "results" / f).exists():
                shutil.copy(ROOT / "results" / f, ws / "results" / f)
        print(f"[workspace] full: {d} -> {src.resolve()}")
        return
    for ds in args.dataset:
        subset_data(ds, src, ws, args.users, args.mode)


if __name__ == "__main__":
    main()
