"""I24 — build collab-augmented QUERY-INPUT matrices (C site, top-3 predecessor centroid).

Each input token = [L2(own i22-enriched emb) ; L2(mean emb of top-3 predecessors)]
-> [N, 2*3072]. Items with no predecessors fall back to their own emb in the second half
(identity signal, keeps scale). Scoring item side is UNCHANGED (i22 matrices).

Leak-safety (I22 convention): predecessor co-occurrence counted on sequences with the
user id dropped and the final held-out event dropped -- inner world from
Beauty_inner.txt, real world from Beauty.txt (semplan.enrich.pred_counts).

Outputs:
  data/amazon_beauty_backbone/Beauty_openai-3-large_i24inner_input.pt   [12101, 6144]
  data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt    [12101, 6144]
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import enrich as E  # noqa: E402

N = 12101
DATA = ROOT / "data/amazon_beauty_backbone"


def build(seq_file: str, emb_file: str, out_file: str) -> None:
    if (DATA / out_file).exists():
        print(f"skip {out_file}: exists", flush=True)
        return
    emb = torch.load(DATA / emb_file, map_location="cpu", weights_only=True).float().numpy()
    out, n_aug = E.centroid_inputs(emb, E.pred_counts(DATA / seq_file))
    torch.save(torch.from_numpy(out), DATA / out_file)
    print(f"{out_file}: shape {out.shape}, pred3-augmented {n_aug}/{N}", flush=True)


if __name__ == "__main__":
    build("Beauty_inner.txt", "Beauty_openai-3-large_i22inner_embeddings.pt",
          "Beauty_openai-3-large_i24inner_input.pt")
    build("Beauty.txt", "Beauty_openai-3-large_i22real_embeddings.pt",
          "Beauty_openai-3-large_i24real_input.pt")
