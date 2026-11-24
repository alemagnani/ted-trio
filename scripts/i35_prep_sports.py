"""I35 -- Sports_and_Outdoors backbone prep: inner world + pred3-enriched item catalogs (E) + I24
input matrices (C). Replicates the exact Beauty frontier data pipeline on this backbone
(18357 items, stock LIGER 5-core); implementation shared with i50_prep_expand.py.

Usage: python scripts/i35_prep_sports.py [--embedder openai|fake|st:<model>]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i50_prep_expand  # noqa: E402

if __name__ == "__main__":
    i50_prep_expand.main(default_dataset="sports")
