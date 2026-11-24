"""I25 — dump inner-world enriched catalog texts (I22 pred3 winner recipe) for
open-stack (openemb) embedding on the GPU box. Text-only; no API calls.

Output: data/amazon_beauty_backbone/i25_texts_inner_pred3.json  (list[str], len 12101)
        data/amazon_beauty_backbone/i25_texts_plain.json        (list[str], len 12101)
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 12101

meta = json.load(open(ROOT / "data/amazon_meta/beauty_id2meta.json"))
base_texts = [""] * N
for k, t in meta.items():
    base_texts[int(k) - 1] = t
titles = [""] * N
for i, t in enumerate(base_texts):
    m = re.search(r"called '(.*?)'\.", t)
    titles[i] = m.group(1) if m else ""

pred = defaultdict(Counter)
for line in open(ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"):
    ids = [int(x) for x in line.split()]
    ids = ids[1:] if len(ids) > 1 else ids
    ids = ids[:-1]  # exclude held-out final event (include-val convention)
    for p, i in zip(ids, ids[1:]):
        pred[i][p] += 1

texts = []
n_enriched = 0
for i in range(N):
    t = base_texts[i] if base_texts[i] else " "
    ps = [p for p, _ in pred.get(i + 1, Counter()).most_common(3) if titles[p - 1]]
    if ps:
        t += " Often bought after: " + "; ".join(f"'{titles[p-1]}'" for p in ps) + "."
        n_enriched += 1
    texts.append(t)

json.dump(texts, open(ROOT / "data/amazon_beauty_backbone/i25_texts_inner_pred3.json", "w"))
json.dump([t if t else " " for t in base_texts],
          open(ROOT / "data/amazon_beauty_backbone/i25_texts_plain.json", "w"))
print(f"enriched {n_enriched}/{N}")
