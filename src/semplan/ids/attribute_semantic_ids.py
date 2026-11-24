"""Interpretable attribute/taxonomy semantic IDs for e-commerce generative retrieval.

Instead of opaque RQ-VAE codes, build a **human-readable** semantic ID directly from a product's
attribute-value pairs + category taxonomy (the HiD-VAE / OneRetrieval direction, but constructed
straight from metadata — no training needed for the interpretable part):

    ID(item) = [cat_L1, cat_L2, cat_L3, price_bucket, (dedup)]         # interpretable
    optionally ++ [rqvae_c1, rqvae_c2, ...]                             # dense/embedding tail (hybrid)

Each token is a real taxonomy level / attribute value, so a generative model that emits this ID is
literally guessing a **taxonomy path + attributes** — interpretable and controllable (§3.2). The
optional embedding tail adds fine discriminability (COBRA/HiD-VAE-style hybrid). Leaves need not be
unique — for the planner use, a prefix intentionally names a *region*; ANN does the final match.

Input format follows LIGER's `<Dataset>_..._id2meta.json`: item_id -> a descriptive string, e.g.
"...It is priced at 5.04. It is manufactured by 'COKA'. It belongs to the categories of Makeup, Face,
Concealers & Neutralizers." We parse category path / brand / price from it (regex, robust to missing
fields), plus optionally consume LIGER's `item2attributes.json` integer attribute lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CAT_RE = re.compile(r"categories of ([^.]+)\.")
_BRAND_RE = re.compile(r"manufactured by '([^']*)'")
_PRICE_RE = re.compile(r"priced at ([0-9]+(?:\.[0-9]+)?)")


def parse_meta(text: str) -> dict:
    """Extract {categories:[...], brand:str, price:float|None} from a LIGER id2meta string."""
    cats: list[str] = []
    m = _CAT_RE.search(text)
    if m:
        cats = [c.strip() for c in m.group(1).split(",") if c.strip()]
    b = _BRAND_RE.search(text)
    brand = b.group(1).strip() if b else ""
    p = _PRICE_RE.search(text)
    price = float(p.group(1)) if p else None
    return {"categories": cats, "brand": brand, "price": price}


def price_bucket(price: float | None, edges=(5, 10, 20, 50, 100)) -> str:
    """Coarse, interpretable price band (log-ish edges typical of e-commerce)."""
    if price is None:
        return "price=NA"
    lo = 0
    for e in edges:
        if price < e:
            return f"price<{e}"
        lo = e
    return f"price>={lo}"


@dataclass
class AttributeSID:
    depth: int  # number of taxonomy levels kept
    ids: dict[str, tuple[str, ...]]          # item_id -> interpretable token path
    token_vocab: dict[str, int] = field(default_factory=dict)  # token -> int (for the decoder)
    members: dict[tuple[str, ...], list[str]] = field(default_factory=dict)  # prefix -> items

    def as_int_ids(self) -> dict[str, tuple[int, ...]]:
        return {i: tuple(self.token_vocab[t] for t in path) for i, path in self.ids.items()}


def build_attribute_sids(
    id2meta: dict[str, str],
    depth: int = 3,
    use_price: bool = True,
    dedup: bool = True,
) -> AttributeSID:
    """Build interpretable taxonomy+attribute semantic IDs from LIGER-style metadata.

    Token path per item: cat level tokens (padded/truncated to ``depth``) + optional price bucket +
    a collision-resolution ordinal so ids are unique when ``dedup`` (TIGER-style 4th token).
    """
    raw: dict[str, list[str]] = {}
    for item, text in id2meta.items():
        meta = parse_meta(text)
        cats = meta["categories"][:depth]
        # pad shallow taxonomies so every id has the same depth (stable decoder positions)
        cats = cats + [f"cat{len(cats)+j}=∅" for j in range(depth - len(cats))]
        path = [f"L{d+1}={c}" for d, c in enumerate(cats)]
        if use_price:
            path.append(price_bucket(meta["price"]))
        raw[item] = path

    # collision resolution: append an ordinal among items sharing the same path
    seen: dict[tuple[str, ...], int] = {}
    ids: dict[str, tuple[str, ...]] = {}
    members: dict[tuple[str, ...], list[str]] = {}
    for item, path in raw.items():
        key = tuple(path)
        members.setdefault(key, []).append(item)
        if dedup:
            n = seen.get(key, 0)
            seen[key] = n + 1
            path = path + [f"#{n}"]
        ids[item] = tuple(path)

    # build token vocab (per-position-agnostic; the decoder learns positions)
    vocab: dict[str, int] = {}
    for path in ids.values():
        for tok in path:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    return AttributeSID(depth=depth, ids=ids, token_vocab=vocab, members=members)


def coverage_stats(sid: AttributeSID) -> dict:
    """Interpretability/quality quick stats: distinct paths, collision rate, vocab size."""
    n = len(sid.ids)
    distinct_prefix = len(sid.members)  # paths before dedup ordinal
    collisions = sum(len(v) - 1 for v in sid.members.values() if len(v) > 1)
    return {
        "items": n,
        "distinct_taxonomy+price_regions": distinct_prefix,
        "avg_items_per_region": round(n / max(distinct_prefix, 1), 2),
        "collision_items": collisions,
        "collision_rate": round(collisions / max(n, 1), 4),
        "token_vocab_size": len(sid.token_vocab),
    }
