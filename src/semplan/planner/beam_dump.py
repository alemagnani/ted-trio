"""Repo-agnostic beam dump: HuggingFace ``generate`` output -> ``BeamRecord`` (spec §10.1).

Every generative baseline we use (RIPOR, GenRet, NCI, TIGER-family decoders) produces its
hypotheses via ``model.generate(..., num_beams=B, num_return_sequences=B, output_scores=True,
return_dict_in_generate=True)``, optionally with a ``prefix_allowed_tokens_fn`` trie constraint.
This module converts that output — for one query — into our canonical ``BeamRecord`` list with
correct per-token log-probabilities (via ``compute_transition_scores`` using ``beam_indices``),
partial prefixes, and an optional semantic-id -> item mapping.

Keeping this in one place means the same tested code path serves every vendor adapter; each
vendor script only needs to hand us its ``generate`` output object + model.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from semplan.common.beam_schema import BeamRecord


def beamrecords_from_semid_tensor(
    query_id: str,
    generated_ids: Sequence[Sequence[int]],
    sequence_logprobs: Sequence[float],
    *,
    sid_to_item: Mapping[tuple[int, ...], str] | None = None,
) -> list[BeamRecord]:
    """Adapter for repos whose beam search returns whole semantic-id rows + cumulative logprobs.

    Used for ``rqvae_recommender``'s custom sampler-based beam search, which yields
    ``generated_ids [k, H]`` and ``log_probas [k]`` (a *sequence* log-prob per beam) but not
    per-token log-probs. We therefore approximate ``token_logprobs`` by splitting the sequence
    log-prob evenly across the H tokens (so it still sums to the true sequence log-prob); the
    exact per-token dumps come from the HF-``generate`` path (RIPOR/GenRet/NCI). This is adequate
    for the M3 smoke test and for diagnostics that only consume sequence log-probs / prefixes.
    """
    records: list[BeamRecord] = []
    for rank, (row, slp) in enumerate(zip(generated_ids, sequence_logprobs)):
        tokens = [int(t) for t in row]
        h = len(tokens)
        per_tok = [float(slp) / h] * h if h else []
        key = tuple(tokens)
        mapped = sid_to_item.get(key) if sid_to_item is not None else None
        records.append(
            BeamRecord.with_prefixes(
                query_id=query_id,
                beam_rank=rank,
                semantic_tokens=tokens,
                token_logprobs=per_tok,
                sequence_logprob=float(slp),
                probability=0.0,
                mapped_item_id=mapped,
                valid=(mapped is not None) if sid_to_item is not None else True,
            )
        )
    return records


def _strip_leading(tokens: list[int], strip_ids: set[int]) -> list[int]:
    """Drop leading special tokens (e.g. T5 decoder_start / pad) from a decoded sequence."""
    i = 0
    while i < len(tokens) and tokens[i] in strip_ids:
        i += 1
    return tokens[i:]


def _trim_trailing(tokens: list[int], logprobs: list[float], stop_ids: set[int]) -> tuple[list[int], list[float]]:
    """Cut a sequence at the first stop/pad/eos token (inclusive boundary removed)."""
    out_t: list[int] = []
    out_l: list[float] = []
    for t, lp in zip(tokens, logprobs):
        if t in stop_ids:
            break
        out_t.append(t)
        out_l.append(lp)
    return out_t, out_l


def beamrecords_from_generate(
    query_id: str,
    gen_output: Any,
    model: Any,
    *,
    sid_to_item: Mapping[tuple[int, ...], str] | None = None,
    decoder_start_ids: Iterable[int] = (),
    stop_ids: Iterable[int] = (),
    id_is_valid: Callable[[tuple[int, ...]], bool] | None = None,
    normalize_logits: bool = True,
) -> list[BeamRecord]:
    """Convert a HF beam-search ``generate`` output for ONE query into BeamRecords.

    Parameters
    ----------
    gen_output : object with ``.sequences`` and ``.scores`` (and ``.beam_indices`` /
        ``.sequences_scores`` for beam search) — i.e. the return of ``generate(...,
        output_scores=True, return_dict_in_generate=True)``.
    model : the generating model; used only for ``compute_transition_scores`` to recover
        per-token log-probs of the *selected* beams.
    sid_to_item : maps a full semantic-id tuple to its catalogue item/doc id (spec §10.2). A
        beam whose tokens are not a key maps to ``None`` (still recorded, still ``valid`` unless
        ``id_is_valid`` says otherwise) — mirrors constrained decoding that can emit unfinished ids.
    decoder_start_ids / stop_ids : token ids stripped from the front / used to cut the tail.
    id_is_valid : optional predicate on the token tuple; overrides validity (default: valid iff
        it maps to an item when ``sid_to_item`` is given, else always valid).

    Returns a list ordered by beam rank (0 = highest scoring).
    """
    import torch

    sequences = gen_output.sequences  # (num_return_sequences, seq_len)
    scores = getattr(gen_output, "scores", None)
    beam_indices = getattr(gen_output, "beam_indices", None)
    seq_scores = getattr(gen_output, "sequences_scores", None)

    start_set = set(decoder_start_ids)
    stop_set = set(stop_ids)

    # Per-token transition logprobs for the selected sequences. compute_transition_scores
    # aligns to the *generated* tokens (excludes the decoder start), handling beam_indices.
    transition = None
    if scores is not None:
        transition = model.compute_transition_scores(
            sequences, scores, beam_indices, normalize_logits=normalize_logits
        )

    records: list[BeamRecord] = []
    n = sequences.shape[0]
    for rank in range(n):
        seq = [int(t) for t in sequences[rank].tolist()]
        gen_tokens = _strip_leading(seq, start_set)

        if transition is not None:
            row = [float(x) for x in transition[rank].tolist()]
            # transition length == number of generated tokens (post-start). Align by tail.
            row = row[-len(gen_tokens):] if len(gen_tokens) <= len(row) else row
        else:
            row = [0.0] * len(gen_tokens)

        tokens, token_logprobs = _trim_trailing(gen_tokens, row, stop_set)

        # Drop -inf padded positions (masked generated steps) that HF fills for finished beams.
        clean = [(t, lp) for t, lp in zip(tokens, token_logprobs) if lp != float("-inf")]
        tokens = [t for t, _ in clean]
        token_logprobs = [lp for _, lp in clean]

        if seq_scores is not None:
            sequence_logprob = float(seq_scores[rank])
        else:
            sequence_logprob = float(sum(token_logprobs))

        key = tuple(tokens)
        mapped = sid_to_item.get(key) if sid_to_item is not None else None
        if id_is_valid is not None:
            valid = bool(id_is_valid(key))
        elif sid_to_item is not None:
            valid = mapped is not None
        else:
            valid = True

        records.append(
            BeamRecord.with_prefixes(
                query_id=query_id,
                beam_rank=rank,
                semantic_tokens=tokens,
                token_logprobs=token_logprobs,
                sequence_logprob=sequence_logprob,
                probability=0.0,  # filled downstream by diagnostics.beam_entropy.normalized_probs
                mapped_item_id=mapped,
                valid=valid,
            )
        )
    return records
