"""Held-out language-modelling loss, with or without an intervention.

The fluency half of the capability check: an ablation that leaves MMLU intact
but triples NLL is not a clean removal, and an ablation that leaves both intact
is the thing we are looking for.
"""

from __future__ import annotations

import torch

from .directions import Direction
from .model import HarmModel


def perplexity(
    model: HarmModel,
    texts: list[str],
    *,
    direction: Direction | None = None,
    layers: list[int] | None = None,
    mode: str = "none",
    batch_size: int = 8,
) -> float:
    """Mean per-token NLL over plain texts, optionally under all-position ablation
    across the layer band."""
    total_nll, total_tok = 0.0, 0
    tup = model.output_is_tuple  # probe BEFORE opening traces
    # NDIF whitelist (see extract.py): hoist envoys + CPU unit vectors out of
    # the trace-body closure.
    lm_head = model.lm_head
    edits = []
    if mode != "none":
        assert direction is not None
        edits = [(model.layers[li], direction.at(li, unit=True).float().cpu())
                 for li in layers or []]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = model.tokenizer(chunk, return_tensors="pt", padding=True)
        with model.trace(chunk):
            for env, r_cpu in edits:
                o = env.output
                h = o[0] if tup else o
                r = r_cpu.to(device=h.device, dtype=h.dtype)
                h[:] = h - (h @ r).unsqueeze(-1) * r
            logits = lm_head.output.detach().cpu().save()
        lg = logits.float()
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        logprobs = torch.log_softmax(lg[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        nll = -logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].bool()
        total_nll += float(nll[m].sum())
        total_tok += int(m.sum())
    return total_nll / max(total_tok, 1)
