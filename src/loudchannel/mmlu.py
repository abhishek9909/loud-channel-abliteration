"""MMLU accuracy under an optional direction ablation — the capability control.

The blog's central trap is that refusal rate plus a repetition-based degeneracy
filter will pass a model sitting at chance on MMLU. This is the cheap check
that catches it: a fixed slice of multiple-choice questions, scored by the
logprob of the four answer letters at the last content token, measured clean
and under the strongest ablation any arm uses (every token position, every
layer in the band).

`loudchannel.channel_interventions.mmlu_accuracy_ops` is the sibling of this
function for direct channel edits (scale / zero / shift a coordinate) rather
than for a direction.
"""

from __future__ import annotations

import torch

from .directions import Direction
from .model import HarmModel
from .readout import _variant_ids

LETTERS = ("A", "B", "C", "D")

MMLU_TEMPLATE = (
    "Answer the following multiple-choice question with a single letter.\n\n"
    "{question}\n\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\n\nAnswer:"
)


def mmlu_accuracy(
    model: HarmModel,
    rows: list[dict],
    *,
    direction: Direction | None,
    layers: list[int] | None,
    mode: str,
    batch_size: int = 8,
) -> list[int]:
    """Per-question correctness (0/1). `mode="none"` measures the clean model."""
    letter_ids = [_variant_ids(model.tokenizer, [w]) for w in LETTERS]
    correct: list[int] = []
    tup = model.output_is_tuple  # probe BEFORE opening traces
    # NDIF whitelist (see extract.py): hoist envoys + CPU unit vectors out of
    # the trace-body closure
    lm_head = model.lm_head
    edits = []
    if mode != "none":
        assert direction is not None
        edits = [(model.layers[li], direction.at(li, unit=True).float().cpu())
                 for li in layers or []]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [
            model.render(MMLU_TEMPLATE.format(
                question=r["question"], c0=r["choices"][0], c1=r["choices"][1],
                c2=r["choices"][2], c3=r["choices"][3]))
            for r in chunk
        ]
        # read at the last CONTENT token per prompt: [:, -1] would hit a PAD
        # position for every prompt but the longest in the batch (right padding)
        rows_t = torch.arange(len(prompts))
        read_idx = torch.tensor([
            len(model.tokenizer(p, add_special_tokens=True)["input_ids"]) - 1
            for p in prompts
        ])
        with model.trace(prompts):
            for env, r_cpu in edits:
                o = env.output
                h = o[0] if tup else o
                r = r_cpu.to(device=h.device, dtype=h.dtype)
                h[:] = h - (h @ r).unsqueeze(-1) * r
            logits = lm_head.output[rows_t, read_idx].detach().cpu().save()
        lg = logits.float()
        scores = torch.stack(
            [torch.logsumexp(lg[:, ids], dim=-1) for ids in letter_ids], dim=-1
        )  # [B, 4]
        pred = scores.argmax(-1)
        correct += [int(p == r["answer"]) for p, r in zip(pred.tolist(), chunk)]
    return correct
