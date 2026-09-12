"""Running a model under ChannelOps (the E1/E2 arms).

Deliberate near-duplicate of `interventions.py` / `pipelines.perplexity` /
`04_h2c_capability.mmlu_accuracy`. Those functions are the instrument the
recipe pipeline's headline numbers (25, 26) were measured with, and they carry
hard-won nnsight-0.6 sequencing comments; threading a new parameter through all
four would put the main pipeline's arms at risk of a silent behaviour change
for the sake of an add-on experiment. So the ops path is its own module and the
originals are untouched. If you change the generation loop here, change it
there too — the per-step `tracer.iter[:]` comments in interventions.py are the
authority on why it is shaped this way.

Same NDIF-whitelist discipline as everywhere else: trace bodies reference only
plain tensors, ints, floats and builtin containers hoisted before the trace.

Op semantics per layer, applied in THIS order (see channel_ops.py — the order
matters: the leak must read the residual the layer arrived with, before route 2
scales that coordinate):
    inject   h <- h + coef * h[..., c:c+1] * u
    scale    h <- h * s          (s is [D], ones outside the target channels)
    shift    h <- h + t          (t is [D], zeros outside the target channels)
When `position` is a named position the same three act on that row only
(prefill, persisting through the KV cache); when it is "all" they act on every
token of every step.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .channel_ops import ChannelOps
from .model import HarmModel
from .positions import PromptPositions
from .remote import with_retries


def logits_with_ops(
    model: HarmModel,
    pps: list[PromptPositions],
    *,
    ops: ChannelOps | None,
    position: str = "all",
    batch_size: int = 16,
) -> Tensor:
    """Final-position logits [N, vocab] with the ops applied (ops=None: clean).

    Readout is the last prompt token (t_post_inst), as everywhere else in the
    repo; `position` says where the EDIT lands, not where we read.
    """
    plan = ops.plan() if ops is not None else {}
    layer_envoys = [(li, model.layers[li]) for li in plan]
    lm_head = model.lm_head
    tup = model.output_is_tuple          # probe BEFORE opening traces
    out: list[Tensor] = []

    for start in range(0, len(pps), batch_size):
        chunk = pps[start : start + batch_size]
        prompts = [p.prompt for p in chunk]
        rows = torch.arange(len(chunk))
        read_idx = torch.tensor([p.t_post_inst for p in chunk])
        if position != "all":
            rows_x = rows
            idx = torch.tensor([p.get(position) for p in chunk])

        def run_batch():
            with model.trace(prompts):
                for li, env in layer_envoys:
                    sc, sh, inj = plan[li]
                    o = env.output
                    h = o[0] if tup else o
                    if position == "all":
                        if inj is not None:
                            coef, c, u = inj
                            uu = u.to(device=h.device, dtype=h.dtype)
                            h[:] = h + coef * h[..., c : c + 1] * uu
                        if sc is not None:
                            h[:] = h * sc.to(device=h.device, dtype=h.dtype)
                        if sh is not None:
                            h[:] = h + sh.to(device=h.device, dtype=h.dtype)
                    else:
                        rr = rows_x.to(h.device)
                        ii = idx.to(h.device)
                        v = h[rr, ii]
                        if inj is not None:
                            coef, c, u = inj
                            uu = u.to(device=h.device, dtype=h.dtype)
                            v = v + coef * v[..., c : c + 1] * uu
                        if sc is not None:
                            v = v * sc.to(device=h.device, dtype=h.dtype)
                        if sh is not None:
                            v = v + sh.to(device=h.device, dtype=h.dtype)
                        h[rr, ii] = v
                logits = lm_head.output[rows, read_idx].detach().cpu().save()
            return logits

        logits = with_retries(run_batch, what=f"ops@{position} batch {start}")
        out.append(logits.float())
    return torch.cat(out, dim=0)


def generate_with_ops(
    model: HarmModel,
    prompts: list[str],
    *,
    ops: ChannelOps | None,
    position: str = "all",
    pps: list[PromptPositions] | None = None,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Greedy generation under the ops (ops=None: clean). Returns continuations only.

    position="all": edit every token position on every generation step (the
      E1 arms, matching the all-position ablation the pipeline uses).
    position=<name>: edit that prompt position on the PREFILL pass only; the
      edit persists through the KV cache (the E2 arms, which move a prompt-side
      value and ask what the model then does). Requires `pps`.
    """
    if position != "all":
        assert pps is not None, "named-position generation needs PromptPositions"
        prompts = [p.prompt for p in pps]
    plan = ops.plan() if ops is not None else {}
    layer_envoys = [(li, model.layers[li]) for li in plan]
    lm_head = model.lm_head
    lm = model.lm
    tup = model.output_is_tuple
    tok = model.tokenizer
    texts: list[str] = []

    prev_side = tok.padding_side
    tok.padding_side = "left"            # batched generation requires it
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            in_len = tok(chunk, return_tensors="pt", padding=True)["input_ids"].shape[1]
            if position != "all":
                ch_pps = pps[start : start + batch_size]
                rows = torch.arange(len(ch_pps))
                # left-pad offset per row + the original right-anchored index
                idx = torch.tensor(
                    [in_len - p.n_tokens + p.get(position) for p in ch_pps])

            if position == "all":
                # see interventions.generate_with_intervention for why this
                # never touches generator.output and rebuilds the text from
                # per-step argmax tokens (nnsight 0.6 in-order execution)
                step_tokens: list = []
                with model.generate(chunk, max_new_tokens=max_new_tokens,
                                    do_sample=False) as tracer:
                    for _ in tracer.iter[:]:
                        for li, env in layer_envoys:
                            sc, sh, inj = plan[li]
                            o = env.output
                            h = o[0] if tup else o
                            if inj is not None:
                                coef, c, u = inj
                                uu = u.to(device=h.device, dtype=h.dtype)
                                h[:] = h + coef * h[..., c : c + 1] * uu
                            if sc is not None:
                                h[:] = h * sc.to(device=h.device, dtype=h.dtype)
                            if sh is not None:
                                h[:] = h + sh.to(device=h.device, dtype=h.dtype)
                        step_tokens.append(
                            lm_head.output[:, -1, :].argmax(-1).detach().cpu())
                assert step_tokens, (
                    "no per-step tokens captured — check nnsight tracer.iter")
                toks = torch.stack(step_tokens, dim=1)
                eos = tok.eos_token_id
                eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
                gc = getattr(model.lm, "generation_config", None)
                gc_eos = getattr(gc, "eos_token_id", None)
                if gc_eos is not None:
                    eos_ids |= set(gc_eos) if isinstance(gc_eos, (list, tuple)) else {gc_eos}
                for row in toks.tolist():
                    cut = next((i for i, t in enumerate(row) if t in eos_ids), len(row))
                    texts.append(tok.decode(row[:cut], skip_special_tokens=True))
            else:
                with model.generate(chunk, max_new_tokens=max_new_tokens,
                                    do_sample=False):
                    for li, env in layer_envoys:
                        sc, sh, inj = plan[li]
                        o = env.output
                        h = o[0] if tup else o
                        rr = rows.to(h.device)
                        ii = idx.to(h.device)
                        v = h[rr, ii]
                        if inj is not None:
                            coef, c, u = inj
                            uu = u.to(device=h.device, dtype=h.dtype)
                            v = v + coef * v[..., c : c + 1] * uu
                        if sc is not None:
                            v = v * sc.to(device=h.device, dtype=h.dtype)
                        if sh is not None:
                            v = v + sh.to(device=h.device, dtype=h.dtype)
                        h[rr, ii] = v
                    out = lm.generator.output.save()
                for row in out:
                    texts.append(tok.decode(row[in_len:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
    return texts


# ---- capability readouts under ops ------------------------------------------

MMLU_TEMPLATE = (
    "The following is a multiple choice question. Answer with a single "
    "letter.\n\n{question}\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\nAnswer:"
)
LETTERS = ("A", "B", "C", "D")


def _letter_ids(tokenizer, word: str) -> list[int]:
    ids = set()
    for variant in (word, " " + word):
        enc = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def mmlu_accuracy_with_ops(model, rows, *, ops: ChannelOps | None,
                           batch_size: int = 8) -> list[int]:
    """MMLU slice accuracy with the ops applied at every token (or clean when
    ops is None). Mirrors 04_h2c_capability.mmlu_accuracy, including reading at
    the last CONTENT token under right padding."""
    letter_ids = [_letter_ids(model.tokenizer, w) for w in LETTERS]
    plan = ops.plan() if ops is not None else {}
    layer_envoys = [(li, model.layers[li]) for li in plan]
    lm_head = model.lm_head
    tup = model.output_is_tuple
    correct: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [
            model.render(MMLU_TEMPLATE.format(
                question=r["question"], c0=r["choices"][0], c1=r["choices"][1],
                c2=r["choices"][2], c3=r["choices"][3]))
            for r in chunk
        ]
        rows_t = torch.arange(len(prompts))
        read_idx = torch.tensor([
            len(model.tokenizer(p, add_special_tokens=True)["input_ids"]) - 1
            for p in prompts
        ])
        with model.trace(prompts):
            for li, env in layer_envoys:
                sc, sh, inj = plan[li]
                o = env.output
                h = o[0] if tup else o
                if inj is not None:
                    coef, c, u = inj
                    uu = u.to(device=h.device, dtype=h.dtype)
                    h[:] = h + coef * h[..., c : c + 1] * uu
                if sc is not None:
                    h[:] = h * sc.to(device=h.device, dtype=h.dtype)
                if sh is not None:
                    h[:] = h + sh.to(device=h.device, dtype=h.dtype)
            logits = lm_head.output[rows_t, read_idx].detach().cpu().save()
        lg = logits.float()
        scores = torch.stack(
            [torch.logsumexp(lg[:, ids], dim=-1) for ids in letter_ids], dim=-1)
        pred = scores.argmax(-1)
        correct += [int(p == r["answer"]) for p, r in zip(pred.tolist(), chunk)]
    return correct


def perplexity_with_ops(model, texts: list[str], *, ops: ChannelOps | None,
                        batch_size: int = 8) -> float:
    """Mean per-token NLL with the ops applied at every token (clean when ops
    is None). Mirrors pipelines.perplexity."""
    plan = ops.plan() if ops is not None else {}
    layer_envoys = [(li, model.layers[li]) for li in plan]
    lm_head = model.lm_head
    tup = model.output_is_tuple
    total_nll, total_tok = 0.0, 0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = model.tokenizer(chunk, return_tensors="pt", padding=True)
        with model.trace(chunk):
            for li, env in layer_envoys:
                sc, sh, inj = plan[li]
                o = env.output
                h = o[0] if tup else o
                if inj is not None:
                    coef, c, u = inj
                    uu = u.to(device=h.device, dtype=h.dtype)
                    h[:] = h + coef * h[..., c : c + 1] * uu
                if sc is not None:
                    h[:] = h * sc.to(device=h.device, dtype=h.dtype)
                if sh is not None:
                    h[:] = h + sh.to(device=h.device, dtype=h.dtype)
            logits = lm_head.output.detach().cpu().save()
        lg = logits.float()
        ids, mask = enc["input_ids"], enc["attention_mask"]
        logprobs = torch.log_softmax(lg[:, :-1], dim=-1)
        nll = -logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].bool()
        total_nll += float(nll[m].sum())
        total_tok += int(m.sum())
    return total_nll / max(total_tok, 1)


Mode = Literal["ablate", "steer", "none"]   # re-exported for symmetry
