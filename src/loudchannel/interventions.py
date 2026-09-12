"""Interventions inside nnsight traces.

  additive steering:     x <- x + alpha * r_hat        (paper's own intervention, H1)
  directional ablation:  x <- x - (x^T r_hat) r_hat    (necessity tests, H2/H4;
                                                        Arditi et al., 2024)

Two application modes:
  - position-targeted, single forward pass (H1-H3 judgment readout):
    intervene at one named position per prompt, at given layers;
  - all-positions during generation (H4 CoT): ablate everywhere, every step.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .directions import Direction
from .model import HarmModel
from .positions import PromptPositions
from .remote import with_retries

Mode = Literal["ablate", "steer", "none"]


def _basis_at(direction: Direction | list[Direction], li: int,
              device, dtype) -> list[Tensor]:
    """Unit vector(s) to intervene with at layer li.

    Single Direction -> its unit vector. A LIST of Directions (H4 joint
    ablation, e.g. d_harm + d_refuse) -> an orthonormal basis of their joint
    span (QR): sequentially projecting out correlated raw directions does NOT
    remove the span (cos(d_harm, d_refuse) ~ 0.28 on llama), but sequential
    projection with an orthonormal basis is exactly the span projection.
    """
    dirs = list(direction) if isinstance(direction, (list, tuple)) else [direction]
    if len(dirs) == 1:
        return [dirs[0].at(li, unit=True).to(device=device, dtype=dtype)]
    B = torch.stack([d.at(li, unit=True).float() for d in dirs], dim=1)  # [D, k]
    Q, _ = torch.linalg.qr(B)  # orthonormal columns spanning the same space
    return [Q[:, j].to(device=device, dtype=dtype) for j in range(Q.shape[1])]


# NOTE: positional edits are inlined at each use site rather than shared via a
# helper — a loudchannel module function referenced inside a trace body breaks the
# NDIF whitelist ("Module loudchannel.interventions is not whitelisted"). Index
# tensors must live on h's device: getitem tolerates CPU indices but the
# setitem (index_put_) does not.


def logits_with_intervention(
    model: HarmModel,
    pps: list[PromptPositions],
    *,
    direction: Direction | list[Direction] | None,
    layers: list[int],
    position: str | None,          # named position to intervene at; None with mode="none"
    mode: Mode = "none",
    alpha: float = 0.0,
    batch_size: int = 16,
) -> Tensor:
    """Final-position logits [N, vocab] under the given intervention.

    The readout is always the last prompt token (t_post_inst) — the
    intervention position varies (H2a grid), the readout does not.
    position="all" applies the intervention at every token position (used by
    the H4 forced-choice readout, matching the all-position CoT ablation).
    position="req_span" applies it at every token of the located request span
    [t_req_first, t_req] (H1f reply inversion: Zhao et al. steer d_harm on the
    tokens before the inversion question; requires request-located pps).
    """
    if mode == "steer" and isinstance(direction, (list, tuple)):
        assert len(direction) == 1, "steering supports a single direction"
    out: list[Tensor] = []
    tup = model.output_is_tuple  # probe BEFORE opening traces

    # NDIF whitelist (see extract.py): trace bodies must not close over
    # loudchannel objects. Hoist envoys + precompute the unit basis as plain CPU
    # tensors; the body moves them to h's device/dtype with pure torch ops.
    layer_envoys = [(li, model.layers[li]) for li in layers]
    lm_head = model.lm_head
    basis_cpu: dict[int, list[Tensor]] = {}
    if mode != "none":
        assert direction is not None and position is not None
        basis_cpu = {
            li: _basis_at(direction, li, torch.device("cpu"), torch.float32)
            for li in layers
        }

    for start in range(0, len(pps), batch_size):
        chunk = pps[start : start + batch_size]
        prompts = [p.prompt for p in chunk]
        rows = torch.arange(len(chunk))
        read_idx = torch.tensor([p.t_post_inst for p in chunk])

        if mode != "none":
            if position == "req_span":
                flat_r, flat_i = [], []
                for r, p in enumerate(chunk):
                    flat_r.extend([r] * (p.get("t_req") - p.get("t_req_first") + 1))
                    flat_i.extend(range(p.get("t_req_first"), p.get("t_req") + 1))
                rows_x = torch.tensor(flat_r)
                idx = torch.tensor(flat_i)
            elif position != "all":
                rows_x = rows
                idx = torch.tensor([p.get(position) for p in chunk])

        def run_batch():
            # per-batch closure (captures this iteration's prompts/indices):
            # re-executable on transient network failures
            with model.trace(prompts):
                if mode != "none":
                    for li, env in layer_envoys:
                        o = env.output
                        h = o[0] if tup else o
                        for r_cpu in basis_cpu[li]:
                            r = r_cpu.to(device=h.device, dtype=h.dtype)
                            if position == "all":
                                # in-place: the edited tensor flows onward either way
                                if mode == "ablate":
                                    h[:] = h - (h @ r).unsqueeze(-1) * r
                                else:
                                    h[:] = h + alpha * r
                            else:
                                # inlined _apply_at (module functions in the body
                                # would break the NDIF whitelist)
                                rr = rows_x.to(h.device)
                                ii = idx.to(h.device)
                                v = h[rr, ii]
                                if mode == "ablate":
                                    h[rr, ii] = v - (v @ r).unsqueeze(-1) * r
                                else:
                                    h[rr, ii] = v + alpha * r
                logits = lm_head.output[rows, read_idx].detach().cpu().save()
            return logits

        logits = with_retries(run_batch, what=f"{mode}@{position} batch {start}")
        out.append(logits.float())
    return torch.cat(out, dim=0)


def generate_with_intervention(
    model: HarmModel,
    prompts: list[str],
    *,
    direction: Direction | list[Direction] | None,
    layers: list[int],
    mode: Mode = "none",
    alpha: float = 1.0,
    position: str = "all",
    pps: list[PromptPositions] | None = None,   # required when position != "all"
    max_new_tokens: int = 512,
    batch_size: int = 8,
    do_sample: bool = False,
    span: Literal["all", "prompt", "gen"] = "all",
) -> list[str]:
    """Greedy generation under an intervention. Returns ONLY the continuations
    (generated text, prompt excluded).

    `span` (position="all" path only) restricts WHERE in the sequence the
    all-position intervention applies — the H4 position-invariance arms:
      all     every token, every step (prefill + generation; the original H4)
      prompt  prefill pass only: every prompt token is edited once and the
              edit persists through the KV cache; generated tokens untouched
      gen     generation steps only: the prompt is processed clean; each new
              token's residual stream is edited as it is produced

    position="all": intervene at every token position on every generation step
      (H4) via the nnsight 0.6 `for _ in tracer.iter[:]` loop; the
      continuation is rebuilt from per-step lm_head argmax tokens (greedy
      only) because generator.output cannot be combined with per-step edits
      under 0.6's in-order execution (see comments in the body).
    position=<name> (e.g. "t_inst"): intervene at that prompt position on the
      PREFILL pass only — no .all(), so the edit applies once and persists
      through the KV cache. This matches the H1 steering-readout condition
      (edit at t_inst, observe downstream), for coherence checks of steered
      judgments. Requires `pps`.
    position="req_span" (v2 §3-replication, Zhao et al. §3.4/App. E.1): prefill-only
      edit at EVERY token of the located request span [t_req_first, t_req]
      ("all tokens of input instructions"). Requires request-located `pps`.

    Padding: batched HF generation REQUIRES left padding (with right padding
    the first sampled token is conditioned on a trailing PAD for every prompt
    but the longest — the transformers warning seen in earlier logs).
    padding_side is flipped to left for the duration and restored, so
    trace-based position indexing (right padding) is unaffected. Named-position
    indices are shifted per row by the left-pad offset using pps.n_tokens.
    """
    if position != "all":
        assert pps is not None, "named-position generation needs PromptPositions"
        prompts = [p.prompt for p in pps]
    assert span == "all" or position == "all", (
        "span restriction is defined for the all-position path only"
    )

    if mode == "steer" and isinstance(direction, (list, tuple)):
        assert len(direction) == 1, "steering supports a single direction"
    texts: list[str] = []
    tup = model.output_is_tuple  # probe BEFORE opening traces

    # NDIF whitelist (see extract.py): hoist envoys + CPU basis out of the
    # trace-body closures.
    layer_envoys = [(li, model.layers[li]) for li in layers]
    lm_head = model.lm_head
    lm = model.lm
    basis_cpu: dict[int, list[Tensor]] = {}
    if mode != "none":
        assert direction is not None
        basis_cpu = {
            li: _basis_at(direction, li, torch.device("cpu"), torch.float32)
            for li in layers
        }

    tok = model.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            # Padded input length: nnsight tokenizes internally with this same
            # tokenizer + settings, so this re-tokenization matches what the
            # generator sees. With left padding, every row's continuation
            # starts exactly at in_len.
            in_len = tok(chunk, return_tensors="pt", padding=True)["input_ids"].shape[1]
            if position != "all" and mode != "none":
                ch_pps = pps[start : start + batch_size]
                if position == "req_span":
                    # every token of the located request span, shifted per row
                    # by the left-pad offset (mirrors logits_with_intervention)
                    flat_r, flat_i = [], []
                    for r, p in enumerate(ch_pps):
                        off = in_len - p.n_tokens
                        lo, hi = p.get("t_req_first"), p.get("t_req")
                        flat_r.extend([r] * (hi - lo + 1))
                        flat_i.extend(off + k for k in range(lo, hi + 1))
                    rows = torch.tensor(flat_r)
                    idx = torch.tensor(flat_i)
                else:
                    rows = torch.arange(len(ch_pps))
                    # left-pad offset per row + original (right-anchored) index
                    idx = torch.tensor(
                        [in_len - p.n_tokens + p.get(position) for p in ch_pps]
                    )
            if mode != "none" and position == "all":
                # nnsight 0.6 semantics: trace-body statements execute IN
                # ORDER, interleaved with the forward passes. Two dead ends,
                # both observed in practice:
                #   - `generator.output.save()` first blocks until generation
                #     completes, so per-step edits declared after it arrive
                #     too late (MissedProviderError at i0);
                #   - statements after a `with tracer.all():` block are not
                #     executed once the iterator exhausts (UnboundLocalError
                #     on `out`), and that block form is deprecated anyway.
                # So this path never touches generator.output: edits AND
                # next-token capture happen inside the per-step loop
                # (`for _ in tracer.iter[:]`, the documented 0.6 form), and
                # the continuation is rebuilt from the captured argmax tokens
                # — exact under greedy decoding.
                assert not do_sample, "all-position intervention path requires greedy decoding"
                # Accumulator lives OUTSIDE the trace and is mutated from
                # inside: tracer.iter[:] schedules one phantom iteration past
                # the last real step (the benign "i<N> was not provided"
                # warning), the worker thread is abandoned while blocked in
                # it, so the trace body never finishes cleanly and its locals
                # (even .save()'d ones) never sync back — assigning inside
                # the body leaves the name unbound out here. Appends to a
                # shared object happen during generation and survive.
                step_tokens: list = []
                with model.generate(chunk, max_new_tokens=max_new_tokens,
                                    do_sample=do_sample) as tracer:
                    for _ in tracer.iter[:]:
                        # step index BEFORE this step's append: 0 == prefill
                        # (statements execute in order, interleaved with the
                        # forward passes, so len() is the live step counter)
                        is_prefill = len(step_tokens) == 0
                        apply_here = (
                            span == "all"
                            or (span == "prompt" and is_prefill)
                            or (span == "gen" and not is_prefill)
                        )
                        if apply_here:
                            for li, env in layer_envoys:
                                o = env.output
                                h = o[0] if tup else o
                                for r_cpu in basis_cpu[li]:
                                    r = r_cpu.to(device=h.device, dtype=h.dtype)
                                    if mode == "ablate":
                                        h[:] = h - (h @ r).unsqueeze(-1) * r
                                    else:
                                        h[:] = h + alpha * r
                        # [batch, seq, vocab] on prefill, [batch, 1, vocab]
                        # after; [:, -1] is this step's next token either way
                        step_tokens.append(
                            lm_head.output[:, -1, :].argmax(-1).detach().cpu())
                assert step_tokens, (
                    "no per-step tokens captured — nnsight did not run the "
                    "iterator body; check nnsight version/tracer.iter semantics"
                )
                toks = torch.stack(step_tokens, dim=1)  # [batch, n_steps]
                # truncate each row at its first EOS: finished rows keep
                # producing argmax garbage until the whole batch stops
                eos = tok.eos_token_id
                eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
                gc = getattr(model.lm, "generation_config", None)
                gc_eos = getattr(gc, "eos_token_id", None)
                if gc_eos is not None:  # e.g. llama3 also stops on <|eot_id|>
                    eos_ids |= set(gc_eos) if isinstance(gc_eos, (list, tuple)) else {gc_eos}
                for row in toks.tolist():
                    cut = next((i for i, t in enumerate(row) if t in eos_ids), len(row))
                    texts.append(tok.decode(row[:cut], skip_special_tokens=True))
            else:
                # clean generation or prefill-only named-position edit; edits
                # (which fire during prefill) are declared BEFORE the
                # generator-output save, matching the in-order semantics
                with model.generate(chunk, max_new_tokens=max_new_tokens, do_sample=do_sample):
                    if mode != "none":
                        for li, env in layer_envoys:
                            o = env.output
                            h = o[0] if tup else o
                            for r_cpu in basis_cpu[li]:
                                r = r_cpu.to(device=h.device, dtype=h.dtype)
                                rr = rows.to(h.device)
                                ii = idx.to(h.device)
                                v = h[rr, ii]
                                if mode == "ablate":
                                    h[rr, ii] = v - (v @ r).unsqueeze(-1) * r
                                else:
                                    h[rr, ii] = v + alpha * r
                    out = lm.generator.output.save()
                for row in out:
                    texts.append(tok.decode(row[in_len:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
    return texts
