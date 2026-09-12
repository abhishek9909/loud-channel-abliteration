"""Activation extraction + diff-of-means directions via nnsight.

d_harm = mean(h[t_inst | harmful]) - mean(h[t_inst | harmless]), per layer.
d_refuse analogous at t_post_inst (sanity check only).
"""

from __future__ import annotations

import torch
from torch import Tensor
from tqdm import tqdm

from .directions import Direction
from .model import HarmModel
from .positions import positions_for
from .remote import with_retries


def extract_activations(
    model: HarmModel,
    instructions: list[str],
    position: str,
    *,
    layers: list[int] | None = None,
    batch_size: int = 16,
    system: str | None = None,
    seed: int = 0,
    pps=None,   # precomputed PromptPositions (H2d suffixed prompts); overrides
                # `instructions` for rendering
) -> Tensor:
    """Hidden states at `position` for every instruction.

    Returns float32 CPU tensor [n_prompts, n_layers, d_model].
    """
    layers = layers if layers is not None else list(range(model.n_layers))
    if pps is None:
        pps = positions_for(model, instructions, system=system, seed=seed)
    rows: list[Tensor] = []
    tup = model.output_is_tuple  # probe BEFORE opening traces

    # NDIF whitelist: the trace body's closure must not reference loudchannel
    # objects (HarmModel, Direction, module functions) — the remote backend
    # serializes closures and rejects non-whitelisted modules ("Module
    # loudchannel.model is not whitelisted"). Hoist envoys/plain values here;
    # inside the body use only envoys, tensors, and torch ops.
    layer_envoys = [model.layers[li] for li in layers]

    for start in tqdm(range(0, len(pps), batch_size), desc=f"extract@{position}"):
        chunk = pps[start : start + batch_size]
        prompts = [p.prompt for p in chunk]
        idx = torch.tensor([p.get(position) for p in chunk])
        b = torch.arange(len(chunk))

        # NOTE: `saved` must be created INSIDE the trace and .save()'d as a
        # list. Mutating a pre-created outer list works locally (the body
        # executes in-process) but silently no-ops under remote=True: NDIF
        # executes the body server-side and only .save()'d values sync back
        # — the outer list stays empty ("stack expects a non-empty
        # TensorList"). `saved = list().save()` is the documented nnsight 0.6
        # form that works in both modes.
        def run_batch():
            # per-batch closure: safe under nnsight's frame inspection (the
            # with-block lives in the frame that calls .trace()) and safe to
            # re-execute on transient network failures (with_retries)
            with model.trace(prompts):
                saved = list().save()
                for env in layer_envoys:
                    out = env.output
                    # [B, T, D]; right padding -> left indices valid
                    h = out[0] if tup else out
                    saved.append(h[b, idx].detach().cpu())
            # after trace exit, `saved` (body-local, .save()'d) holds values
            return saved

        saved = with_retries(run_batch, what=f"extract@{position} batch {start}")
        rows.append(torch.stack([s.float() for s in saved], dim=1))  # [B, L, D]

    return torch.cat(rows, dim=0)


def diff_of_means(
    model: HarmModel,
    positive: list[str],
    negative: list[str],
    position: str,
    *,
    kind: str,
    pos_dataset: str = "",
    neg_dataset: str = "",
    batch_size: int = 16,
    system: str | None = None,
) -> Direction:
    acts_p = extract_activations(model, positive, position, batch_size=batch_size, system=system)
    acts_n = extract_activations(model, negative, position, batch_size=batch_size, system=system)
    vec = acts_p.mean(0) - acts_n.mean(0)  # [n_layers, d_model]
    assert not vec.isnan().any()
    return Direction(
        vec=vec, kind=kind, position=position, model_name=model.cfg.name,
        pos_dataset=pos_dataset, neg_dataset=neg_dataset,
        n_pos=len(positive), n_neg=len(negative),
    )
