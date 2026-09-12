"""Direction artifacts: storage, controls (norm-matched random, semantic)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import Tensor


@dataclasses.dataclass
class Direction:
    """Per-layer direction [n_layers, d_model] with provenance metadata."""

    vec: Tensor
    kind: str                  # "d_harm" | "d_refuse" | "d_sentiment" | "random"
    position: str              # extraction position name
    model_name: str
    pos_dataset: str = ""
    neg_dataset: str = ""
    n_pos: int = 0
    n_neg: int = 0
    seed: int | None = None

    def unit(self) -> Tensor:
        return self.vec / self.vec.norm(dim=-1, keepdim=True)

    def norms(self) -> Tensor:
        return self.vec.norm(dim=-1)

    def at(self, layer: int, *, unit: bool = True) -> Tensor:
        v = self.vec[layer]
        return v / v.norm() if unit else v

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataclasses.asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "Direction":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**d)


def cosine(a: Direction, b: Direction) -> Tensor:
    """Per-layer cosine similarity between two directions."""
    return (a.unit() * b.unit()).sum(-1)


def random_directions(
    reference: Direction, n: int, *, seed: int = 0
) -> list[Direction]:
    """n random directions, norm-matched per layer to `reference` (H2b null)."""
    g = torch.Generator().manual_seed(seed)
    ref_norms = reference.norms()  # [n_layers]
    out = []
    for i in range(n):
        r = torch.randn(reference.vec.shape, generator=g, dtype=torch.float32)
        r = r / r.norm(dim=-1, keepdim=True) * ref_norms.unsqueeze(-1)
        out.append(
            Direction(
                vec=r, kind="random", position=reference.position,
                model_name=reference.model_name, seed=seed + i,
            )
        )
    return out


def zero_channels(direction: Direction, channels: dict[int, list[int]]) -> Direction:
    """Zero the given per-layer coordinates of a Direction, renormalising each
    layer back to its original norm (so a null stays norm-matched).

    Why the recipe pipeline needs this: on Gemma an ablation null must be drawn
    ORTHOGONAL to the massive-activation coordinates, or it measures the
    RMSNorm leak (a random unit vector has a ~1/sqrt(D) component on e_c*, and
    (x.r)r then carries ~M/sqrt(D) of the bias coordinate into the residual on
    every prompt — the Consequence-4 leak). On Llama/Qwen the nuisance set is
    empty, so this is the identity and `random_perp` coincides with `random` —
    which is exactly the control that shows the leak is Gemma-specific.
    """
    v = direction.vec.clone().float()
    for li, cs in channels.items():
        if cs and 0 <= li < v.shape[0]:
            v[li, cs] = 0.0
    norms = v.norm(dim=-1, keepdim=True)
    scale = direction.norms().unsqueeze(-1) / norms.clamp_min(1e-12)
    return dataclasses.replace(direction, vec=v * scale,
                               kind=f"{direction.kind}_perp")


def orthogonal_component(
    a: Direction, b: Direction
) -> tuple[Direction, list[float]]:
    """Component of `a` orthogonal to `b`, per layer, unit-renormalized
    (C2 steering arms / Section-D H4 perp ablation). Returns (Direction,
    per-layer cos(a, b)) — the cos is the removed fraction and must be
    reported alongside any perp-arm result.
    NOTE 20_v2_fig4.perp_direction is a local copy (kept to avoid touching
    an in-flight sweep); consolidate here after the C2 runs land."""
    ua, ub = a.unit(), b.unit()
    cos = (ua * ub).sum(-1)                              # [L]
    perp = ua - cos.unsqueeze(-1) * ub
    perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (dataclasses.replace(a, vec=perp, kind=f"{a.kind}_perp"),
            [round(float(c), 4) for c in cos])


def norm_match(direction: Direction, reference: Direction) -> Direction:
    """Rescale `direction` per layer to `reference`'s norms (for the semantic control)."""
    scaled = direction.unit() * reference.norms().unsqueeze(-1)
    return dataclasses.replace(direction, vec=scaled)
