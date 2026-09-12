"""Per-coordinate residual-stream edits: the E1/E2 instruments.

The recipe pipeline can only do two things to the residual stream: project out
a direction, or add a multiple of one. Neither can answer the two questions the
channel account leaves open, because both are questions about ONE coordinate:

  E1  projection ablation of a direction d = a*e_c + b*u does two destructive
      things at once when |x_c| dwarfs the ordinary residual --

          h <- h - (a*x_c + b*(u.h)) * (a*e_c + b*u)
             ~= h - a^2*x_c*e_c        route 2: removes a^2 of the channel,
                                       i.e. scales it by (1 - a^2), which
                                       raises every downstream RMSNorm gain
               - a*b*x_c*u             route 1: a fixed-sign multi-sigma shove
                                       along the feature part, every token,
                                       every layer

      The random-direction null (a ~ 1/sqrt(D)) shows route 1 alone is enough
      to do damage, but says nothing about how the r0 collapse splits between
      the two. Separating them needs a scale op and an inject op.

  E2  "is the class-correlated value of c* at the post-instruction positions
      causally used?" Reader gain 0 says no sublayer reads c* AS A COORDINATE;
      it does not say the value is inert, because c* carries most of the
      residual's squared norm and therefore sets the gain on everything the
      next block reads. The test is to move x_c* by the measured class gap and
      watch refusal -- a shift op -- with a global-gain arm as the control that
      separates "this channel" from "any equivalent norm change".

Three primitives, applied to the residual h [B, T, D] at chosen layers and
either every token or one named position:

  scale    h[..., c] *= g                       (E1 route 2; E2 gain control)
  shift    h[..., c] += delta                   (E2 class-gap move)
  inject   h += coef * h[..., c:c+1] * u        (E1 route 1, the leak)

Order within a layer is INJECT -> scale -> shift, and it is not arbitrary: the
inject term of a projection ablation is -a*b*x_c*u with x_c the value the layer
ARRIVED with, so it must read h before route 2 scales that same coordinate.
Applying scale first makes the leak (1-a^2) times too small and the two arms no
longer sum to the operation they decompose -- which is what
tests/test_channel_ops.py::test_scale_plus_inject_reproduces_projection_ablation
pins. (scale and shift touch only the target coordinate, and u is orthogonal to
it, so nothing else in the ordering interacts.)

Every op is stored as a plain [D] CPU tensor so trace bodies can use it without
referencing a loudchannel object (the NDIF whitelist constraint documented in
extract.py).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor


@dataclasses.dataclass
class ChannelOps:
    """Per-layer residual edits, as plain tensors.

    scale[li]   [D] multiplier, applied elementwise (ones outside the targets)
    shift[li]   [D] additive offset (zeros outside the targets)
    inject[li]  (coef, channel, u[D]) -> h += coef * h[..., channel] * u

    Applied in the order inject, scale, shift (see the module docstring).
    """

    scale: dict[int, Tensor] = dataclasses.field(default_factory=dict)
    shift: dict[int, Tensor] = dataclasses.field(default_factory=dict)
    inject: dict[int, tuple[float, int, Tensor]] = dataclasses.field(default_factory=dict)
    label: str = ""
    meta: dict = dataclasses.field(default_factory=dict)

    def layers(self) -> list[int]:
        return sorted(set(self.scale) | set(self.shift) | set(self.inject))

    def plan(self) -> dict[int, tuple]:
        """{layer: (scale|None, shift|None, inject|None)} of plain CPU float32
        tensors — the form the trace bodies hoist and index."""
        out: dict[int, tuple] = {}
        for li in self.layers():
            sc = self.scale.get(li)
            sh = self.shift.get(li)
            inj = self.inject.get(li)
            out[li] = (
                None if sc is None else sc.detach().float().cpu(),
                None if sh is None else sh.detach().float().cpu(),
                None if inj is None else (float(inj[0]), int(inj[1]),
                                          inj[2].detach().float().cpu()),
            )
        return out

    def describe(self) -> dict:
        """JSON-safe summary for the results file (never the [D] tensors)."""
        d = {"label": self.label, "layers": self.layers(), **self.meta}
        if self.scale:
            li = self.layers()[0]
            nz = (self.scale[li] != 1.0).nonzero().flatten().tolist()
            d["scale"] = {"channels": nz[:16],
                          "gains": [round(float(self.scale[l][c]), 6)
                                    for l in self.layers()[:4] for c in nz[:1]],
                          "n_channels_scaled": len(nz)}
        if self.shift:
            li = self.layers()[0]
            nz = (self.shift[li] != 0.0).nonzero().flatten().tolist()
            d["shift"] = {"channels": nz[:16],
                          "deltas": {str(l): round(float(self.shift[l][c]), 2)
                                     for l in self.layers() for c in nz[:1]}}
        if self.inject:
            coef, c, _ = self.inject[self.layers()[0]]
            d["inject"] = {"coef": round(coef, 6), "channel": c}
        return d


# ---- builders ---------------------------------------------------------------


def scale_channel(channel: int, gain: float, layers: list[int], d_model: int,
                  *, label: str = "", meta: dict | None = None) -> ChannelOps:
    """Multiply ONE coordinate by `gain` at every listed layer.

    gain=0.0 is exactly projection ablation of the one-hot e_channel (the
    "route 2 only, no leak" arm); gain=1-a^2 reproduces the amount of the
    channel that the real estimated direction removes.
    """
    v = torch.ones(d_model)
    v[channel] = float(gain)
    return ChannelOps(scale={li: v.clone() for li in layers},
                      label=label or f"scale[{channel}]={gain:g}",
                      meta={**(meta or {}), "channel": channel, "gain": gain})


def global_scale(gains: dict[int, float], d_model: int, *, label: str = "",
                 meta: dict | None = None) -> ChannelOps:
    """Multiply EVERY coordinate by a per-layer gain — the E2 control that asks
    whether an equivalent change in the residual's norm, spread over all
    coordinates, does what moving c* does."""
    return ChannelOps(
        scale={li: torch.full((d_model,), float(g)) for li, g in gains.items()},
        label=label or "global_scale",
        meta={**(meta or {}), "gains": {str(k): round(float(v), 6)
                                        for k, v in gains.items()}})


def shift_channel(channel: int, deltas: dict[int, float], d_model: int,
                  *, label: str = "", meta: dict | None = None) -> ChannelOps:
    """Add a per-layer offset to ONE coordinate (E2: the measured class gap)."""
    shift = {}
    for li, delta in deltas.items():
        v = torch.zeros(d_model)
        v[channel] = float(delta)
        shift[li] = v
    return ChannelOps(shift=shift, label=label or f"shift[{channel}]",
                      meta={**(meta or {}), "channel": channel,
                            "deltas": {str(k): round(float(v), 3)
                                       for k, v in deltas.items()}})


def shift_vector(deltas: dict[int, "Tensor"], *, label: str = "",
                 meta: dict | None = None) -> ChannelOps:
    """Add a full [D] offset per layer — the E2 positive control.

    Why it exists: E2's first positive control moved the layer's best ORDINARY
    evidence coordinate by its own class gap, and did nothing. That was a badly
    chosen control, and its null is uninformative: §2.2 already establishes the
    class evidence is diffuse (best single coordinate AUROC ~0.87 out of
    hundreds), so no single ordinary coordinate is expected to be causally
    sufficient either. The control that validates the instrument has to move
    the DIFFUSE direction -- a cleaned estimator's vector -- at the same
    position, the same layers, and a matched change in residual RMS. If that
    moves refusal and the c* shift does not, the c* null is a result; if it
    also does nothing, the instrument has no sensitivity at this position and
    neither null means anything.
    """
    return ChannelOps(shift={li: v.detach().float().flatten() for li, v in deltas.items()},
                      label=label or "shift_vector",
                      meta={**(meta or {}), "kind": "dense",
                            "norms": {str(li): round(float(v.norm()), 2)
                                      for li, v in deltas.items()}})


def inject_leak(channel: int, coef: float, u: Tensor, layers: list[int],
                *, label: str = "", meta: dict | None = None) -> ChannelOps:
    """h += coef * h[..., channel] * u — the off-axis term of a projection
    ablation, isolated from the projection itself (E1 route 1).

    With (a, b, u) from `decompose_row`, coef = -a*b reproduces exactly the
    leak that ablating that direction injects, and nothing else.
    """
    uu = u.detach().float().flatten()
    uu = uu / uu.norm().clamp_min(1e-12)
    return ChannelOps(inject={li: (float(coef), int(channel), uu.clone()) for li in layers},
                      label=label or f"inject[{channel}]*{coef:+.4f}",
                      meta={**(meta or {}), "channel": channel, "coef": float(coef)})


def merge(*ops: ChannelOps, label: str = "") -> ChannelOps:
    """Combine op sets. Scales multiply, shifts add, injects must not collide."""
    out = ChannelOps(label=label or "+".join(o.label for o in ops))
    for o in ops:
        for li, v in o.scale.items():
            out.scale[li] = v.clone() if li not in out.scale else out.scale[li] * v
        for li, v in o.shift.items():
            out.shift[li] = v.clone() if li not in out.shift else out.shift[li] + v
        for li, v in o.inject.items():
            assert li not in out.inject, f"two inject ops at layer {li}"
            out.inject[li] = v
        out.meta.setdefault("merged", []).append(o.describe())
    return out


# ---- decomposition of an estimated direction --------------------------------


def decompose_row(row: Tensor, channel: int) -> dict:
    """Split an estimated direction into its component on ONE coordinate and
    the rest: d_hat = a*e_c + b*u, with a signed, b >= 0, u a unit vector
    orthogonal to e_c.

    Returns a, b, u and the two derived intervention strengths:
      route2_gain = 1 - a^2   the fraction of x_c that survives the ablation
      leak_coef   = -a*b      the coefficient of x_c injected along u
    """
    r = row.detach().float().flatten()
    n = float(r.norm())
    assert n > 1e-12, "zero-norm direction"
    a = float(r[channel]) / n
    rest = r.clone()
    rest[channel] = 0.0
    b = float(rest.norm()) / n
    u = rest / rest.norm().clamp_min(1e-12)
    return {"channel": int(channel), "a": a, "b": b, "u": u,
            "route2_gain": 1.0 - a * a, "leak_coef": -a * b,
            "row_norm": n}
