"""CPU tests for the E1/E2 instruments — no model, no download.

The load-bearing claim is the one in channel_ops' docstring: on a residual
whose norm is dominated by one coordinate, projection ablation of
d = a*e_c + b*u equals (scale c by 1-a^2) + (inject -a*b*x_c along u), up to a
term of order b^2*(u.x) that the massive coordinate dwarfs. If that algebra is
wrong the E1 arms do not decompose the operation they claim to decompose, so it
is pinned here against an explicit reference implementation.

    pytest tests/test_channel_ops.py
"""

import torch

from loudchannel.channel_ops import (
    ChannelOps,
    decompose_row,
    global_scale,
    inject_leak,
    merge,
    scale_channel,
    shift_channel,
)

D = 64
C = 7


def _apply(ops: ChannelOps, h: torch.Tensor, layer: int = 0) -> torch.Tensor:
    """Reference application, in the documented order (inject, scale, shift) —
    the same arithmetic the trace bodies inline. The order is load-bearing: the
    leak reads the coordinate the layer arrived with, not the scaled one."""
    sc, sh, inj = ops.plan()[layer]
    out = h.clone()
    if inj is not None:
        coef, c, u = inj
        out = out + coef * out[..., c : c + 1] * u
    if sc is not None:
        out = out * sc
    if sh is not None:
        out = out + sh
    return out


def _massive_residual(n=32, scale=3e4, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, D, generator=g)
    h[:, C] += scale                      # one coordinate carries ~all the norm
    return h


def test_decomposition_is_an_orthonormal_split():
    g = torch.Generator().manual_seed(1)
    row = torch.randn(D, generator=g)
    row[C] = 12.0                          # make c dominant in the DIRECTION too
    d = decompose_row(row, C)
    assert abs(d["a"] ** 2 + d["b"] ** 2 - 1.0) < 1e-5
    assert abs(float(d["u"][C])) < 1e-6            # u is orthogonal to e_c
    assert abs(float(d["u"].norm()) - 1.0) < 1e-5
    # reassembly reproduces the unit direction
    rebuilt = d["a"] * torch.nn.functional.one_hot(torch.tensor(C), D).float() \
        + d["b"] * d["u"]
    assert torch.allclose(rebuilt, row / row.norm(), atol=1e-5)


def test_scale_plus_inject_reproduces_projection_ablation():
    """The E1 claim: the two ops ARE the ablation, on a massive-channel residual.

    Also pins the ordering. With scale applied before inject the leak carries
    (1-a^2)*x_c instead of x_c and this assertion fails by ~30%, which is how
    the ordering bug was found in the first place.
    """
    g = torch.Generator().manual_seed(2)
    row = torch.randn(D, generator=g)
    row[C] = 20.0
    dec = decompose_row(row, C)
    r = (row / row.norm())
    h = _massive_residual()

    exact = h - (h @ r).unsqueeze(-1) * r                   # projection ablation
    two_routes = _apply(merge(
        scale_channel(C, dec["route2_gain"], [0], D),
        inject_leak(C, dec["leak_coef"], dec["u"], [0], ),
    ), h)

    # the omitted term is -b*(u.h)*(a e_c + b u); with |x_c| ~ 3e4 it is tiny
    # relative to the edit, which is the regime the whole account is about
    err = (exact - two_routes).norm(dim=-1)
    edit = (h - exact).norm(dim=-1)
    assert float((err / edit).max()) < 0.02, float((err / edit).max())


def test_scale_zero_is_projection_of_the_one_hot():
    h = _massive_residual()
    e = torch.zeros(D)
    e[C] = 1.0
    exact = h - (h @ e).unsqueeze(-1) * e
    got = _apply(scale_channel(C, 0.0, [0], D), h)
    assert torch.allclose(exact, got, atol=1e-4)
    assert float(got[:, C].abs().max()) == 0.0


def test_shift_moves_only_the_target_channel():
    h = _massive_residual()
    ops = shift_channel(C, {0: 500.0}, D)
    got = _apply(ops, h)
    assert torch.allclose(got[:, C], h[:, C] + 500.0, atol=1e-3)
    other = [i for i in range(D) if i != C]
    assert torch.allclose(got[:, other], h[:, other])


def test_global_scale_changes_norm_by_the_gain():
    h = _massive_residual()
    got = _apply(global_scale({0: 1.25}, D), h)
    assert torch.allclose(got.norm(dim=-1), h.norm(dim=-1) * 1.25, rtol=1e-4)


def test_merge_composes_and_describe_is_json_safe():
    import json

    ops = merge(scale_channel(C, 0.5, [3], D), shift_channel(C, {3: 10.0}, D))  # noqa: E501
    h = _massive_residual()
    got = _apply(ops, h, layer=3)
    assert torch.allclose(got[:, C], h[:, C] * 0.5 + 10.0, atol=1e-3)
    json.dumps(ops.describe())            # must not raise / must carry no tensors


def test_layers_and_plan_cover_every_requested_layer():
    ops = scale_channel(C, 0.1, [0, 5, 9], D)
    assert ops.layers() == [0, 5, 9]
    assert set(ops.plan()) == {0, 5, 9}
    for li in ops.layers():
        sc, sh, inj = ops.plan()[li]
        assert sc is not None and sh is None and inj is None
        assert sc.dtype == torch.float32 and sc.device.type == "cpu"
