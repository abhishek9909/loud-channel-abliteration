"""Estimator recipes on synthetic activations — no model, no GPU.

Two synthetic worlds, both with the SAME planted feature direction:

  gemma_like   one coordinate is a bias: huge mean, huge variance, tiny class
               effect (d' ~ 0.15) — the massive-activation case;
  healthy      no such coordinate — the Llama/Qwen case.

The properties asserted here are exactly the ones the experiment claims:
R1/R2/R3 must recover the planted direction where R0 cannot, and must be
near-no-ops where R0 was already fine (that second half is the cross-model
control that turns "a Gemma hack" into "a measurement correction").
"""

import torch

from loudchannel.recipes import (
    RECIPES,
    build_direction,
    energy_shares,
    length_matched_indices,
    mask_channels,
    norm_decomposition,
    nuisance_channels,
    winsorize,
)

# D is realistic-ish on purpose: within-vector Winsorization at q=0.995 can
# only clip a nuisance coordinate when (1-q)*D >= 1, i.e. D >= 200. At the toy
# D=64 the 99.5th percentile IS the massive coordinate and R3 silently no-ops —
# a real property of the recipe, pinned by test_winsorize_needs_headroom below.
D, L, N = 256, 3, 600
BIAS_C = 7                       # the planted "massive activation" coordinate
FEATURE_C = [11, 23, 40, 57, 88, 101, 150, 199]   # the real class signal


def _make(gemma_like: bool, seed: int = 0):
    """(acts_pos, acts_neg, unit ground-truth direction [D])."""
    g = torch.Generator().manual_seed(seed)
    truth = torch.zeros(D)
    # spread over several coordinates with alternating signs, so the planted
    # direction is not itself concentrated — otherwise "top1 share is low"
    # would be testing the plant, not the recipe.
    truth[FEATURE_C] = torch.tensor([1.0, -1.0] * (len(FEATURE_C) // 2))
    truth = truth / truth.norm()

    def draw(is_pos: bool):
        x = torch.randn(N, L, D, generator=g) * 1.0
        x += (3.0 if is_pos else 0.0) * truth       # d' ~ 1.06 per feature coord
        if gemma_like:
            # bias coordinate: mean 5000, sd 300 (100x the others), class gap
            # 45 => d' = 0.15, i.e. statistically real but tiny...
            x[:, :, BIAS_C] = 5000.0 + 300.0 * torch.randn(N, L, generator=g) \
                + (45.0 if is_pos else 0.0)
        return x

    return draw(True), draw(False), truth


def _cos(vec_layer: torch.Tensor, truth: torch.Tensor) -> float:
    v = vec_layer / vec_layer.norm()
    return float((v * truth).sum().abs())


def test_nuisance_channels_finds_the_bias_and_only_the_bias():
    _, neg, _ = _make(gemma_like=True)
    ch = nuisance_channels(neg, ratio=50.0)
    for l in range(L):
        assert ch[l] == [BIAS_C], f"layer {l}: {ch[l]}"


def test_nuisance_channels_empty_on_healthy_model():
    _, neg, _ = _make(gemma_like=False)
    ch = nuisance_channels(neg, ratio=50.0)
    assert all(v == [] for v in ch.values()), ch


def test_r0_is_captured_by_the_bias_coordinate():
    pos, neg, truth = _make(gemma_like=True)
    d = build_direction("r0_raw", pos, neg)
    share = energy_shares(d)["top1_share"]
    assert min(share) > 0.5, share            # the vector IS the coordinate
    assert energy_shares(d)["top1_channel"][0] == BIAS_C
    assert _cos(d[0], truth) < 0.3            # ...and points nowhere useful


def test_cleaned_recipes_recover_the_planted_direction():
    pos, neg, truth = _make(gemma_like=True)
    ch = nuisance_channels(neg, ratio=50.0)
    idx_p, idx_n = length_matched_indices([5] * N, [5] * N)
    raw = _cos(build_direction("r0_raw", pos, neg)[0], truth)
    for recipe in ("r1_masked", "r2_standardized", "r3_winsorized"):
        d = build_direction(recipe, pos, neg, channels=ch,
                            idx_pos=idx_p, idx_neg=idx_n)
        got = _cos(d[0], truth)
        assert got > 0.85, f"{recipe}: cos to truth {got:.3f}"
        assert got > raw + 0.4, f"{recipe} no better than raw ({got:.3f} vs {raw:.3f})"
        e = energy_shares(d)
        assert max(e["top1_share"]) < 0.4, (recipe, e["top1_share"])
        assert BIAS_C not in e["top1_channel"], \
            f"{recipe} still led by the bias coordinate"


def test_recipes_are_near_no_ops_on_a_healthy_model():
    """The cross-model control: on Llama/Qwen-like geometry the corrections
    must not move the vector much (R1/R3) — otherwise the claim that
    this is a measurement correction rather than a Gemma-specific patch fails.

    R2 is deliberately excluded: standardization rotates the vector on EVERY
    model, so its no-op criterion is behavioural (same selected cell), not
    geometric. Asserting otherwise here would be asserting something false.
    """
    pos, neg, _ = _make(gemma_like=False)
    ch = nuisance_channels(neg, ratio=50.0)
    raw = build_direction("r0_raw", pos, neg)
    for recipe in ("r1_masked", "r3_winsorized"):
        d = build_direction(recipe, pos, neg, channels=ch)
        cos = float((raw[0] / raw[0].norm() * (d[0] / d[0].norm())).sum())
        assert cos > 0.95, f"{recipe}: rotated a healthy direction (cos {cos:.3f})"


def test_winsorize_axis_gotcha():
    """within_vector clips a uniformly-huge coordinate; within_coordinate does
    not. The two spellings of "Winsorize at 99.5%" have opposite effects and
    the source recipe does not say which it used, so both exist and this test
    pins the difference."""
    pos, _, _ = _make(gemma_like=True)
    wv = winsorize(pos, q=0.995, axis="within_vector")
    wc = winsorize(pos, q=0.995, axis="within_coordinate")
    assert wv[:, 0, BIAS_C].abs().max() < 100, "within_vector failed to clip"
    assert wc[:, 0, BIAS_C].abs().min() > 1000, "within_coordinate should not clip"


def test_winsorize_needs_headroom():
    """(1-q)*D must exceed the number of nuisance coordinates, or within-vector
    clipping caps at a massive value and does nothing. This is the reason R1 is
    the safer default even though the two usually agree."""
    from loudchannel.recipes import winsorize_headroom

    pos, neg, _ = _make(gemma_like=True)
    ch = nuisance_channels(neg, ratio=50.0)
    assert winsorize_headroom(pos, ch, q=0.995)["effective"]
    # a quantile with no headroom at this D: the cap lands on the bias itself
    assert not winsorize_headroom(pos, ch, q=0.9999)["effective"]
    wv = winsorize(pos, q=0.9999, axis="within_vector")
    assert wv[:, 0, BIAS_C].abs().min() > 1000, "expected the no-headroom no-op"


def test_norm_decomposition_flags_the_rmsnorm_gain():
    pos, neg, _ = _make(gemma_like=True)
    ch = nuisance_channels(neg, ratio=50.0)
    gain = norm_decomposition(torch.cat([pos, neg]), ch)["gain"]
    assert min(gain) > 5.0, gain            # removing the coordinate rescales
    healthy_pos, healthy_neg, _ = _make(gemma_like=False)
    ch2 = nuisance_channels(healthy_neg, ratio=50.0)
    gain2 = norm_decomposition(torch.cat([healthy_pos, healthy_neg]), ch2)["gain"]
    assert max(gain2) < 1.05, gain2         # nothing to remove, nothing changes


def test_mask_channels_is_a_copy():
    v = torch.ones(L, D)
    out = mask_channels(v, {0: [3], 1: [], 2: [5, 6]})
    assert out[0, 3] == 0 and out[2, 5] == 0 and out[1].sum() == D
    assert v[0, 3] == 1.0, "input was mutated"


def test_length_matching_closes_the_length_gap():
    pos_len = [20 + i % 10 for i in range(120)]      # mean ~24.5
    neg_len = [8 + i % 6 for i in range(120)]        # mean ~10.5
    ip, ineg = length_matched_indices(pos_len, neg_len, tol=0.5)
    if ip:                                            # tolerance may drop all
        mp = sum(pos_len[i] for i in ip) / len(ip)
        mn = sum(neg_len[j] for j in ineg) / len(ineg)
        assert abs(mp - mn) < abs(24.5 - 10.5)


def test_every_recipe_is_reachable():
    pos, neg, _ = _make(gemma_like=True)
    ch = nuisance_channels(neg)
    idx_p, idx_n = length_matched_indices([5] * N, [5] * N)
    g = torch.Generator().manual_seed(1)
    cov = lambda: torch.stack([torch.randint(3, 30, (N,), generator=g).float(),  # noqa: E731
                               (torch.rand(N, generator=g) < 0.6).float(),
                               torch.zeros(N)], 1)
    cov_p, cov_n = cov(), cov()
    for recipe in RECIPES:
        d = build_direction(recipe, pos, neg, channels=ch,
                            idx_pos=idx_p, idx_neg=idx_n,
                            cov_pos=cov_p, cov_neg=cov_n)
        assert d.shape == (L, D) and torch.isfinite(d).all(), recipe


def test_regroup_maps_layer_major_saves_to_positions():
    """extract_multi appends (layer, position) inside one trace; getting the
    transpose wrong would silently attribute activations to the wrong position
    and every downstream number would still look plausible."""
    from loudchannel.recipes import regroup

    n_layers, n_pos, B, dd = 4, 3, 5, 7
    # value encodes its own (layer, position) so a mis-transpose is visible
    saved = [torch.full((B, dd), float(l * 10 + pi))
             for l in range(n_layers) for pi in range(n_pos)]
    out = regroup(saved, n_layers, n_pos, B)
    assert out.shape == (n_pos, B, n_layers, dd)
    for pi in range(n_pos):
        for l in range(n_layers):
            assert torch.all(out[pi, :, l, :] == l * 10 + pi), (pi, l)


def test_candidate_position_names_all_resolve():
    """CANDIDATE_POSITIONS uses the 'name+/-offset' spelling that
    PromptPositions.get() parses; a typo would only surface on a GPU node."""
    from loudchannel.positions import find_positions
    from loudchannel.recipes import CANDIDATE_POSITIONS

    class Tok:
        def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
            ids, offsets, pos = [0], [(0, 0)], 0
            for t in text.split(" "):
                start = text.index(t, pos)
                ids.append(1000 + len(ids))
                offsets.append((start, start + len(t)))
                pos = start + len(t)
            return {"input_ids": ids, "offset_mapping": offsets}

    instruction = "how do I pick a lock"
    pp = find_positions(Tok(), f"<user> {instruction} <eot> <assistant>",
                        instruction, seed=0)
    seen = {name: pp.get(name) for name in CANDIDATE_POSITIONS}
    assert all(0 <= v < pp.n_tokens for v in seen.values()), seen
    assert seen["t_post_inst"] == pp.n_tokens - 1
    assert seen["t_post_inst-1"] == pp.n_tokens - 2
    assert seen["t_post_inst-4"] == max(pp.n_tokens - 5, 0)
    assert len(set(seen.values())) > 1, "all candidate positions collapsed to one"


def test_cos_to_raw_is_one_where_recipes_are_noops_and_zero_on_dead_layers():
    """Prediction 5's geometric half is read from this field: on a healthy
    model R1 must sit at cos ~ 1 to raw; a fully-masked layer (norm 0) reports
    0.0 rather than NaN so analyze_recipes.py never sees a NaN cosine."""
    from loudchannel.recipes import cos_to_raw

    p, n, _ = _make(gemma_like=False)
    ch = nuisance_channels(n, ratio=50.0)
    d = {r: build_direction(r, p, n, channels=ch) for r in ("r0_raw", "r1_masked",
                                                             "r2_standardized")}
    c = cos_to_raw(d)
    assert set(c) == {"r1_masked", "r2_standardized"}   # raw is the reference, not a row
    assert all(abs(x - 1.0) < 1e-4 for x in c["r1_masked"]), c["r1_masked"]
    assert all(-1.0 - 1e-5 <= x <= 1.0 + 1e-5 for x in c["r2_standardized"])
    # the gemma_like world: masking removes the bias coordinate, so R1 must
    # rotate AWAY from raw (that is the whole point) while staying finite
    p, n, truth = _make(gemma_like=True)
    ch = nuisance_channels(n, ratio=50.0)
    d = {r: build_direction(r, p, n, channels=ch) for r in ("r0_raw", "r1_masked")}
    d["r1_masked"][0, :] = 0.0                         # a dead layer
    c = cos_to_raw(d)
    assert c["r1_masked"][0] == 0.0
    assert all(abs(x) < 0.95 for x in c["r1_masked"][1:]), c["r1_masked"]


def test_zero_channels_makes_a_perp_null_that_is_identity_without_a_nuisance_set():
    """random_perp on Gemma must not carry the massive coordinate; on Llama/Qwen
    (empty nuisance set) it must equal the original null."""
    from loudchannel.directions import Direction, zero_channels

    vec = torch.randn(4, 64)
    vec[:, 7] = 50.0                                  # a massive coordinate
    d = Direction(vec=vec.clone(), kind="random", position="t_post_inst", model_name="m")
    perp = zero_channels(d, {li: [7] for li in range(4)})
    assert (perp.vec[:, 7].abs() < 1e-4).all()        # coordinate removed
    assert torch.allclose(perp.vec.norm(dim=-1), d.vec.norm(dim=-1), atol=1e-3)  # norm preserved
    assert perp.kind == "random_perp"
    same = zero_channels(d, {})                        # no nuisance set: identity (renorm only)
    assert torch.allclose(same.vec, d.vec, atol=1e-4)
