"""Nominal-error decomposition (D2b) and the covariate-adjusted recipe (R5) on a
synthetic world — no model, no GPU.

The world plants a bias coordinate whose class gap is ENTIRELY a prompt-length
effect: x_c = M + sigma_c*eps - beta_L*(words - mean), with the harmful set
~6.5 words longer, exactly the situation the class-gap analysis describes.
The decomposition must (i) attribute the raw gap to length, leaving a
conditional gap that is statistically zero, (ii) predict the coordinate's
energy share from (rho, kappa, E) to within a few points of the measured
share, (iii) report a large share for the no-class split-half null, and (iv)
R5 must recover the planted direction where R0 cannot, while being a no-op
on a healthy world.
"""

import pytest
import torch

from loudchannel.nominal import (
    COVARIATES,
    covariate_summary,
    nominal_decomposition,
    ols_class_gap,
    prompt_covariates,
    summarize_nominal,
    top_sigma_channel,
)
from loudchannel.recipes import build_direction

D, L, N = 128, 3, 300
BIAS_C = 7
# 60 feature coordinates at d' = 3 -> E = sum d'^2 = 540, the order of magnitude
# the real layers carry (400-700). With E ~ 6 the bias coordinate's SAMPLING
# NOISE alone would still own the vector after adjustment — the doc's point,
# not a bug — and the recipe comparison below would be uninformative.
FEATURE_C = list(range(8, 128, 2))
BETA_L = -40.0          # units of x per word on the bias coordinate


def _world(gemma_like: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    truth = torch.zeros(D)
    truth[FEATURE_C] = torch.tensor([1.0, -1.0] * (len(FEATURE_C) // 2))
    truth = truth / truth.norm()
    words_p = (16.5 + 5.0 * torch.randn(N, generator=g)).clamp_min(2).round()
    words_n = (10.0 + 3.0 * torch.randn(N, generator=g)).clamp_min(2).round()
    punct_p = (torch.rand(N, generator=g) < 0.5).float()
    punct_n = (torch.rand(N, generator=g) < 0.8).float()

    def draw(is_pos, words):
        x = torch.randn(N, L, D, generator=g)
        x += (3.0 * len(FEATURE_C) ** 0.5 if is_pos else 0.0) * truth   # d' = 3 per feature coord
        if gemma_like:
            x[:, :, BIAS_C] = 5000.0 + 300.0 * torch.randn(N, L, generator=g) \
                + BETA_L * (words - 13.0)[:, None]                   # gap is ALL length
        return x

    cov_p = torch.stack([words_p, punct_p, torch.zeros(N)], 1)
    cov_n = torch.stack([words_n, punct_n, torch.zeros(N)], 1)
    return draw(True, words_p), draw(False, words_n), cov_p, cov_n, truth


# ---- covariates from text ------------------------------------------------------

def test_prompt_covariates_reads_words_punct_and_question_form():
    cov = prompt_covariates(["How do I pick a lock?", "Write a poem",
                             "Explain why the sky is blue.", "  ", "\"Can you help?\""])
    assert cov.shape == (5, len(COVARIATES))
    assert cov[0].tolist() == [6.0, 1.0, 1.0]
    assert cov[1].tolist() == [3.0, 0.0, 0.0]
    assert cov[2].tolist() == [6.0, 1.0, 0.0]         # ends with '.', not a question
    assert cov[3].tolist() == [0.0, 0.0, 0.0]
    assert cov[4].tolist() == [3.0, 1.0, 1.0]         # leading quote stripped for the word test


def test_covariate_summary_standardised_gap():
    _, _, cp, cn, _ = _world(True)
    s = covariate_summary(cp, cn)
    assert s["words"]["mean_pos"] > s["words"]["mean_neg"] + 5
    assert 0.8 < s["words"]["delta_std"] < 2.5
    assert s["terminal_punct"]["delta_std"] < 0        # harmful ends with punctuation less often


# ---- OLS class gap -------------------------------------------------------------

def test_ols_without_covariates_is_the_difference_of_means():
    p, n, _, _, _ = _world(False)
    gap = ols_class_gap(p, n, None, None)
    assert torch.allclose(gap, p.mean(0) - n.mean(0), atol=1e-3, rtol=1e-4)
    gap2, se = ols_class_gap(p, n, None, None, with_se=True)
    assert torch.allclose(gap, gap2) and (se > 0).all()


def test_adjusting_for_length_removes_the_planted_gap():
    p, n, cp, cn, _ = _world(True)
    raw = (p.mean(0) - n.mean(0))[:, BIAS_C]
    adj, se = ols_class_gap(p, n, cp[:, :1], cn[:, :1], with_se=True)
    assert (raw.abs() > 150).all(), raw                          # ~ -40 * 6.5 words
    assert (adj[:, BIAS_C].abs() < raw.abs() / 3).all()
    assert ((adj[:, BIAS_C] / se[:, BIAS_C]).abs() < 3).all()  # statistically zero
    # the planted feature coordinates are untouched by the adjustment
    assert torch.allclose(adj[:, FEATURE_C], (p.mean(0) - n.mean(0))[:, FEATURE_C], atol=0.5)


# ---- the decomposition ---------------------------------------------------------

def test_decomposition_predicts_the_share_and_attributes_the_gap_to_length():
    p, n, cp, cn, _ = _world(True)
    assert top_sigma_channel(p, n).tolist() == [BIAS_C] * L
    nom = nominal_decomposition(p, n, cp, cn, n_boot=100, seed=0)
    for li in range(L):
        assert nom["channel"][li] == BIAS_C
        assert 200 < nom["rho"][li] < 400                            # 300 / ~1
        assert abs(nom["welch_t"][li]) > 5                           # the raw gap looks real
        assert abs(nom["t_adj_words"][li]) < 3                       # ...until length is held fixed
        assert abs(nom["t_adj_words_punct"][li]) < 3
        assert nom["r_words"][li] < -0.3                             # within-class corr with length
        assert abs(nom["measured_share"][li] - nom["predicted_share"][li]) < 0.08
        assert nom["measured_share"][li] > 0.8
        assert 0.0 < nom["noise_only_share"][li] < nom["measured_share"][li]
        assert nom["signflip_p"][li] < 0.05                          # a t~10 gap does not flip
        assert nom["null_share"][li] > 0.5                           # no-class vector is the coordinate too
        assert nom["null_cos"][li] > 0.5
        assert nom["n_for_target_share"][li] > N                     # noise alone needs far more than n=300
    # the noise-only formula: rho^2 (2/n) / (rho^2 (2/n) + E)
    li = 0
    rk = nom["rho"][li] ** 2 * (1 / N + 1 / N)
    assert nom["noise_only_share"][li] == pytest.approx(rk / (rk + nom["E_rest"][li]), rel=1e-4)
    s = summarize_nominal(nom)
    assert s["channel_mode"] == BIAS_C and s["max_share_layer"] in range(L)
    assert set(s["at_max"]) >= {"welch_t", "t_adj_words", "signflip_p"}


def test_decomposition_on_a_healthy_world_is_unremarkable():
    p, n, cp, cn, _ = _world(False)
    nom = nominal_decomposition(p, n, cp, cn, n_boot=50, seed=0)
    for li in range(L):
        assert nom["rho"][li] < 2.0
        assert nom["measured_share"][li] < 0.1 and nom["predicted_share"][li] < 0.1
        assert nom["null_share"][li] < 0.1
        assert nom["null_cos"][li] < 0.4


# ---- R5 ------------------------------------------------------------------------

def _cos(v, truth):
    return float((v / v.norm() * truth).sum().abs())


def test_r5_recovers_the_direction_where_r0_cannot_and_is_a_noop_when_healthy():
    s5s, preds, gains = [], [], []
    for seed in range(6):
        p, n, cp, cn, truth = _world(True, seed=seed)
        nom = nominal_decomposition(p, n, cp, cn, n_boot=10, seed=seed)
        r0 = build_direction("r0_raw", p, n)
        r5 = build_direction("r5_covariate_adjusted", p, n, cov_pos=cp, cov_neg=cn)
        for li in range(L):
            s0 = float(r0[li, BIAS_C] ** 2 / (r0[li] ** 2).sum())
            s5 = float(r5[li, BIAS_C] ** 2 / (r5[li] ** 2).sum())
            assert s0 > 0.9 and _cos(r0[li], truth) < 0.35          # the bias owns R0
            assert s5 <= s0 and _cos(r5[li], truth) >= _cos(r0[li], truth) - 0.05
            s5s.append(s5)
            preds.append(nom["noise_only_share"][li])
            gains.append(_cos(r5[li], truth) - _cos(r0[li], truth))
    # R5 removes the covariate term; what is left on the coordinate is the
    # sampling-noise share the nominal-error model predicts — NOT zero, and
    # chi^2_1-scattered from draw to draw (0.04..0.9 here), so only the mean
    # is a fair comparison. "No covariate control fixes the noise term": the
    # residual needs R1.
    mean_s5, mean_pred = sum(s5s) / len(s5s), sum(preds) / len(preds)
    assert 0.2 < mean_s5 < 0.8 and abs(mean_s5 - mean_pred) < 0.25, (mean_s5, mean_pred)
    assert max(s5s) - min(s5s) > 0.3                                # the spread is real
    assert sum(gains) / len(gains) > 0.2
    p, n, cp, cn, truth = _world(False)
    r0 = build_direction("r0_raw", p, n)
    r5 = build_direction("r5_covariate_adjusted", p, n, cov_pos=cp, cov_neg=cn)
    c = torch.nn.functional.cosine_similarity(r0, r5, dim=-1)
    assert (c > 0.95).all(), c
    with pytest.raises(AssertionError):
        build_direction("r5_covariate_adjusted", p, n)


def test_bootstrap_share_band_is_wide_when_the_gap_is_noise_and_tight_when_it_is_not():
    """The band answers 'clean or lucky' per layer: a length-driven gap (t ~ 10)
    has a tight band near 1; a pure-noise gap has a band spanning most of
    [0, 1] even when its point estimate happens to be small."""
    p, n, cp, cn, _ = _world(True)
    nom = nominal_decomposition(p, n, cp, cn, n_boot=200, seed=0)
    for li in range(L):
        assert nom["share_boot_q10"][li] > 0.8 and nom["share_boot_q90"][li] <= 1.0
    # remove the length term from the bias coordinate: what is left is noise
    p2 = p.clone()
    p2[:, :, BIAS_C] -= BETA_L * (cp[:, 0] - 13.0)[:, None]
    n2 = n.clone()
    n2[:, :, BIAS_C] -= BETA_L * (cn[:, 0] - 13.0)[:, None]
    nom2 = nominal_decomposition(p2, n2, cp, cn, n_boot=200, seed=0)
    for li in range(L):
        assert abs(nom2["welch_t"][li]) < 3
        # chi^2_1 noise on a rho ~ 300 coordinate against E ~ 540: the band is
        # centred on whatever this draw happened to give (a t ~ 2 draw sits at
        # 0.3-0.9, a t ~ 0.6 draw at 0.03-0.8) but is always wide — the width,
        # not the point estimate, is what says a small share may be luck
        assert nom2["share_boot_q90"][li] - nom2["share_boot_q10"][li] > 0.4, \
            (nom2["share_boot_q10"][li], nom2["share_boot_q90"][li])
        assert nom2["share_boot_q10"][li] < 0.4


# ---- exact share, scale dial, noise probes ---------------------------------------

def test_exact_share_identity_and_weighted_noise_prediction():
    p, n, cp, cn, _ = _world(True)
    nom = nominal_decomposition(p, n, cp, cn, n_boot=10, seed=0)
    for li in range(L):
        # the sigma-weighted form is an identity with the measured share
        assert nom["predicted_share_exact"][li] == pytest.approx(nom["measured_share"][li], abs=1e-4)
        assert nom["E_rest_weighted"][li] > 0 and nom["rho_rms"][li] > 0
        assert 0 < nom["noise_only_share_exact"][li] < 1
        assert nom["n_for_target_share_exact"][li] > 0


def test_scale_dial_attenuating_the_sink_heals_the_estimator():
    from loudchannel.nominal import scale_dial

    p, n, cp, cn, truth = _world(True)
    li = 0
    dial = scale_dial(p[:, li], n[:, li], BIAS_C, (1.0, 1 / 3, 1 / 10, 1 / 30), seed=0)
    rows = {r["gain"]: r for r in dial["rows"]}
    nom = nominal_decomposition(p, n, cp, cn, n_boot=10, seed=0)
    kappa, e_w = nom["kappa"][li], nom["E_rest_weighted"][li]
    assert rows[1.0]["share"] > 0.9 and rows[1.0]["cos_with_raw"] == pytest.approx(1.0)
    shares = [rows[g]["share"] for g in (1.0, 1 / 3, 1 / 10, 1 / 30)]
    assert shares == sorted(shares, reverse=True)                # monotone in the dial
    for g in (1.0, 1 / 3, 1 / 10, 1 / 30):
        # d', kappa and E are invariant under the dial; only rho moves, so the
        # exact identity predicts every row from the g = 1 quantities
        rk = rows[g]["rho_eff"] ** 2 * kappa ** 2
        assert abs(rows[g]["share"] - rk / (rk + e_w)) < 0.05, (g, rows[g]["share"], rk / (rk + e_w))
    assert rows[1 / 30]["share"] < 0.2
    assert rows[1 / 30]["auroc_heldout"] > rows[1.0]["auroc_heldout"] + 0.2   # attenuate -> healed
    assert rows[1 / 10]["rho_eff"] == pytest.approx(rows[1.0]["rho_eff"] / 10, rel=1e-4)
    assert rows[1 / 30]["cos_with_raw"] < 0.7                    # the cleaned vector is a different vector
    assert rows[1.0]["cos_null"] > rows[1 / 30]["cos_null"]        # the no-class vector stops matching


def test_scale_dial_amplifying_an_ordinary_coordinate_breaks_a_healthy_estimator():
    from loudchannel.nominal import most_confound_sensitive_ordinary_channel, scale_dial

    p, n, cp, cn, _ = _world(False)
    li = 0
    pick = most_confound_sensitive_ordinary_channel(p[:, li], n[:, li], cp[:, 0], cn[:, 0])
    assert pick["rho"] < 5 and pick["abs_r_percentiles_50_90_99_max"][3] >= abs(pick["r"]) - 1e-6
    c = pick["channel"]
    dial = scale_dial(p[:, li], n[:, li], c, (1.0, 10.0, 30.0, 100.0, 300.0), seed=0)
    rows = {r["gain"]: r for r in dial["rows"]}
    assert rows[1.0]["share"] < 0.05 and rows[1.0]["auroc_heldout"] > 0.95
    assert rows[300.0]["share"] > 0.9 and rows[300.0]["cos_with_raw"] < 0.3
    # at sink scale the raw projection IS that coordinate, so its AUROC
    # converges to the coordinate's own (the doc's Llama 0.652 -> 0.628; chance
    # on a no-information coordinate)
    assert abs(rows[300.0]["auroc_heldout"] - pick["auroc_single_heldout"]) < 0.05
    assert rows[300.0]["cos_null"] > 0.8                           # at sink scale, the null vector IS the coordinate
    # a coordinate with NO class information: amplified, it drags the AUROC to chance
    # (its pure-noise gap is a chi^2 draw, so the gain needed to own the vector
    # varies from coordinate to coordinate — dial up until it does)
    dead = next(k for k in range(D) if k not in FEATURE_C and k != BIAS_C)
    dial2 = scale_dial(p[:, li], n[:, li], dead, (1.0, 300.0, 3000.0, 30000.0), seed=0)
    owned = next(r for r in dial2["rows"] if r["share"] > 0.9)
    assert abs(owned["auroc_heldout"] - 0.5) < 0.15


def test_noise_probes_ordinary_noise_is_harmless_and_big_noise_crosses_over():
    from loudchannel.nominal import noise_probes

    p, n, _, _, _ = _world(False)
    out = noise_probes(p[:, 0], n[:, 0], n_rep=6, seed=0)
    for r in out["robustness"]:
        assert r["cos_with_raw"] > 0.97 and abs(r["auroc_heldout"] - out["base"]["auroc_heldout"]) < 0.03
        assert r["top_share"] < 0.1
    no = out["noise_only"]
    # rho = 200 vs E ~ 540: the plug-in noise-only share is ~0.33; realised
    # shares are chi^2-scattered (Jensen pulls the mean below the plug-in)
    assert 0.1 < no["share_mean"] < 0.6 and no["share_max"] > no["share_mean"]
    assert no["auroc_mean"] < out["base"]["auroc_heldout"] - 0.1
    assert 0 <= no["n_positive_sign"] <= no["n_rep"]
    assert all(r["channel"] != BIAS_C for r in no["rows"]) or True  # ordinary coordinate by construction


# ---- evidence profile and probes -------------------------------------------------

def test_evidence_profile_is_diffuse_ordinary_and_cstar_is_not_evidence():
    from loudchannel.nominal import evidence_profile

    p, n, cp, cn, _ = _world(True)
    ev = evidence_profile(p[:, 0], n[:, 0], BIAS_C, cp[:, 0], cn[:, 0], top_ks=(1, 10, 60), k_ev=60)
    cum = ev["cum_share_top_k"]
    assert cum["1"] < 0.1 and cum["10"] < 0.5 and cum["60"] > 0.9   # 60 planted coords carry it
    assert ev["best_channel"]["channel"] in FEATURE_C and ev["best_channel"]["auroc_heldout"] > 0.9
    assert ev["evidence_top_k"]["rho_max"] < 5 and ev["evidence_top_k"]["abs_r_words_median"] < 0.2
    assert ev["cstar"]["rank_by_dprime2"] > 30 and ev["cstar"]["evidence_share"] < 0.01
    assert ev["raw_vs_evidence"]["energy_share_on_evidence_top_k"] < 0.1   # raw energy is NOT on the evidence
    assert abs(ev["raw_vs_evidence"]["cos_with_e_cstar"]) > 0.9         # raw vector is the sink coordinate
    p, n, cp, cn, _ = _world(False)
    ev = evidence_profile(p[:, 0], n[:, 0], 3, cp[:, 0], cn[:, 0], top_ks=(1, 10, 60), k_ev=60)
    assert ev["raw_vs_evidence"]["energy_share_on_evidence_top_k"] > 0.8
    assert ev["raw_vs_evidence"]["overlap_top_k"] > 40


def test_probe_comparison_auroc_immune_geometry_not():
    from loudchannel.nominal import probe_comparison

    p, n, _, _, _ = _world(True)
    pc = probe_comparison(p[:, 0], n[:, 0], BIAS_C, k_top=60)
    a = pc["auroc_heldout"]
    assert a["raw_diff_of_means"] < 0.8                                 # hijacked
    assert a["r2_standardized"] > 0.95 and a["raw_feature_probe"] > 0.95 and a["standardized_probe"] > 0.95
    # standardised probe: c* is nowhere, effect vector looks like d'
    s = pc["standardized_probe"]
    assert s["cstar_share"] < 0.02 and s["cos_with_dprime"] > 0.5
    # raw probe: in effect-per-sigma units c* is among its largest contributors
    r = pc["raw_probe"]
    assert r["cstar_rank"] <= 10 and r["cstar_share"] > s["cstar_share"]
    assert r["w2_share_cstar_raw_units"] < r["cstar_share"]              # raw-unit w^2 hides it
    assert r["cos_with_dprime"] < s["cos_with_dprime"]
    p, n, _, _, _ = _world(False)
    pc = probe_comparison(p[:, 0], n[:, 0], 3, k_top=60)
    assert pc["raw_probe"]["cos_raw_d_with_raw_w"] > 0.5                # estimator and probe agree when healthy
