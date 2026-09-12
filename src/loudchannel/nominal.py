"""D2b — the nominal-error decomposition of the class gap on a nuisance coordinate.

Link 2 of the mechanism chain (`class-gap-nominal-error-analysis`, 2026-09-11):
the class gap the difference of means measures on Gemma's massive coordinate
is not a class effect. Per prompt and coordinate c,

    x_c = b_c + beta_c*y + sum_j gamma_cj*Z_j + eps_c          (Z_j: nuisance covariates)

so the estimated gap is

    Delta_c = beta_c + sum_j gamma_cj*(Zbar_j1 - Zbar_j0) + (epsbar_1 - epsbar_0).

With r_cj = within-class corr(x_c, Z_j) and delta_j the standardised covariate
gap between the two prompt sets, the covariate term is r_cj*delta_j*sigma_c
and the noise term has sd sigma_c*sqrt(1/n1 + 1/n0): EVERY error term is the
coordinate's own scale times a dimensionless number fixed by the datasets.
The layer's real evidence is E = sum_k d'_k^2 (a few hundred here), so the
share of the vector that lands on c is

    s ~ rho^2 kappa^2 / (rho^2 kappa^2 + E),   rho = sigma_c/sigma_typ,  kappa = Delta_c/sigma_c

and a coordinate with rho ~ 200 owns the vector at kappa ~ 0.1 — sampling
noise alone (kappa = sqrt(2/n)) gives 24% at n = 300 and 40% at n = 128.
Llama/Qwen coordinates with the SAME r*delta take 2-6% because their rho is
10-13. No n fixes the covariate terms; only removing the coordinate (R1),
adjusting for the covariates (R5), or both, does.

Everything here is pure tensor maths on the cached extraction activations
plus per-prompt covariates read off the instruction strings, so it runs
inside stage 01_extract at no GPU cost and is unit-tested on synthetic worlds.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

COVARIATES: tuple[str, ...] = ("words", "terminal_punct", "question_form")
_QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "whom", "which",
                   "can", "could", "should", "would", "is", "are", "do", "does",
                   "did", "will", "may", "might")
_WORD_RE = re.compile(r"\S+")


# ---- covariates from the prompt text ----------------------------------------


def prompt_covariates(instructions: list[str]) -> Tensor:
    """[n, 3] float: word count, ends with . ? ! (0/1), question form (0/1:
    ends with ? or starts with a question word). The measured gaps between
    the harmful and harmless sets are ~6.5 words, 50% vs 80% terminal
    punctuation, 1% vs 12% question form."""
    rows = []
    for s in instructions:
        t = (s or "").strip()
        words = len(_WORD_RE.findall(t))
        core = t.rstrip("\"')]")            # a closing quote does not hide the . ? !
        punct = 1.0 if core.endswith((".", "?", "!")) else 0.0
        first = t.split(" ", 1)[0].lower().strip("\"'([") if t else ""
        question = 1.0 if (core.endswith("?") or first in _QUESTION_WORDS) else 0.0
        rows.append([float(words), punct, question])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, len(COVARIATES))


def covariate_summary(cov_pos: Tensor, cov_neg: Tensor) -> dict:
    """Per covariate: class means and the standardised gap delta_j (pooled sd)."""
    out = {}
    for j, name in enumerate(COVARIATES):
        a, b = cov_pos[:, j], cov_neg[:, j]
        sd = torch.sqrt(0.5 * (a.var(unbiased=True) + b.var(unbiased=True))).clamp_min(1e-12)
        out[name] = {"mean_pos": float(a.mean()), "mean_neg": float(b.mean()),
                     "delta_std": float((a.mean() - b.mean()) / sd)}
    return out


# ---- OLS class coefficient with covariates -----------------------------------


def _design(n_pos: int, n_neg: int, cov_pos: Tensor | None, cov_neg: Tensor | None) -> Tensor:
    """[n_pos + n_neg, 2 + k]: intercept, class indicator, covariates."""
    n = n_pos + n_neg
    cols = [torch.ones(n), torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])]
    if cov_pos is not None and cov_pos.shape[1]:
        Z = torch.cat([cov_pos.float(), cov_neg.float()])                     # [n, k]
        Z = Z - Z.mean(0, keepdim=True)          # centring keeps the intercept tame
        cols.extend(Z[:, j] for j in range(Z.shape[1]))
    return torch.stack(cols, dim=1)


def ols_class_gap(acts_pos: Tensor, acts_neg: Tensor, cov_pos: Tensor | None,
                  cov_neg: Tensor | None, *, with_se: bool = False):
    """Per-layer, per-coordinate coefficient on the class indicator when the
    covariates are held fixed — the conditional class gap. [L, D].

    With no covariates this is exactly the difference of means. The design is
    shared by every (layer, coordinate), so one small solve gives every
    coefficient at once. `with_se` also returns the coefficient's standard
    error [L, D] (homoscedastic OLS), i.e. an adjusted t-statistic is
    gap / se."""
    p, n = acts_pos.float(), acts_neg.float()
    n_p, n_n = p.shape[0], n.shape[0]
    L, D = p.shape[1], p.shape[2]
    A = _design(n_p, n_n, cov_pos, cov_neg)                                   # [N, q]
    X = torch.cat([p, n]).reshape(n_p + n_n, L * D)                          # [N, L*D]
    AtA_inv = torch.linalg.inv(A.T @ A)
    B = AtA_inv @ (A.T @ X)                                                   # [q, L*D]
    gap = B[1].reshape(L, D)
    if not with_se:
        return gap
    resid = X - A @ B
    dof = max(A.shape[0] - A.shape[1], 1)
    s2 = (resid ** 2).sum(0) / dof                                            # [L*D]
    se = torch.sqrt(s2 * AtA_inv[1, 1]).reshape(L, D).clamp_min(1e-12)
    return gap, se


# ---- the decomposition ---------------------------------------------------------


def top_sigma_channel(acts_pos: Tensor, acts_neg: Tensor) -> Tensor:
    """Per layer, the coordinate with the largest pooled within-class sd — the
    doc's c* (2339 at every layer on Gemma-3-12B). [L] long."""
    p, n = acts_pos.float(), acts_neg.float()
    sd = torch.sqrt(0.5 * (p.var(0, unbiased=True) + n.var(0, unbiased=True)))
    return sd.argmax(-1)


def _within_class_corr(x_p: Tensor, x_n: Tensor, z_p: Tensor, z_n: Tensor) -> float:
    """corr(x, z) after centring each class separately (pooled)."""
    x = torch.cat([x_p - x_p.mean(), x_n - x_n.mean()])
    z = torch.cat([z_p - z_p.mean(), z_n - z_n.mean()])
    den = (x.norm() * z.norm()).clamp_min(1e-12)
    return float((x * z).sum() / den)


def nominal_decomposition(
    acts_pos: Tensor, acts_neg: Tensor, cov_pos: Tensor, cov_neg: Tensor,
    *, channels: Tensor | list[int] | None = None, n_boot: int = 200, seed: int = 0,
    target_share: float = 0.05,
) -> dict:
    """Per layer, for the coordinate c* (default: top-sigma), every quantity
    the nominal-error model needs, side by side with what the raw estimator
    measured. All lists are indexed by layer.

      rho, kappa, E_rest      scale ratio, standardised gap, the rest of the
                              layer's evidence (sum of d'^2 over k != c*)
      measured_share          c*'s energy share in the raw difference of means
      predicted_share         rho^2 kappa^2 / (rho^2 kappa^2 + E_rest)
      noise_only_share        the same with kappa^2 = 1/n1 + 1/n0 — what pure
                              sampling noise would give at this n
      n_for_target_share      per-class n at which noise alone falls below
                              `target_share` (covariate terms do not shrink)
      raw_gap / welch_t       the marginal gap and its t
      gap_adj_words / t_adj_words              conditional on word count
      gap_adj_words_punct / t_adj_words_punct  conditional on words + terminal punct
      r_words, r_punct        within-class correlation of x_c* with each covariate
      signflip_p              bootstrap probability that the raw gap changes sign
      share_boot_q10/50/90    bootstrap band on c*'s share of the raw vector —
                              wide = "clean by luck" if the point estimate is small
      null_share, null_cos    split-half of the HARMLESS set only: c*'s share of
                              that no-class vector and |cos| with the raw vector

    `noise_only_share` is the plug-in expectation; the share realised by any
    one extraction is chi^2_1-distributed around it and can land anywhere from
    ~0.05 to ~0.9 at the same layer (see tests/test_nominal.py), which is why
    the bootstrap band is reported next to it.
    """
    p, n = acts_pos.float(), acts_neg.float()
    n_p, n_n, L = p.shape[0], n.shape[0], p.shape[1]
    cs = top_sigma_channel(p, n) if channels is None else torch.as_tensor(list(channels))
    assert cs.numel() == L, "one channel per layer"
    delta = p.mean(0) - n.mean(0)                                             # [L, D]
    vp, vn = p.var(0, unbiased=True), n.var(0, unbiased=True)
    sd = torch.sqrt(0.5 * (vp + vn)).clamp_min(1e-12)
    se = torch.sqrt(vp / n_p + vn / n_n).clamp_min(1e-12)
    dprime2 = (delta / sd) ** 2
    e = delta ** 2
    share = e / e.sum(-1, keepdim=True).clamp_min(1e-30)

    gw, sw = ols_class_gap(p, n, cov_pos[:, :1], cov_neg[:, :1], with_se=True)
    gwp, swp = ols_class_gap(p, n, cov_pos[:, :2], cov_neg[:, :2], with_se=True)

    g = torch.Generator().manual_seed(seed)
    ip = torch.randint(0, n_p, (n_boot, n_p), generator=g)
    im = torch.randint(0, n_n, (n_boot, n_n), generator=g)
    perm = torch.randperm(n_n, generator=g)
    half_a, half_b = perm[: n_n // 2], perm[n_n // 2 : 2 * (n_n // 2)]
    d_null = n[half_a].mean(0) - n[half_b].mean(0)                            # [L, D]
    null_share_all = d_null ** 2 / (d_null ** 2).sum(-1, keepdim=True).clamp_min(1e-30)
    null_cos = torch.nn.functional.cosine_similarity(d_null, delta, dim=-1).abs()

    ar = torch.arange(L)
    c = cs.long()
    sd_typ = sd.median(dim=-1).values
    rho = sd[ar, c] / sd_typ.clamp_min(1e-12)
    kappa = delta[ar, c] / sd[ar, c]
    e_rest = dprime2.sum(-1) - dprime2[ar, c]
    rk = rho ** 2 * kappa ** 2
    noise_k2 = 1.0 / n_p + 1.0 / n_n
    rn = rho ** 2 * noise_k2
    # noise-only share at per-class n: rho^2 (2/n) / (rho^2 (2/n) + E) = target
    n_target = 2.0 * rho ** 2 * (1.0 - target_share) / (target_share * e_rest.clamp_min(1e-12))
    # The scaling form above takes sigma_typ = median sigma_k and E = sum d'^2,
    # which over-predicts s by 2-3x in rho^2 terms because sigma_k^2 is
    # heavy-tailed. The exact identity is
    #     s = Delta_c^2 / (Delta_c^2 + sum_{k != c} Delta_k^2),
    # i.e. the same formula with E replaced by the sigma-WEIGHTED evidence
    # E_w = sum_{k != c} sigma_k^2 d'_k^2 / sigma_typ^2 (invariant under
    # rescaling c). E_w is what the noise-only prediction should use; the
    # rho_rms variant (sigma_typ := RMS of the other sigma_k) says the same
    # thing from the other side.
    e_raw_rest = (delta ** 2).sum(-1) - delta[ar, c] ** 2                     # [L]
    e_w = e_raw_rest / sd_typ.clamp_min(1e-12) ** 2
    sd2_rest = (sd ** 2).sum(-1) - sd[ar, c] ** 2
    sd_rms = torch.sqrt(sd2_rest / max(sd.shape[-1] - 1, 1))
    rho_rms = sd[ar, c] / sd_rms.clamp_min(1e-12)
    rn_w = rho ** 2 * noise_k2
    n_target_exact = 2.0 * rho ** 2 * (1.0 - target_share) / (target_share * e_w.clamp_min(1e-12))

    x_p, x_n = p[:, ar, c], n[:, ar, c]                                       # [n, L]
    boot_gap = x_p[ip].mean(1) - x_n[im].mean(1)                              # [B, L]
    signflip = (torch.sign(boot_gap) != torch.sign(delta[ar, c])).float().mean(0)
    # bootstrap band on c*'s share of the raw vector, holding the rest of the
    # layer's energy at its point estimate (it moves far less than gap_c^2 on
    # a rho ~ 200 coordinate). A layer whose share is small at the point
    # estimate but wide here is "clean by luck" (theory brief §10 Q3); the
    # single-draw share is chi^2_1-distributed and scatters over an order of
    # magnitude, so the point estimate alone says little.
    e_rest_raw = e.sum(-1) - e[ar, c]                                         # [L]
    boot_share = boot_gap ** 2 / (boot_gap ** 2 + e_rest_raw.clamp_min(1e-30))
    share_q = torch.quantile(boot_share, torch.tensor([0.1, 0.5, 0.9]), dim=0)  # [3, L]

    r_words = [_within_class_corr(x_p[:, li], x_n[:, li], cov_pos[:, 0], cov_neg[:, 0])
               for li in range(L)]
    r_punct = [_within_class_corr(x_p[:, li], x_n[:, li], cov_pos[:, 1], cov_neg[:, 1])
               for li in range(L)]

    return {
        "channel": c.tolist(),
        "rho": rho.tolist(), "kappa": kappa.tolist(), "E_rest": e_rest.tolist(),
        "sigma_c": sd[ar, c].tolist(), "sigma_typ": sd_typ.tolist(),
        "measured_share": share[ar, c].tolist(),
        "predicted_share": (rk / (rk + e_rest)).tolist(),
        "noise_only_share": (rn / (rn + e_rest)).tolist(),
        "n_for_target_share": n_target.tolist(), "target_share": target_share,
        # exact (sigma-weighted) forms — see the comment above
        "E_rest_weighted": e_w.tolist(),
        "predicted_share_exact": (rk / (rk + e_w)).tolist(),      # == measured_share, an identity check
        "noise_only_share_exact": (rn_w / (rn_w + e_w)).tolist(),
        "n_for_target_share_exact": n_target_exact.tolist(),
        "sigma_rms": sd_rms.tolist(), "rho_rms": rho_rms.tolist(),
        "raw_gap": delta[ar, c].tolist(), "welch_t": (delta[ar, c] / se[ar, c]).tolist(),
        "gap_adj_words": gw[ar, c].tolist(), "t_adj_words": (gw[ar, c] / sw[ar, c]).tolist(),
        "gap_adj_words_punct": gwp[ar, c].tolist(),
        "t_adj_words_punct": (gwp[ar, c] / swp[ar, c]).tolist(),
        "r_words": r_words, "r_punct": r_punct,
        "signflip_p": signflip.tolist(), "n_boot": n_boot,
        "share_boot_q10": share_q[0].tolist(), "share_boot_q50": share_q[1].tolist(),
        "share_boot_q90": share_q[2].tolist(),
        "null_share": null_share_all[ar, c].tolist(), "null_cos": null_cos.tolist(),
        "null_top1_share": null_share_all.max(-1).values.tolist(),
        "covariates": covariate_summary(cov_pos, cov_neg),
        "n_pos": n_p, "n_neg": n_n,
    }


def summarize_nominal(nom: dict, band: tuple[int, int] | None = None) -> dict:
    """Medians (and the max-share layer) for a log line / a table row."""
    L = len(nom["channel"])
    idx = list(range(L)) if band is None else [li for li in range(L) if band[0] <= li <= band[1]]

    def med(key):
        v = torch.tensor([nom[key][li] for li in idx])
        return float(v.median())

    lmax = max(idx, key=lambda li: nom["measured_share"][li])
    return {
        "layers": [idx[0], idx[-1]],
        "channel_mode": max(set(nom["channel"][li] for li in idx),
                            key=[nom["channel"][li] for li in idx].count),
        "rho": med("rho"), "E_rest": med("E_rest"),
        "rho_rms": med("rho_rms"), "E_rest_weighted": med("E_rest_weighted"),
        "measured_share": med("measured_share"), "predicted_share": med("predicted_share"),
        "noise_only_share": med("noise_only_share"),
        "noise_only_share_exact": med("noise_only_share_exact"),
        "null_share": med("null_share"), "null_cos": med("null_cos"),
        "signflip_p": med("signflip_p"),
        "max_share_layer": lmax,
        "at_max": {k: nom[k][lmax] for k in ("channel", "measured_share", "welch_t",
                                              "t_adj_words", "t_adj_words_punct",
                                              "r_words", "r_punct", "signflip_p")},
    }


# ---- ablations: is it the magnitude, or a special property of the coordinate? --
#
# Everything below is about the ESTIMATOR on cached activations at ONE layer —
# statements about what diff-of-means does to a coordinate of a given scale,
# never about the model's causal structure. The scale dial rescales a single
# coordinate (d', r and its own single-coordinate AUROC are invariant; only
# sigma changes) and watches the raw estimator break or heal; the robustness
# and noise-only probes perturb ordinary coordinates to show the estimator is
# not fragile to nominal-scale noise and to locate the pure-noise crossover.


def _auroc(scores_pos: Tensor, scores_neg: Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    y = torch.cat([torch.ones(len(scores_pos)), torch.zeros(len(scores_neg))]).numpy()
    s = torch.cat([scores_pos, scores_neg]).detach().float().numpy()
    return float(roc_auc_score(y, s))


def _even_odd(n: int) -> tuple[Tensor, Tensor]:
    idx = torch.arange(n)
    return idx[::2], idx[1::2]


def heldout_projection_auroc(xp: Tensor, xn: Tensor, weight: Tensor | None = None) -> float:
    """Direction from the EVEN rows of each class (diff of means, optionally
    reweighted per coordinate), scored on the ODD rows: the raw estimator's
    own held-out AUROC. xp/xn: [n, D] at one layer."""
    ep, op = _even_odd(xp.shape[0])
    en, on = _even_odd(xn.shape[0])
    d = xp[ep].mean(0) - xn[en].mean(0)
    if weight is not None:
        d = d * weight
    u = d / d.norm().clamp_min(1e-12)
    return _auroc(xp[op] @ u, xn[on] @ u)


def _split_half_null(xn: Tensor, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(xn.shape[0], generator=g)
    h = xn.shape[0] // 2
    return xn[perm[:h]].mean(0) - xn[perm[h : 2 * h]].mean(0)


def _abs_cos(a: Tensor, b: Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).abs())


def scale_dial(xp: Tensor, xn: Tensor, channel: int, gains: tuple[float, ...],
               *, seed: int = 0) -> dict:
    """Rescale ONE coordinate by each gain and re-run the raw estimator.

    Per gain: rho_eff (its sd over the median sd of the others), the
    coordinate's share of the raw vector, the raw projection's held-out
    AUROC, |cos| of the rescaled-data vector with the original one, and
    |cos| of a harmless split-half null with the rescaled-data vector.
    gains < 1 attenuate the sink coordinate (Gemma: "attenuate 2339 by 10x
    and the estimator heals"); gains > 1 amplify an ordinary one on a control
    model ("amplify Llama's most length-correlated coordinate 100x and it
    reproduces Gemma's pathology to three decimals").
    """
    xp, xn = xp.float(), xn.float()
    d_raw = xp.mean(0) - xn.mean(0)
    sd = torch.sqrt(0.5 * (xp.var(0, unbiased=True) + xn.var(0, unbiased=True)))
    others = torch.cat([sd[:channel], sd[channel + 1 :]])
    sd_typ = float(others.median())
    rows = []
    for g in gains:
        p, n = xp.clone(), xn.clone()
        p[:, channel] *= g
        n[:, channel] *= g
        d = p.mean(0) - n.mean(0)
        rows.append({
            "gain": float(g),
            "rho_eff": float(g * sd[channel]) / max(sd_typ, 1e-12),
            "share": float(d[channel] ** 2 / (d ** 2).sum().clamp_min(1e-30)),
            "auroc_heldout": heldout_projection_auroc(p, n),
            "cos_with_raw": _abs_cos(d, d_raw),
            "cos_null": _abs_cos(_split_half_null(n, seed), d),
        })
    return {"channel": int(channel), "sigma_c": float(sd[channel]), "sigma_typ": sd_typ,
            "rows": rows}


def most_confound_sensitive_ordinary_channel(xp: Tensor, xn: Tensor, z_p: Tensor, z_n: Tensor,
                                             *, rho_max: float = 5.0) -> dict:
    """Among ORDINARY coordinates (sd < rho_max x median sd) the one with the
    largest within-class |corr(x_c, covariate)| — the natural coordinate to
    amplify on a control model, and the demonstration that tracking the
    covariate is generic (every model has 3-8 coordinates with |r| >= 0.4)."""
    xp, xn = xp.float(), xn.float()
    sd = torch.sqrt(0.5 * (xp.var(0, unbiased=True) + xn.var(0, unbiased=True)))
    rho = sd / sd.median().clamp_min(1e-12)
    cp = xp - xp.mean(0, keepdim=True)
    cn = xn - xn.mean(0, keepdim=True)
    zp = (z_p.float() - z_p.float().mean())[:, None]
    zn = (z_n.float() - z_n.float().mean())[:, None]
    num = (cp * zp).sum(0) + (cn * zn).sum(0)
    den = torch.sqrt((cp ** 2).sum(0) + (cn ** 2).sum(0)) * torch.sqrt((zp ** 2).sum() + (zn ** 2).sum())
    r = num / den.clamp_min(1e-12)                                           # [D]
    ordinary = rho < rho_max
    r_ord = torch.where(ordinary, r.abs(), torch.zeros_like(r))
    c = int(r_ord.argmax())
    q = torch.quantile(r.abs(), torch.tensor([0.5, 0.9, 0.99]))
    # the coordinate's OWN held-out AUROC: when amplified to sink scale the
    # raw projection's AUROC converges to this number (0.652 -> 0.628 on the
    # doc's Llama coordinate; chance on Qwen's no-information one)
    ep, op = _even_odd(xp.shape[0])
    en, on = _even_odd(xn.shape[0])
    sgn = 1.0 if float(xp[ep, c].mean() - xn[en, c].mean()) >= 0 else -1.0
    return {"channel": c, "r": float(r[c]), "rho": float(rho[c]),
            "auroc_single_heldout": _auroc(sgn * xp[op, c], sgn * xn[on, c]),
            "abs_r_percentiles_50_90_99_max": [float(q[0]), float(q[1]), float(q[2]),
                                               float(r.abs().max())],
            "n_channels_abs_r_ge_0.4": int((r.abs() >= 0.4).sum())}


def noise_probes(xp: Tensor, xn: Tensor, *, rho_max: float = 5.0, big_rho: float = 200.0,
                 n_rep: int = 10, seed: int = 0) -> dict:
    """Two perturbations of ORDINARY coordinates (no class or covariate
    structure added by construction):

    robustness      N(0, (k sigma_c)^2) on a fraction of ordinary coordinates,
                    k in {1, 3}, fraction in {0.1, 1.0}: the top coordinate's
                    share, the held-out AUROC and |cos| with the raw vector
                    should not move — nominal noise on ordinary coordinates is
                    harmless;
    noise_only      N(0, (big_rho sigma_typ)^2) on ONE random ordinary
                    coordinate, n_rep times: its share of the vector, the
                    held-out AUROC and the sign of its (pure-noise) gap —
                    the pure-noise crossover, rho* ~ sqrt(E)/sqrt(2/n).
    """
    xp, xn = xp.float(), xn.float()
    g = torch.Generator().manual_seed(seed)
    sd = torch.sqrt(0.5 * (xp.var(0, unbiased=True) + xn.var(0, unbiased=True)))
    sd_typ = float(sd.median())
    ordinary = torch.where(sd / max(sd_typ, 1e-12) < rho_max)[0]
    d_raw = xp.mean(0) - xn.mean(0)
    base_auroc = heldout_projection_auroc(xp, xn)
    top_share = lambda d: float((d ** 2).max() / (d ** 2).sum().clamp_min(1e-30))  # noqa: E731

    robustness = []
    for k in (1.0, 3.0):
        for frac in (0.1, 1.0):
            m = max(int(frac * len(ordinary)), 1)
            pick = ordinary[torch.randperm(len(ordinary), generator=g)[:m]]
            p, n = xp.clone(), xn.clone()
            p[:, pick] += k * sd[pick] * torch.randn(p.shape[0], m, generator=g)
            n[:, pick] += k * sd[pick] * torch.randn(n.shape[0], m, generator=g)
            d = p.mean(0) - n.mean(0)
            robustness.append({"k": k, "fraction": frac, "n_perturbed": m,
                               "top_share": top_share(d), "auroc_heldout": heldout_projection_auroc(p, n),
                               "cos_with_raw": _abs_cos(d, d_raw)})

    noise_only = []
    for _ in range(n_rep):
        c = int(ordinary[torch.randint(len(ordinary), (1,), generator=g)])
        p, n = xp.clone(), xn.clone()
        p[:, c] += big_rho * sd_typ * torch.randn(p.shape[0], generator=g)
        n[:, c] += big_rho * sd_typ * torch.randn(n.shape[0], generator=g)
        d = p.mean(0) - n.mean(0)
        noise_only.append({"channel": c, "share": float(d[c] ** 2 / (d ** 2).sum().clamp_min(1e-30)),
                           "auroc_heldout": heldout_projection_auroc(p, n),
                           "sign": int(torch.sign(d[c]))})
    shares = torch.tensor([r["share"] for r in noise_only])
    return {
        "base": {"top_share": top_share(d_raw), "auroc_heldout": base_auroc},
        "robustness": robustness,
        "noise_only": {"big_rho": big_rho, "n_rep": n_rep, "rows": noise_only,
                       "share_mean": float(shares.mean()), "share_max": float(shares.max()),
                       "auroc_mean": float(torch.tensor([r["auroc_heldout"] for r in noise_only]).mean()),
                       "n_positive_sign": int(sum(r["sign"] > 0 for r in noise_only))},
    }


# ---- closing the loop: where the evidence lives, and what a probe uses -------


def evidence_profile(xp: Tensor, xn: Tensor, channel: int, z_words_p: Tensor, z_words_n: Tensor,
                     *, top_ks: tuple[int, ...] = (1, 10, 100, 300, 1000), k_ev: int = 100) -> dict:
    """How diffuse the class evidence is, whether the coordinates that carry it
    are ordinary, where c* ranks among them, and how much of the raw vector's
    energy sits on the evidence at all.

    Evidence = d'^2 per coordinate (pooled within-class sd), E = sum d'^2.
    Reported: share of E in the top-k coordinates by d'^2; the best single
    coordinate (index, d', its held-out single-coordinate AUROC); rho and
    within-class |r(x, words)| of the top-k_ev evidence coordinates; c*'s rank
    by d'^2 and evidence share; the raw vector's energy share on the evidence
    top-k_ev, the overlap of its own top-k_ev (by energy) with them, and the
    signed cos(d_hat, e_c*).
    """
    xp, xn = xp.float(), xn.float()
    D = xp.shape[1]
    delta = xp.mean(0) - xn.mean(0)
    sd = torch.sqrt(0.5 * (xp.var(0, unbiased=True) + xn.var(0, unbiased=True))).clamp_min(1e-12)
    dprime = delta / sd
    ev = dprime ** 2
    E = float(ev.sum())
    order = torch.argsort(ev, descending=True)
    cum = {str(k): float(ev[order[:k]].sum() / max(E, 1e-30)) for k in top_ks if k <= D}
    best = int(order[0])
    ep, op = _even_odd(xp.shape[0])
    en, on = _even_odd(xn.shape[0])
    sgn = 1.0 if float(xp[ep, best].mean() - xn[en, best].mean()) >= 0 else -1.0
    best_auroc = _auroc(sgn * xp[op, best], sgn * xn[on, best])
    rho = sd / sd.median().clamp_min(1e-12)
    top_ev = order[: min(k_ev, D)]
    # within-class |r| with words for the evidence coordinates
    cp = xp[:, top_ev] - xp[:, top_ev].mean(0, keepdim=True)
    cn = xn[:, top_ev] - xn[:, top_ev].mean(0, keepdim=True)
    zp = (z_words_p.float() - z_words_p.float().mean())[:, None]
    zn = (z_words_n.float() - z_words_n.float().mean())[:, None]
    r = ((cp * zp).sum(0) + (cn * zn).sum(0)) / (
        torch.sqrt((cp ** 2).sum(0) + (cn ** 2).sum(0)) *
        torch.sqrt((zp ** 2).sum() + (zn ** 2).sum())).clamp_min(1e-12)
    rank_c = int((ev > ev[channel]).sum()) + 1
    energy = delta ** 2
    top_raw = torch.argsort(energy, descending=True)[: min(k_ev, D)]
    overlap = len(set(top_raw.tolist()) & set(top_ev.tolist()))
    return {
        "channel": int(channel), "E": E, "d_model": D,
        "cum_share_top_k": cum,
        "best_channel": {"channel": best, "dprime": float(dprime[best]),
                         "auroc_heldout": best_auroc},
        "evidence_top_k": {"k": int(len(top_ev)), "rho_median": float(rho[top_ev].median()),
                           "rho_max": float(rho[top_ev].max()),
                           "abs_r_words_median": float(r.abs().median())},
        "cstar": {"dprime": float(dprime[channel]), "rank_by_dprime2": rank_c,
                  "evidence_share": float(ev[channel] / max(E, 1e-30)),
                  "rho": float(rho[channel])},
        "raw_vs_evidence": {"energy_share_on_evidence_top_k": float(energy[top_ev].sum() / energy.sum().clamp_min(1e-30)),
                            "overlap_top_k": overlap,
                            "cos_with_e_cstar": float(delta[channel] / delta.norm().clamp_min(1e-12))},
        "dprime_rms_typical": float(torch.sqrt(ev.mean())),
    }


def probe_comparison(xp: Tensor, xn: Tensor, channel: int, *, C: float = 1.0, k_top: int = 100,
                     max_iter: int = 2000) -> dict:
    """Four readouts fit on the EVEN rows of each class, scored on the ODD rows:
    the raw diff-of-means projection, the R2 (d'/sigma) projection, a
    logistic probe on raw features, and a logistic probe on standardised
    features (per-coordinate within-class mean/sd from the training rows).

    Geometry, in EFFECT-PER-SIGMA units (w_c * sigma_c for the raw probe, w_c
    for the standardised one — the coordinate's contribution to the decision
    per one-sd move): c*'s rank and share, the largest per-coordinate share,
    cos with the d' vector, the top-k overlap with the d' top-k; for the raw
    probe also its top-k overlap with the raw estimator's top-k (by energy)
    and cos(raw d, raw-probe w) in raw units. The caveat this exists for: L2
    on raw features makes loud coordinates cheap (a large effect for a tiny
    w), so the raw probe over-uses c* even though its AUROC is immune —
    standardise before interpreting any probe direction.
    """
    from sklearn.linear_model import LogisticRegression

    xp, xn = xp.float(), xn.float()
    ep, op = _even_odd(xp.shape[0])
    en, on = _even_odd(xn.shape[0])
    tr_p, tr_n, te_p, te_n = xp[ep], xn[en], xp[op], xn[on]
    D = xp.shape[1]
    delta = tr_p.mean(0) - tr_n.mean(0)
    sd = torch.sqrt(0.5 * (tr_p.var(0, unbiased=True) + tr_n.var(0, unbiased=True))).clamp_min(1e-12)
    dprime = delta / sd
    mu = 0.5 * (tr_p.mean(0) + tr_n.mean(0))

    def proj_auroc(w: Tensor) -> float:
        u = w / w.norm().clamp_min(1e-12)
        return _auroc(te_p @ u, te_n @ u)

    X_tr = torch.cat([tr_p, tr_n]).numpy()
    y_tr = torch.cat([torch.ones(len(tr_p)), torch.zeros(len(tr_n))]).numpy()
    Xs_tr = ((torch.cat([tr_p, tr_n]) - mu) / sd).numpy()
    raw = LogisticRegression(C=C, solver="newton-cg", max_iter=max_iter).fit(X_tr, y_tr)
    std = LogisticRegression(C=C, solver="newton-cg", max_iter=max_iter).fit(Xs_tr, y_tr)
    w_raw = torch.tensor(raw.coef_[0], dtype=torch.float32)
    w_std = torch.tensor(std.coef_[0], dtype=torch.float32)
    te_raw = _auroc(torch.tensor(raw.decision_function(te_p.numpy())),
                    torch.tensor(raw.decision_function(te_n.numpy())))
    te_std = _auroc(torch.tensor(std.decision_function(((te_p - mu) / sd).numpy())),
                    torch.tensor(std.decision_function(((te_n - mu) / sd).numpy())))

    top_dp = set(torch.argsort(dprime.abs(), descending=True)[:k_top].tolist())
    top_raw_est = set(torch.argsort(delta.abs(), descending=True)[:k_top].tolist())

    def geometry(effect: Tensor) -> dict:
        e2 = effect ** 2
        share = e2 / e2.sum().clamp_min(1e-30)
        top = set(torch.argsort(e2, descending=True)[:k_top].tolist())
        return {"cstar_rank": int((e2 > e2[channel]).sum()) + 1,
                "cstar_share": float(share[channel]),
                "max_share": float(share.max()),
                "cos_with_dprime": float(torch.nn.functional.cosine_similarity(effect, dprime, dim=0)),
                "overlap_top_k_with_dprime": len(top & top_dp),
                "overlap_top_k_with_raw_estimator": len(top & top_raw_est)}

    return {
        "channel": int(channel), "C": C, "k_top": k_top, "n_train": int(len(tr_p) + len(tr_n)),
        "auroc_heldout": {"raw_diff_of_means": proj_auroc(delta),
                          "r2_standardized": proj_auroc(delta / sd ** 2),
                          "raw_feature_probe": te_raw, "standardized_probe": te_std},
        "raw_probe": {**geometry(w_raw * sd),                    # effect per sigma
                      "cos_raw_d_with_raw_w": float(torch.nn.functional.cosine_similarity(delta, w_raw, dim=0)),
                      "w2_share_cstar_raw_units": float(w_raw[channel] ** 2 / (w_raw ** 2).sum().clamp_min(1e-30))},
        "standardized_probe": geometry(w_std),
        "d_model": D,
    }
