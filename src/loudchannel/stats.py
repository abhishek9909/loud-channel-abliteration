"""Statistics required by: Cohen's d, bootstrap 95% CIs (>=1000
resamples), McNemar on matched clean-vs-ablated items, Holm-Bonferroni across
the position x direction x layer grid, Cohen's kappa for CoT coding."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [
        statistic(values[rng.integers(0, len(values), len(values))])
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def mcnemar_exact(clean: np.ndarray, treated: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes (same items, two conditions)."""
    clean = np.asarray(clean, dtype=bool)
    treated = np.asarray(treated, dtype=bool)
    b = int((clean & ~treated).sum())   # correct -> incorrect
    c = int((~clean & treated).sum())   # incorrect -> correct
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))
    return {"b_clean_only": b, "c_treated_only": c, "n_discordant": n, "p": p}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down correction. Returns {key: {p, p_adj, reject}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max, rejecting = {}, 0.0, True
    for rank, (k, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, p_adj)  # enforce monotonicity
        if p > alpha / (m - rank):
            rejecting = False
        out[k] = {"p": p, "p_adj": running_max, "reject": rejecting}
    return out


def cohens_kappa(r1: np.ndarray, r2: np.ndarray) -> float:
    """Inter-annotator agreement for binary codes (H4 secondary readout)."""
    r1 = np.asarray(r1, dtype=int)
    r2 = np.asarray(r2, dtype=int)
    po = float((r1 == r2).mean())
    cats = np.unique(np.concatenate([r1, r2]))
    pe = float(sum((r1 == c).mean() * (r2 == c).mean() for c in cats))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def paired_flip_rate(clean: np.ndarray, treated: np.ndarray) -> float:
    """Fraction of items whose binary outcome changed under intervention."""
    return float((np.asarray(clean, bool) != np.asarray(treated, bool)).mean())


def tost_equivalence(a, b, margin, *, paired: bool = True,
                     alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for statistical EQUIVALENCE of two samples
    within +/- `margin` on the mean difference (a - b).

    This is the primitive jailbreak sub-claim B depends on (see
    ). Equivalence is the ALTERNATIVE hypothesis:
    it is licensed only when BOTH one-sided tests reject, i.e.
    max(p_low, p_high) < alpha. A plain non-significant difference test does
    NOT establish equivalence — an underpowered study fails to reject in both
    directions, and this function correctly reports it as non-equivalent (wide
    CI, large p). "We failed to find a difference" is not "there is none."

    paired=True  matched items; uses the per-item difference (requires equal n).
    paired=False Welch two-sample (unequal n / variance).

    Returns {mean_diff, margin, se, df, t_low, t_high, p_low, p_high, p,
    equivalent, ci_low, ci_hi}. The CI is the (1 - 2*alpha) interval, so
    `equivalent` holds iff that CI lies inside [-margin, +margin].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    margin = abs(float(margin))

    if paired:
        assert len(a) == len(b), "paired TOST needs equal-length samples"
        d = a - b
        n = len(d)
        assert n > 1, "TOST needs at least 2 paired items"
        mean_diff = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n))
        df = float(n - 1)
    else:
        na, nb = len(a), len(b)
        assert na > 1 and nb > 1, "TOST needs at least 2 items per group"
        mean_diff = float(a.mean() - b.mean())
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        se = float(np.sqrt(va / na + vb / nb))
        # Welch-Satterthwaite degrees of freedom
        df = float((va / na + vb / nb) ** 2 /
                   ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)))

    if se == 0.0:
        # Degenerate (no residual variance): equivalent iff the point estimate
        # itself sits strictly inside the margin.
        equivalent = abs(mean_diff) < margin
        p = 0.0 if equivalent else 1.0
        return {"mean_diff": mean_diff, "margin": margin, "se": 0.0, "df": df,
                "t_low": float("inf"), "t_high": float("-inf"),
                "p_low": p, "p_high": p, "p": p, "equivalent": equivalent,
                "ci_low": mean_diff, "ci_hi": mean_diff}

    t_low = (mean_diff + margin) / se       # H0_low:  diff <= -margin
    t_high = (mean_diff - margin) / se      # H0_high: diff >= +margin
    p_low = float(sps.t.sf(t_low, df))      # reject if diff > -margin
    p_high = float(sps.t.cdf(t_high, df))   # reject if diff <  +margin
    p = max(p_low, p_high)
    equivalent = p < alpha

    tcrit = float(sps.t.ppf(1 - alpha, df))  # (1 - 2*alpha) two-sided CI
    return {"mean_diff": mean_diff, "margin": margin, "se": se, "df": df,
            "t_low": float(t_low), "t_high": float(t_high),
            "p_low": p_low, "p_high": p_high, "p": p, "equivalent": equivalent,
            "ci_low": mean_diff - tcrit * se, "ci_hi": mean_diff + tcrit * se}
