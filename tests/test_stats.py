import numpy as np
import pytest

from loudchannel.stats import (
    bootstrap_ci,
    cohens_d,
    cohens_kappa,
    holm_bonferroni,
    mcnemar_exact,
    paired_flip_rate,
    tost_equivalence,
)


def test_cohens_d_sign_and_zero():
    a = np.array([2.0, 2.1, 1.9, 2.0])
    b = np.array([1.0, 1.1, 0.9, 1.0])
    assert cohens_d(a, b) > 5
    assert cohens_d(b, a) < -5
    assert cohens_d(a, a) == 0.0


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 500)
    point, lo, hi = bootstrap_ci(x, seed=1)
    assert lo < point < hi
    assert abs(point - 5) < 0.2


def test_mcnemar_symmetric_null():
    clean = np.array([1, 1, 0, 0, 1, 0] * 20, dtype=bool)
    res = mcnemar_exact(clean, clean)
    assert res["n_discordant"] == 0 and res["p"] == 1.0
    treated = clean.copy()
    treated[:30] = ~treated[:30]
    res2 = mcnemar_exact(clean, treated)
    assert res2["n_discordant"] == 30
    assert 0 <= res2["p"] <= 1


def test_holm_monotone_and_bounds():
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    out = holm_bonferroni(p)
    adj = [out[k]["p_adj"] for k in ("a", "b", "c", "d")]
    assert adj == sorted(adj)
    assert all(0 <= x <= 1 for x in adj)
    assert out["a"]["reject"] and not out["d"]["reject"]


def test_kappa_perfect_and_chance():
    r = np.array([0, 1, 0, 1, 1, 0])
    assert cohens_kappa(r, r) == 1.0
    assert abs(cohens_kappa(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))) < 1e-9


def test_flip_rate():
    assert paired_flip_rate([1, 0, 1, 0], [1, 0, 0, 1]) == pytest.approx(0.5)


def test_tost_equivalent_tight_paired():
    # Two near-identical samples with low noise: their mean difference sits well
    # inside the margin AND is tightly estimated -> ruled equivalent.
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 200)
    a = base + rng.normal(0.0, 0.02, 200)
    b = base + rng.normal(0.0, 0.02, 200)
    out = tost_equivalence(a, b, margin=0.5, paired=True)
    assert out["equivalent"]
    assert out["ci_low"] > -0.5 and out["ci_hi"] < 0.5


def test_tost_underpowered_is_not_equivalent():
    # THE load-bearing case: a small, noisy sample
    # whose mean difference is ~0 but poorly estimated. A plain t-test would be
    # non-significant; TOST must NOT call this equivalent — the CI spills past
    # the margin, so we cannot rule out a real difference.
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 5)
    b = rng.normal(0.0, 1.0, 5)
    out = tost_equivalence(a, b, margin=0.2, paired=True)
    assert not out["equivalent"]
    assert out["ci_hi"] > 0.2 or out["ci_low"] < -0.2


def test_tost_real_difference_not_equivalent():
    # A genuine shift larger than the margin is (correctly) not equivalent.
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.3, 100)
    b = rng.normal(0.0, 0.3, 100)
    out = tost_equivalence(a, b, margin=0.3, paired=True)
    assert not out["equivalent"]
    assert out["p_high"] > 0.05  # cannot reject "diff >= +margin"


def test_tost_unpaired_welch_unequal_n():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.5, 120)
    b = rng.normal(0.02, 0.5, 80)
    out = tost_equivalence(a, b, margin=0.5, paired=False)
    assert out["equivalent"]
    assert out["df"] > 0


def test_tost_symmetry_and_margin_sign():
    # margin is applied symmetrically; its sign is irrelevant.
    a = np.array([0.1, 0.2, 0.15, 0.05, 0.12, 0.18])
    b = np.array([0.11, 0.19, 0.14, 0.06, 0.13, 0.17])
    assert (tost_equivalence(a, b, 0.3)["equivalent"]
            == tost_equivalence(a, b, -0.3)["equivalent"])
