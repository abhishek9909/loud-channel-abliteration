"""Linear probes on extracted activations (H1: AUROC >> chance at t_inst)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def projection_auroc(acts_pos, acts_neg, direction) -> dict[int, float]:
    """Per-layer AUROC of the 1-D projection onto a frozen Direction.

    Tests whether the *same* direction transfers (H2d positions, H3 datasets)
    — no probe is fit. acts_*: torch [n, n_layers, d]; returns {layer: auroc}.
    """
    u = direction.unit().float()  # [L, D]
    y = np.concatenate([np.ones(len(acts_pos)), np.zeros(len(acts_neg))])
    out = {}
    for li in range(acts_pos.shape[1]):
        s = np.concatenate([
            acts_pos[:, li, :].numpy() @ u[li].numpy(),
            acts_neg[:, li, :].numpy() @ u[li].numpy(),
        ])
        out[li] = float(roc_auc_score(y, s))
    return out


def projection_stats(acts_pos, acts_neg, direction) -> dict[int, dict]:
    """Per-layer ALIGNMENT of activations with a frozen Direction.

    Beyond the rank statistic (AUROC), reports the projection magnitudes
    themselves — mean h·d̂ per class, their gap, and Cohen's d — so
    "position X has the strongest alignment with d_harm" is quantified in
    the direction's own units, not only by separability.
    acts_*: torch [n, n_layers, d]; returns {layer: {...}}.
    """
    from .stats import cohens_d

    u = direction.unit().float()  # [L, D]
    y = np.concatenate([np.ones(len(acts_pos)), np.zeros(len(acts_neg))])
    out = {}
    for li in range(acts_pos.shape[1]):
        sp = (acts_pos[:, li, :] @ u[li]).numpy()
        sn = (acts_neg[:, li, :] @ u[li]).numpy()
        out[li] = {
            "mean_pos": float(sp.mean()),
            "mean_neg": float(sn.mean()),
            "gap": float(sp.mean() - sn.mean()),
            "cohens_d": cohens_d(sp, sn),
            "auroc": float(roc_auc_score(y, np.concatenate([sp, sn]))),
        }
    return out


def probe_auroc_per_layer(
    acts_pos: np.ndarray,   # [n_pos, n_layers, d]
    acts_neg: np.ndarray,   # [n_neg, n_layers, d]
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[int, dict]:
    """Cross-validated logistic-probe AUROC per layer, with bootstrap 95% CI."""
    n_layers = acts_pos.shape[1]
    X_all = np.concatenate([acts_pos, acts_neg], axis=0)
    y = np.concatenate([np.ones(len(acts_pos)), np.zeros(len(acts_neg))])
    out = {}
    rng = np.random.default_rng(seed)
    for li in range(n_layers):
        X = X_all[:, li, :]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = np.zeros_like(y, dtype=float)
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            scores[te] = clf.decision_function(X[te])
        auroc = roc_auc_score(y, scores)
        # bootstrap CI over items
        boots = []
        idx = np.arange(len(y))
        for _ in range(1000):
            b = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(y[b])) < 2:
                continue
            boots.append(roc_auc_score(y[b], scores[b]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[li] = {"auroc": float(auroc), "ci_lo": float(lo), "ci_hi": float(hi)}
    return out
