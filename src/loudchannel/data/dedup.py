"""Deduplicate a replication prompt set against the set the estimator was
developed on, so that a "different dataset" really is different. Token-set
Jaccard on normalized text; threshold from configs/experiment.yaml
(dedup_threshold)."""

from __future__ import annotations

import re


def _norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedup_against(
    candidates: list[dict],
    reference: list[dict],
    *,
    key: str = "instruction",
    threshold: float = 0.6,
) -> tuple[list[dict], list[dict]]:
    """(kept, dropped). Drops candidates with Jaccard >= threshold vs any reference row."""
    ref_tokens = [_norm_tokens(r[key]) for r in reference]
    kept, dropped = [], []
    for c in candidates:
        tc = _norm_tokens(c[key])
        dup = any(
            tc and tr and len(tc & tr) / len(tc | tr) >= threshold for tr in ref_tokens
        )
        (dropped if dup else kept).append(c)
    return kept, dropped
