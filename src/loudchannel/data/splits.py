"""Frozen splits (: 'exact splits frozen and published').

freeze_split() samples deterministically, writes data/splits/{name}.json with a
content hash; load_split() refuses to run against a modified file. The JSON
files are committed to the repo — reviewers rerun against byte-identical data.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..config import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _hash(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def freeze_split(name: str, rows: list[dict], *, n: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = list(rows)
    if n is not None and n < len(rows):
        rows = rng.sample(rows, n)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "seed": seed, "n": len(rows), "sha256_16": _hash(rows), "rows": rows}
    (SPLITS_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False)
    )
    return rows


def load_split(name: str) -> list[dict]:
    p = SPLITS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen split '{name}' missing — run scripts/freeze_splits.py first"
        )
    payload = json.loads(p.read_text())
    if _hash(payload["rows"]) != payload["sha256_16"]:
        raise ValueError(f"Frozen split '{name}' has been modified (hash mismatch)")
    return payload["rows"]


SUBSETS = ("stratified", "head")


def interleave_by_source(rows: list[dict], key: str = "source") -> list[dict]:
    """Round-robin the rows across their `source` values, order preserved within
    each source, so that any head slice `[:n]` is source-balanced.

    Why this exists: the frozen harmful splits are stored as 150 AdvBench rows
    followed by 150 SORRY-Bench rows (freeze_split samples per source and
    concatenates; nothing shuffles on load). A plain `rows[:128]` is therefore
    AdvBench-only and `rows[:150]` never sees SORRY-Bench — which silently
    changes the prompt-length gap the recipe pipeline is meant to control for
    (12.3 vs 10.2 words on the head slice, 17.3 vs 10.3 on the full split) and
    makes the evaluation slice incomparable to Zhao et al.'s Table 1. The frozen
    files are NOT touched (load_split still hash-checks them); this is a pure
    reordering of the list they return. Single-source splits (Alpaca) come
    back unchanged, so the harmless side is identical either way.

    Deterministic, seed-free: row i of source s keeps its rank among source-s
    rows, and sources are visited in first-appearance order.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "")), []).append(r)
    if len(groups) <= 1:
        return list(rows)
    out: list[dict] = []
    queues = [list(g) for g in groups.values()]
    i = 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    """Apply a named ordering before head-slicing (see SUBSETS)."""
    if subset == "stratified":
        return interleave_by_source(rows)
    if subset == "head":
        return list(rows)
    raise ValueError(f"unknown subset {subset!r}; known: {SUBSETS}")


def source_counts(rows: list[dict], key: str = "source") -> dict[str, int]:
    """{source: n} for a slice — printed by the recipe stages so the composition
    of every train/val/test set is in the log next to the numbers."""
    out: dict[str, int] = {}
    for r in rows:
        s = str(r.get(key, ""))
        out[s] = out.get(s, 0) + 1
    return out
