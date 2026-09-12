"""Source-stratified head slices of the frozen splits (no torch, no model).

The recipe stages take `rows[:n]` of the frozen harmful splits. Those files are
stored source-by-source (150 AdvBench, then 150 SORRY-Bench), so a plain head
slice is single-source. `interleave_by_source` makes any head slice balanced
without touching the hash-checked files. These tests pin the properties the
stages rely on: determinism, order preservation within a source, balance of
every prefix, the single-source no-op, and — against the real frozen split —
that 23's train / val slices are disjoint and both balanced.
"""

from pathlib import Path

import pytest

from loudchannel.data.splits import SUBSETS, interleave_by_source, source_counts, subset_rows

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def _rows(spec):
    """spec: list of (source, n) -> rows with an index that records the frozen order."""
    out = []
    for src, n in spec:
        out.extend({"source": src, "i": k, "instruction": f"{src}-{k}"} for k in range(n))
    return out


def test_round_robin_balances_every_prefix():
    rows = interleave_by_source(_rows([("a", 4), ("b", 4)]))
    assert [r["source"] for r in rows] == ["a", "b"] * 4
    for n in range(2, 9, 2):
        assert source_counts(rows[:n]) == {"a": n // 2, "b": n // 2}


def test_order_within_source_is_preserved():
    rows = interleave_by_source(_rows([("a", 5), ("b", 3)]))
    for src in ("a", "b"):
        ranks = [r["i"] for r in rows if r["source"] == src]
        assert ranks == sorted(ranks)


def test_unequal_sources_drain_the_shorter_one_then_continue():
    rows = interleave_by_source(_rows([("a", 5), ("b", 2)]))
    assert [r["source"] for r in rows] == ["a", "b", "a", "b", "a", "a", "a"]
    assert len(rows) == 7


def test_single_source_is_identity():
    rows = _rows([("alpaca", 6)])
    assert interleave_by_source(rows) == rows
    assert subset_rows(rows, "stratified") == rows
    assert subset_rows(rows, "head") == rows


def test_is_a_permutation_and_deterministic():
    rows = _rows([("a", 7), ("b", 11), ("c", 3)])
    once, twice = interleave_by_source(rows), interleave_by_source(rows)
    assert once == twice
    assert sorted(r["instruction"] for r in once) == sorted(r["instruction"] for r in rows)


def test_head_subset_is_the_frozen_order():
    rows = _rows([("a", 3), ("b", 3)])
    assert subset_rows(rows, "head") == rows
    with pytest.raises(ValueError):
        subset_rows(rows, "shuffled")
    assert set(SUBSETS) == {"stratified", "head"}


def test_frozen_harmful_extract_train_val_slices_are_balanced_and_disjoint():
    """23 uses rows[:128] for training and rows[128:160] as the Stage-S val
    slice. Under the frozen (head) order that is 128 AdvBench / (22+10); under
    stratified it must be 64+64 / 16+16, with no prompt in both."""
    p = SPLITS / "harmful_extract.json"
    if not p.exists():
        pytest.skip("frozen split not present")
    from loudchannel.data.splits import load_split

    rows = load_split("harmful_extract")
    assert source_counts(rows[:128]) == {"advbench": 128}, "frozen order changed?"
    strat = subset_rows(rows, "stratified")
    assert source_counts(strat[:128]) == {"advbench": 64, "sorrybench": 64}
    assert source_counts(strat[128:160]) == {"advbench": 16, "sorrybench": 16}
    train = {r["instruction"] for r in strat[:128]}
    val = {r["instruction"] for r in strat[128:160]}
    assert not (train & val)
    assert len(strat) == len(rows) and len({r["instruction"] for r in strat}) == len(rows)
