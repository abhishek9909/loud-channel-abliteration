"""Named prompt-set pairs for the recipe pipeline — the replication axis.

Everything in the recipe experiment so far ran on ONE contrast built from one
pair of sources: AdvBench + SORRY-Bench against Alpaca. That is enough to show
the estimator fails on Gemma, and not enough to show the finding is about the
model rather than about those prompt sets. The covariate structure in
particular is dataset-specific — the ~8-word length gap and the terminal-
punctuation imbalance are properties of AdvBench-vs-Alpaca, not of refusal — so
a replication on sources with a *different* covariate structure is the cleanest
test that scale, not the specific confound, is what breaks the estimator.

A dataset here is four row lists: harmful/harmless for extraction (which `01_extract`
splits into train + val) and harmful/harmless for evaluation (`03_confirm`). The frozen
default keeps the four hand-frozen splits and is byte-identical to calling
`load_split` directly — `tests/test_datasets.py` pins that. Every other entry
is built from a single frozen file per class, cut into DISJOINT extract and
eval ranges, so no evaluation prompt was ever seen by the estimator.

The dataset name is written into `directions.pt` by `01_extract` and read back by
`02_select`/`03_confirm`/`04_dose`, exactly like `subset`, so a scoring run can never be paired with
directions estimated on different data. Use `--tag` to keep the artifacts of a
replication run separate from the main one.

Note on `--subset`: `stratified` interleaves sources before slicing, which
matters for the default (two harmful sources stored back to back) and is a
no-op for the single-source replication sets. Their source balance is whatever
the frozen file already is.
"""

from __future__ import annotations

import dataclasses

from .splits import load_split

PARTS = ("harmful_extract", "harmless_extract", "harmful_eval", "harmless_eval")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Either four pre-frozen split names, or one file per class cut in two."""

    description: str
    # form A: four frozen splits (the default)
    splits: dict[str, str] | None = None
    # form B: one file per class + disjoint row ranges
    harmful: str | None = None
    harmless: str | None = None
    extract_range: tuple[int, int] | None = None
    eval_range: tuple[int, int] | None = None

    def rows(self, part: str) -> list[dict]:
        assert part in PARTS, part
        if self.splits is not None:
            return load_split(self.splits[part])
        cls, which = part.rsplit("_", 1)
        name = self.harmful if cls == "harmful" else self.harmless
        lo, hi = self.extract_range if which == "extract" else self.eval_range
        rows = load_split(name)[lo:hi]
        assert rows, f"{name}[{lo}:{hi}] is empty — check the ranges"
        return rows

    def provenance(self) -> dict:
        if self.splits is not None:
            return {"form": "frozen_splits", "splits": dict(self.splits),
                    "description": self.description}
        return {"form": "single_file_per_class", "harmful": self.harmful,
                "harmless": self.harmless, "extract_range": list(self.extract_range),
                "eval_range": list(self.eval_range), "description": self.description}


DATASETS: dict[str, DatasetSpec] = {
    # the frozen contrast every published number in this repo was measured on
    "default": DatasetSpec(
        description="AdvBench + SORRY-Bench vs Alpaca (the frozen splits)",
        splits={p: p for p in PARTS}),

    # Replication A — both sides swapped. StrongReject is a different harmful
    # taxonomy (and phrased as questions far more often than AdvBench's
    # imperatives); WildJailbreak's benign half is adversarially-styled benign
    # text, so the length/punctuation imbalance that drives the covariate term
    # on the default pair is not the same imbalance here.
    "strongreject_wjb": DatasetSpec(
        description="StrongReject vs WildJailbreak-benign",
        harmful="strongreject_eval", harmless="wjb_benign",
        extract_range=(0, 160), eval_range=(160, 300)),

    # Replication B — a deliberately harder contrast: WildJailbreak harmful
    # against OR-Bench "hard benign", prompts written to look harmful while
    # being safe. If the account is right this should behave like any other
    # binary contrast; if the channel tracked something about surface harm
    # wording, this is where it would show.
    "wjb_orbench": DatasetSpec(
        description="WildJailbreak-harmful vs OR-Bench hard-benign",
        harmful="wjb_harmful", harmless="orbench_hard",
        extract_range=(0, 160), eval_range=(160, 300)),

    # Replication C — a third harmful taxonomy against the familiar harmless
    # side, isolating a change on the harmful side alone.
    "orbench_alpaca": DatasetSpec(
        description="OR-Bench toxic vs Alpaca (harmful side swapped only)",
        harmful="orbench_toxic", harmless="harmless_extract",
        extract_range=(0, 160), eval_range=(160, 300)),
}

DATASET_NAMES = tuple(DATASETS)


def get(name: str) -> DatasetSpec:
    assert name in DATASETS, f"unknown dataset {name!r}; have {DATASET_NAMES}"
    return DATASETS[name]


def load_rows(name: str, part: str) -> list[dict]:
    """The rows for one part of one dataset."""
    return get(name).rows(part)


def describe(name: str) -> dict:
    return {"dataset": name, **get(name).provenance()}
