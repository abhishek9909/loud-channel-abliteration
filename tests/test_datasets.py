"""The replication datasets, and the guarantee that the default is unchanged.

The recipe pipeline gained a `--dataset` switch so the whole experiment can be
re-run on prompt sets disjoint from the ones it was developed on. The risk in
that change is silent: if `default` stopped resolving to exactly the four frozen
splits, every published number in the repo would quietly be measured on
different rows. The first test pins that. The rest check the property the
replications depend on — extraction and evaluation prompts never overlap.

    pytest tests/test_datasets.py
"""

import pytest

from loudchannel.data import datasets as ds
from loudchannel.data.splits import load_split

PARTS = ds.PARTS


def _available(name: str) -> bool:
    try:
        for p in PARTS:
            ds.load_rows(name, p)
        return True
    except (FileNotFoundError, AssertionError):
        return False


@pytest.mark.parametrize("part", PARTS)
def test_default_is_byte_identical_to_the_frozen_splits(part):
    """`--dataset default` must be the exact rows the repo always used."""
    assert ds.load_rows("default", part) == load_split(part)


@pytest.mark.parametrize("name", [n for n in ds.DATASET_NAMES if n != "default"])
def test_extract_and_eval_are_disjoint(name):
    if not _available(name):
        pytest.skip(f"{name}: underlying splits not present")
    for cls in ("harmful", "harmless"):
        ex = {r["instruction"] for r in ds.load_rows(name, f"{cls}_extract")}
        ev = {r["instruction"] for r in ds.load_rows(name, f"{cls}_eval")}
        assert ex and ev
        assert not (ex & ev), f"{name}/{cls}: {len(ex & ev)} prompts in both halves"


@pytest.mark.parametrize("name", [n for n in ds.DATASET_NAMES if n != "default"])
def test_extract_half_is_big_enough_for_the_arditi_budget(name):
    """23 needs n_train 128 + n_val 32 per class."""
    if not _available(name):
        pytest.skip(f"{name}: underlying splits not present")
    for cls in ("harmful", "harmless"):
        n = len(ds.load_rows(name, f"{cls}_extract"))
        assert n >= 160, f"{name}/{cls}: only {n} extraction rows, need 160"


@pytest.mark.parametrize("name", ds.DATASET_NAMES)
def test_classes_are_not_accidentally_the_same_rows(name):
    if not _available(name):
        pytest.skip(f"{name}: underlying splits not present")
    a = {r["instruction"] for r in ds.load_rows(name, "harmful_extract")}
    b = {r["instruction"] for r in ds.load_rows(name, "harmless_extract")}
    assert not (a & b), f"{name}: harmful and harmless share prompts"


@pytest.mark.parametrize("name", ds.DATASET_NAMES)
def test_describe_is_json_safe(name):
    import json
    json.dumps(ds.describe(name))


def test_replication_sources_differ_from_the_default():
    """The point of a replication is different sources on BOTH sides."""
    if not (_available("strongreject_wjb") and _available("default")):
        pytest.skip("splits not present")
    def srcs(name, part):
        return {r.get("source") for r in ds.load_rows(name, part)}
    for part in ("harmful_extract", "harmless_extract"):
        assert not (srcs("default", part) & srcs("strongreject_wjb", part)), part
