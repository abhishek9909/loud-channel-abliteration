"""Freeze every prompt split this study runs on. Run ONCE.

The JSON outputs in `data/splits/` are committed and hash-checked on load, so
they are what actually gets used — this script exists so the sampling is
auditable and re-derivable, not because it needs to be run again.

    python scripts/freeze_splits.py             # the core contrast + controls
    python scripts/freeze_splits.py --replication   # + the dataset-swap pairs

The core contrast is AdvBench + SORRY-Bench against Alpaca, split into
disjoint extraction and evaluation halves. The controls are an MMLU slice
(capability) and SST-2 (a binary contrast with no safety content, used to show
the failure reproduces with the class signal set to zero).

`--replication` freezes the swapped-source pairs that
`loudchannel.data.datasets` exposes as `strongreject_wjb`, `wjb_orbench` and
`orbench_alpaca`. Harmful sides are deduplicated (token-set Jaccard) against
AdvBench + SORRY-Bench so a "different dataset" really is different. The
WildJailbreak loader needs the gated HF dataset `allenai/wildjailbreak`
(license accepted and HF_TOKEN set).
"""

from __future__ import annotations

import argparse
import random

from loudchannel.config import load_experiment_config
from loudchannel.data.dedup import dedup_against
from loudchannel.data.registry import load_dataset_by_name
from loudchannel.data.splits import freeze_split


def freeze_core(cfg: dict) -> None:
    seed = cfg["seed"]
    n = cfg["n_per_condition"]
    rng = random.Random(seed)

    # --- harmful: AdvBench + SORRY-Bench, disjoint extract/eval halves -------
    adv = load_dataset_by_name("advbench")
    sorry = load_dataset_by_name("sorrybench")
    rng.shuffle(adv)
    rng.shuffle(sorry)
    half = n // 2
    freeze_split("harmful_extract", adv[:half] + sorry[:half], n=None, seed=seed)
    freeze_split("harmful_eval", adv[half : 2 * half] + sorry[half : 2 * half], n=None, seed=seed)

    # --- harmless: Alpaca, disjoint halves -----------------------------------
    alp = load_dataset_by_name("alpaca")
    rng.shuffle(alp)
    freeze_split("harmless_extract", alp[:n], n=None, seed=seed)
    freeze_split("harmless_eval", alp[n : 2 * n], n=None, seed=seed)

    # --- capability control ---------------------------------------------------
    mmlu = load_dataset_by_name("mmlu")
    freeze_split("mmlu_slice", mmlu, n=cfg["mmlu_slice"], seed=seed)

    # --- unrelated binary contrast (sentiment) --------------------------------
    sst = load_dataset_by_name("sst2")
    pos = [r for r in sst if r["label"] == "pos"]
    neg = [r for r in sst if r["label"] == "neg"]
    freeze_split("sst2_pos", pos, n=n, seed=seed)
    freeze_split("sst2_neg", neg, n=n, seed=seed)

    print("Core splits frozen to data/splits/.")


def freeze_replication(cfg: dict) -> None:
    """The dataset-swap pairs: same pipeline, prompt sets with a different
    covariate structure (in one of them the length gap points the other way)."""
    seed = cfg["seed"]
    n = cfg["n_per_condition"]
    thr = cfg["dedup_threshold"]
    rng = random.Random(seed)

    ref = load_dataset_by_name("advbench") + load_dataset_by_name("sorrybench")

    sr = load_dataset_by_name("strongreject")
    freeze_split("strongreject_eval", sr, n=min(n, len(sr)), seed=seed)

    # OR-Bench: toxic (harmful) vs hard-1k (seemingly-toxic-but-benign)
    ob = load_dataset_by_name("orbench")
    toxic = [r for r in ob if r["label"] == "harmful"]
    hard = [r for r in ob if r["label"] == "hard_benign"]
    kept, dropped = dedup_against(toxic, ref, threshold=thr)
    print(f"OR-Bench toxic dedup vs AdvBench+SORRY: kept {len(kept)}, dropped {len(dropped)}")
    freeze_split("orbench_toxic", kept, n=min(n, len(kept)), seed=seed)
    freeze_split("orbench_hard", hard, n=min(n, len(hard)), seed=seed)

    # WildJailbreak vanilla: harmful + matched benign look-alikes (gated on HF).
    # ~50k/class -> pre-subsample 3n before the O(cand x ref) dedup pass.
    wj = load_dataset_by_name("wildjailbreak")
    wj_h = [r for r in wj if r["label"] == "harmful"]
    wj_b = [r for r in wj if r["label"] == "harmless"]
    wj_h = rng.sample(wj_h, min(len(wj_h), 3 * n))
    kept, dropped = dedup_against(wj_h, ref, threshold=thr)
    print(f"WJB vanilla-harmful dedup vs AdvBench+SORRY: kept {len(kept)}, "
          f"dropped {len(dropped)} (of {len(wj_h)} pre-sampled)")
    freeze_split("wjb_harmful", kept, n=min(n, len(kept)), seed=seed)
    freeze_split("wjb_benign", wj_b, n=min(n, len(wj_b)), seed=seed)

    print("Replication splits frozen to data/splits/.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replication", action="store_true",
                    help="also freeze the dataset-swap pairs (StrongReject, "
                         "WildJailbreak, OR-Bench)")
    ap.add_argument("--only-replication", action="store_true",
                    help="freeze ONLY the dataset-swap pairs; the core splits "
                         "are left untouched")
    args = ap.parse_args()

    cfg = load_experiment_config()
    if not args.only_replication:
        freeze_core(cfg)
    if args.replication or args.only_replication:
        freeze_replication(cfg)


if __name__ == "__main__":
    main()
