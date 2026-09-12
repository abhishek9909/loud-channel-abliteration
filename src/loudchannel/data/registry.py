"""Dataset loaders. Every loader returns list[dict] with at least
{"instruction": str}; extra fields are dataset-specific and preserved into
frozen splits.

Sources:
  advbench       Zou et al. 2023           HF walledai/AdvBench (mirror of
                                           llm-attacks harmful_behaviors.csv)
  sorrybench     Xie et al. 2024           HF sorry-bench/sorry-bench-202406
  alpaca         Taori et al. 2023         HF tatsu-lab/alpaca (no-input rows)
  strongreject   Souly et al. 2024         GitHub CSV (alexandrasouly/strongreject)
  mmlu           Hendrycks et al. 2021     HF cais/mmlu (test) — capability slice
  sst2           Socher et al. 2013        HF stanfordnlp/sst2 — the unrelated
                                           binary contrast (no safety content)
  orbench        Cui et al. ICML 2025      HF bench-llm/or-bench (toxic +
                                           hard-1k: matched-style hard negatives)
  wildjailbreak  Jiang et al. NeurIPS 2024 HF allenai/wildjailbreak — GATED:
                                           accept license on HF + HF_TOKEN.
                                           Vanilla prompts only.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from typing import Callable


def load_dataset(*args, **kwargs):
    """Lazy import so dataset-free code (dedup, pair logic, tests) never
    requires the `datasets` package or network access."""
    from datasets import load_dataset as _ld

    return _ld(*args, **kwargs)


STRONGREJECT_CSV = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)

ADVBENCH_CSV = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


def _fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_advbench() -> list[dict]:
    # Original source (Zou et al. 2023 repo) — the HF mirrors are gated.
    rows = _fetch_csv(ADVBENCH_CSV)
    return [{"instruction": r["goal"], "label": "harmful", "source": "advbench"} for r in rows]


def load_sorrybench() -> list[dict]:
    ds = load_dataset("sorry-bench/sorry-bench-202406", split="train")
    out = []
    for r in ds:
        # keep only the unmutated base prompts
        if r.get("prompt_style", "base") != "base":
            continue
        turns = r.get("turns") or []
        if not turns:
            continue
        out.append(
            {"instruction": turns[0], "label": "harmful", "source": "sorrybench",
             "category": r.get("category", "")}
        )
    return out


def load_alpaca() -> list[dict]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    return [
        {"instruction": r["instruction"], "label": "harmless", "source": "alpaca"}
        for r in ds
        if not r.get("input")
    ]


def load_strongreject() -> list[dict]:
    rows = _fetch_csv(STRONGREJECT_CSV)
    return [
        {"instruction": r["forbidden_prompt"], "label": "harmful",
         "source": "strongreject", "category": r.get("category", "")}
        for r in rows
    ]


def load_mmlu() -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for r in ds:
        out.append(
            {
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": int(r["answer"]),
                "subject": r["subject"],
                "source": "mmlu",
            }
        )
    return out


def load_sst2() -> list[dict]:
    ds = load_dataset("stanfordnlp/sst2", split="train")
    return [
        {"instruction": r["sentence"].strip(), "label": "pos" if r["label"] == 1 else "neg",
         "source": "sst2"}
        for r in ds
    ]


def load_orbench() -> list[dict]:
    """OR-Bench (Cui et al., ICML 2025). Two subsets from one generation
    pipeline, so surface form/style is matched across classes:
      or-bench-toxic    (~600)  genuinely toxic prompts    -> label 'harmful'
      or-bench-hard-1k  (~1k)   seemingly-toxic-but-benign -> label 'hard_benign'
    The Toxic-vs-Hard pairing tests whether d_harm tracks harmfulness rather
    than toxic-sounding surface features (H3 hard-negative arm).
    """
    out = []
    for config, label in [("or-bench-toxic", "harmful"),
                          ("or-bench-hard-1k", "hard_benign")]:
        ds = load_dataset("bench-llm/or-bench", config, split="train")
        for r in ds:
            out.append(
                {"instruction": r["prompt"], "label": label, "source": "orbench",
                 "category": r.get("category", "")}
            )
    return out


def load_wildjailbreak() -> list[dict]:
    """WildJailbreak (Jiang et al., NeurIPS 2024). GATED on HF: visit
    huggingface.co/datasets/allenai/wildjailbreak, accept the AI2 license,
    and set HF_TOKEN before freezing splits.

    Vanilla prompts only — direct harmful requests plus matched benign
    look-alikes ('benign queries that resemble harmful queries in form').
    Adversarial (jailbreak-transformed) rows are excluded: they change the
    position structure of the prompt and are out of scope here.
    TSV-backed dataset: the card mandates delimiter/keep_default_na kwargs.
    """
    ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t",
                      keep_default_na=False, split="train")
    label_map = {"vanilla_harmful": "harmful", "vanilla_benign": "harmless"}
    out = []
    for r in ds:
        label = label_map.get(r.get("data_type", ""))
        if label is None:
            continue
        text = (r.get("vanilla") or "").strip()
        if not text:
            continue
        out.append({"instruction": text, "label": label, "source": "wildjailbreak"})
    return out


LOADERS: dict[str, Callable[[], list[dict]]] = {
    "advbench": load_advbench,
    "sorrybench": load_sorrybench,
    "alpaca": load_alpaca,
    "strongreject": load_strongreject,
    "mmlu": load_mmlu,
    "sst2": load_sst2,
    "orbench": load_orbench,
    "wildjailbreak": load_wildjailbreak,
}


def load_dataset_by_name(name: str) -> list[dict]:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(LOADERS)}")
    return LOADERS[name]()
