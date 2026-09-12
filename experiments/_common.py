"""Shared CLI plumbing for experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loudchannel.config import artifacts_dir, load_experiment_config
from loudchannel.model import HarmModel


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="llama3-8b", help="config stem in configs/models/")
    ap.add_argument("--remote", action="store_true", help="execute on NDIF")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per split (smoke/debug runs)")
    return ap


def setup(args) -> tuple[HarmModel, dict]:
    # Every experiment is inference-only. Without this, nnsight traces retain
    # the autograd graph across all layers -> OOM on 12B models (46GB+).
    import torch

    torch.set_grad_enabled(False)

    exp = load_experiment_config()
    if args.batch_size:
        exp["batch_size"] = args.batch_size
    model = HarmModel(args.model, remote=args.remote)
    return model, exp


def instructions(rows: list[dict], limit: int | None = None) -> list[str]:
    out = [r["instruction"] for r in rows]
    return out[:limit] if limit else out


def dump(exp: dict, model_name: str, sub: str, name: str, obj) -> Path:
    d = artifacts_dir(exp, model_name, sub)
    p = d / name
    p.write_text(json.dumps(obj, indent=1))
    print(f"wrote {p}")
    return p
