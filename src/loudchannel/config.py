"""YAML config loading for models and experiment defaults."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclasses.dataclass
class ModelConfig:
    name: str
    hf_id: str
    role: str = "generalization"
    model_class: str = "language"   # "language" | "vision" (multimodal backbones, text-only use)
    layers_path: str = "model.layers"
    lm_head_path: str = "lm_head"
    n_layers: int | None = None
    d_model: int | None = None
    dtype: str = "bfloat16"
    chat_template_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    layer_band: list[int] | None = None  # frozen after anchor layer sweep


def load_model_config(name: str) -> ModelConfig:
    """`name` is either a config stem under configs/models/ or a path to a yaml."""
    p = Path(name)
    if not p.exists():
        p = CONFIG_DIR / "models" / f"{name}.yaml"
    if not p.exists():
        available = sorted(f.stem for f in (CONFIG_DIR / "models").glob("*.yaml"))
        raise FileNotFoundError(f"No model config '{name}'. Available: {available}")
    raw = yaml.safe_load(p.read_text())
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in fields})


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_DIR / "experiment.yaml"
    return yaml.safe_load(p.read_text())


def artifacts_dir(exp_cfg: dict, model_name: str, sub: str) -> Path:
    d = REPO_ROOT / exp_cfg.get("artifacts_dir", "artifacts") / model_name / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
