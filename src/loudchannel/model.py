"""NNsight model wrapper.

Wraps `nnsight.LanguageModel` with:
  - config-driven access to the decoder layer list (handles nested paths for
    encoder-free multimodal backbones, e.g. Gemma-4) and lm_head;
  - chat-template rendering that returns both the templated string and the
    character span of the raw instruction inside it (consumed by positions.py —
    positions are always read from the tokenizer, never hard-coded);
  - a `remote` flag threaded through every trace for NDIF execution.
"""

from __future__ import annotations

import os
from functools import partial, reduce
from typing import Any

import torch

from .config import ModelConfig, load_model_config

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _resolve(obj: Any, dotted: str) -> Any:
    return reduce(getattr, dotted.split("."), obj)


class HarmModel:
    def __init__(self, cfg: ModelConfig | str, *, remote: bool = False, device_map: str = "auto"):
        if isinstance(cfg, str):
            cfg = load_model_config(cfg)
        self.cfg = cfg
        self.remote = remote
        self._layers = None
        self._tuple_out: bool | None = None

        # Deferred imports: keeps dataset-only code GPU-free.
        if cfg.model_class == "vision":
            # Multimodal-registered backbones (Qwen3.5, Gemma-4) refuse
            # LanguageModel; VisionLanguageModel extends it with an
            # AutoProcessor. We use it strictly text-only.
            from nnsight import VisionLanguageModel as _Cls
        else:
            from nnsight import LanguageModel as _Cls

        kwargs: dict[str, Any] = {"dtype": DTYPES[cfg.dtype]}
        if remote:
            # NDIF executes server-side; local weights not needed.
            kwargs["dispatch"] = False
        else:
            kwargs["device_map"] = device_map
        self.lm = _Cls(cfg.hf_id, **kwargs)
        tok = getattr(self.lm, "tokenizer", None)
        if tok is None:  # VLM exposes a processor wrapping the tokenizer
            tok = self.lm.processor.tokenizer
        self.tokenizer = tok
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # positions.py indexes from the left

        # Double-BOS guard (OPT-IN via LOUDCHANNEL_FIX_BOS=1). render() emits the
        # chat template with BOS as text; positions.py and nnsight then
        # re-tokenize with add_special_tokens=True, so tokenizers that prepend
        # BOS (llama3, gemma3 — verified 2026-07-08; qwen has none) fed the
        # model <bos><bos>. When enabled, render() strips the leading BOS text
        # so re-tokenization adds exactly one. Off by default: legacy artifacts
        # and any in-flight jobs keep the old (internally consistent) behavior.
        self._fix_bos = os.environ.get("LOUDCHANNEL_FIX_BOS") == "1"
        self._tok_adds_bos = False
        if self.tokenizer.bos_token_id is not None:
            probe_ids = self.tokenizer("x", add_special_tokens=True)["input_ids"]
            self._tok_adds_bos = probe_ids[:1] == [self.tokenizer.bos_token_id]

    # ---- structure ---------------------------------------------------------

    # Common locations of the decoder-layer list across text-only and
    # multimodal wrappers; tried in order if the configured path fails.
    LAYER_PATH_CANDIDATES = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "model.text_model.layers",
        "model.model.layers",
    )

    @property
    def layers(self):
        if self._layers is None:
            tried = []
            for path in (self.cfg.layers_path, *self.LAYER_PATH_CANDIDATES):
                if path in tried:
                    continue
                tried.append(path)
                try:
                    cand = _resolve(self.lm, path)
                    len(cand)  # must behave like a ModuleList
                except (AttributeError, TypeError):
                    continue
                if path != self.cfg.layers_path:
                    print(f"[loudchannel] layers_path '{self.cfg.layers_path}' failed; "
                          f"using '{path}' — freeze this into configs/models/{self.cfg.name}.yaml")
                self._layers = cand
                break
            else:
                raise AttributeError(
                    f"Could not locate decoder layers on {self.cfg.hf_id}; tried {tried}. "
                    "Inspect `print(model.lm)` and set layers_path in the model yaml."
                )
        return self._layers

    LM_HEAD_CANDIDATES = ("lm_head", "language_model.lm_head", "model.lm_head")

    @property
    def lm_head(self):
        for path in (self.cfg.lm_head_path, *self.LM_HEAD_CANDIDATES):
            try:
                return _resolve(self.lm, path)
            except AttributeError:
                continue
        raise AttributeError(
            f"Could not locate lm_head on {self.cfg.hf_id}; set lm_head_path in the yaml."
        )

    @property
    def n_layers(self) -> int:
        if self.cfg.n_layers:
            return self.cfg.n_layers
        return len(self.layers)

    def layer_band(self) -> list[int]:
        """Frozen effective band if set in config, else all layers."""
        if self.cfg.layer_band:
            return list(self.cfg.layer_band)
        return list(range(self.n_layers))

    # ---- prompting ---------------------------------------------------------

    def render(self, instruction: str, *, system: str | None = None) -> str:
        """Apply the model's own chat template; return the fully templated string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        templ = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.cfg.chat_template_kwargs,
        )
        if (self._fix_bos and self._tok_adds_bos
                and self.tokenizer.bos_token
                and templ.startswith(self.tokenizer.bos_token)):
            templ = templ[len(self.tokenizer.bos_token):]
        return templ

    def render_no_post(self, instruction: str, *, system: str | None = None) -> str:
        """v2 §3.1 replication: chat template truncated after the instruction
        (the original's 'without post-instruction special tokens' condition)."""
        from .positions import strip_post_instruction

        return strip_post_instruction(
            self.render(instruction, system=system), instruction
        )

    # ---- tracing entry points ----------------------------------------------
    # These MUST be passthroughs, not wrapper methods: nnsight locates the
    # `with` block by inspecting the frame that called .trace()/.generate().
    # A wrapper method interposes a frame with no `with` block ->
    # WithBlockNotFoundError. functools.partial adds no Python frame, so it is
    # safe for injecting remote=True.

    @property
    def trace(self):
        return partial(self.lm.trace, remote=True) if self.remote else self.lm.trace

    @property
    def generate(self):
        return partial(self.lm.generate, remote=True) if self.remote else self.lm.generate

    @property
    def output_is_tuple(self) -> bool:
        """Whether decoder layers return a tuple (hidden, ...) or a bare
        hidden-states tensor — transformers changed this across versions, and
        indexing `[0]` into a bare tensor silently slices the batch dim.
        Probed once with a real 1-token trace; call BEFORE opening any trace
        (traces don't nest)."""
        if self._tuple_out is None:
            first = self.layers[0]
            with self.trace("probe"):
                out = first.output.save()
            self._tuple_out = isinstance(out, tuple)
        return self._tuple_out

    def hidden_of(self, layer_output, *, tuple_out: bool):
        """Normalize a layer's .output proxy to the hidden-states tensor."""
        return layer_output[0] if tuple_out else layer_output
