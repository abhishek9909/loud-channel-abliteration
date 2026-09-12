"""D7 — weight-level provenance of the nuisance coordinates (RMSNorm gains).

Why this exists
---------------
Stage D identifies "massive" residual coordinates from activation statistics
alone (class-blind |mean| on the harmless split). The audit of that finding
(`massive-channel-measurement-audit`, 2026-09-10) ruled out every measurement
artifact it could reach offline — hook point, padding/indexing, tuple-vs-bare
output, bf16, double BOS — and left ONE check that needs the weights: is the
coordinate produced by the architecture, i.e. by a per-channel RMSNorm gain?

Gemma-2/3 decoder block (transformers modeling_gemma3.py):

    x = x + post_attention_layernorm( attn( input_layernorm(x) ) )
    x = x + post_feedforward_layernorm( mlp( pre_feedforward_layernorm(x) ) )
    RMSNorm(v) = v / rms(v) * (1 + w)              # gain is (1 + w), not w

A post-block ("writer") norm emits unit-RMS vectors times a per-channel gain,
so a channel with a huge fixed-sign writer gain receives an O(gain) increment
at every layer at every token — exactly the smooth, monotone, all-prompt,
fixed-index growth the activation cache shows. The pre-block ("reader") gains
say whether downstream blocks read the channel as content (gain ~ 1) or hide
it (gain ~ 0), leaving it to act only through the RMS denominator — which is
what decides whether the refusal direction can be orthogonal to it by
construction (README.md).

Llama/Qwen blocks have only reader norms (input_layernorm and, despite its
name, post_attention_layernorm, which is applied BEFORE the MLP) and add the
block outputs to the residual un-normalised, so there is no writer gain to
find. On those models this report is the control: it should single out
nothing.

Everything is read from the loaded model's parameters (a few thousand floats
per norm); no forward pass, no extra download. `scripts/check_gemma_norm_gains.py`
computes the same numbers from the safetensors shards without a model load,
for the offline case.
"""

from __future__ import annotations

import inspect
from functools import reduce
from typing import Any

import torch
from torch import Tensor

LAYER_PATH_CANDIDATES = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "model.model.layers",
)


# ---- pure tensor maths (unit-tested without a model) ------------------------


def classify_norms(names: list[str]) -> tuple[list[str], list[str]]:
    """(readers, writers) among a decoder layer's d_model-sized norms.

    Four norms (Gemma-2/3): the two `post_*` are applied to block OUTPUTS
    before the residual add — writers. Two norms (Llama/Qwen/Mistral):
    both are applied to the residual before a block — readers — even though
    one is called `post_attention_layernorm`. Name-based rules would get the
    Llama case wrong, which is why the count decides.
    """
    if len(names) >= 4:
        writers = [n for n in names if n.startswith("post_")]
        readers = [n for n in names if n not in writers]
        return readers, writers
    return list(names), []


def gain_offset(mod: Any) -> float:
    """1.0 when the module's forward multiplies by (1 + weight) (Gemma
    convention: weights initialised at zero), else 0.0 (Llama: weight IS the
    gain). Read from the source so a renamed class cannot fool it; class-name
    fallback when the source is unavailable (e.g. compiled)."""
    try:
        src = inspect.getsource(type(mod).forward)
        if "1.0 + self.weight" in src or "1 + self.weight" in src:
            return 1.0
        if "self.weight" in src:
            return 0.0
    except (OSError, TypeError):
        pass
    return 1.0 if type(mod).__name__.lower().startswith("gemma") else 0.0


def channel_stats(G: Tensor, channels: list[int]) -> dict:
    """Per-channel statistics of a gain stack G [L, D].

    For each channel: gain per layer, |g| rank per layer (1 = largest of D),
    mean/min/max, sign consistency over the layers where the channel is not
    negligible, and the share of layers where it ranks first / in the top 8.
    Also the layer-wise median and max |g| (context for "how large is large").
    """
    G = G.float()
    ag = G.abs()
    med = ag.median(dim=-1).values                        # [L]
    mx, argmx = ag.max(dim=-1)                            # [L]
    out: dict[str, Any] = {
        "median_abs_per_layer": med.tolist(),
        "max_abs_per_layer": mx.tolist(),
        "argmax_per_layer": argmx.tolist(),
        "at": {},
    }
    for c in channels:
        g = G[:, c]
        rank = (ag > ag[:, c : c + 1]).sum(-1) + 1        # [L]
        big = ag[:, c] > med                              # where it is not noise
        signs = torch.sign(g[big])
        out["at"][str(c)] = {
            "gain": g.tolist(),
            "rank": rank.tolist(),
            "mean": float(g.mean()), "min": float(g.min()), "max": float(g.max()),
            "abs_over_median": (ag[:, c] / med.clamp_min(1e-12)).tolist(),
            "sign_consistent": bool(len(signs) == 0 or (signs == signs[0]).all()),
            "frac_rank1": float((rank == 1).float().mean()),
            "frac_rank_le8": float((rank <= 8).float().mean()),
        }
    return out


def top_channels(G: Tensor, k: int = 8) -> dict:
    """Which channels do the gains single out model-wide? mean |g| over layers."""
    score = G.float().abs().mean(0)                        # [D]
    top = torch.argsort(score, descending=True)[:k]
    return {
        "channels": [int(c) for c in top],
        "mean_abs_gain": [float(score[c]) for c in top],
        "median_channel": float(score.median()),
    }


def verdict(report: dict, channels: list[int]) -> dict:
    """The audit's predictions, as booleans per channel, from the numbers
    already in `report`. None where the architecture has no such norm."""
    out: dict[str, dict] = {}
    writers, readers = report["writer_norms"], report["reader_norms"]
    for c in channels:
        k = str(c)
        v: dict[str, Any] = {}
        if writers:
            w = report["per_norm"][writers[-1]]["at"][k]  # the post-FFN norm
            v["writer_gain_dominant"] = w["frac_rank1"] >= 0.5
            v["writer_sign_consistent"] = w["sign_consistent"]
            v["writer_frac_rank1"] = w["frac_rank1"]
            if len(writers) >= 2:
                # Do class-carrying writes and sink-generating writes pass
                # through the SAME gain at c*? Attention output goes through
                # the first writer norm, the MLP output through the last. A
                # ratio ~ 1 means both paths are amplified alike (no
                # mechanistic 1/rho suppression of a class write at c*); a
                # ratio << 1 means only the MLP path is (a class-carrying
                # attention write at c* is suppressed ~1/rho relative to the
                # sink). The theory brief's beta_c note asks exactly this.
                a = report["per_norm"][writers[0]]["at"][k]
                ga = torch.tensor(a["gain"]).abs()
                gf = torch.tensor(w["gain"]).abs()
                ratio = ga / gf.clamp_min(1e-12)
                v["writer_attn_over_ffn_gain_median"] = float(ratio.median())
                v["writer_attn_frac_rank1"] = a["frac_rank1"]
                v["writer_attn_over_ffn_per_layer"] = ratio.tolist()
        else:
            v["writer_gain_dominant"] = None
            v["writer_sign_consistent"] = None
        if readers:
            ratios = [report["per_norm"][n]["at"][k]["abs_over_median"] for n in readers]
            med_ratio = float(torch.tensor(ratios).median())
            v["reader_abs_over_median"] = med_ratio
            v["reader_gain_hidden"] = med_ratio < 0.5           # hidden from content reads
            v["reader_gain_bias_input"] = med_ratio > 2.0       # read as a large constant input
        fn = report.get("final_norm")
        if fn:
            v["final_norm_rank"] = fn["at"][k]["rank"]
            v["final_norm_abs_over_median"] = fn["at"][k]["abs_over_median"]
            # magnitude, not rank: rank is tie-fragile (a model with uniform
            # final gains ranks every channel first)
            v["writes_to_logits"] = fn["at"][k]["abs_over_median"] > 2.0
        out[k] = v
    return out


def candidate_channels(channels_by_position: dict[str, dict[int, list[int]]],
                       acts_neg: Tensor, *, max_k: int = 16) -> tuple[list[int], str, dict[int, int]]:
    """Which channels D7 reports on: the union of the class-blind nuisance sets
    over every position and layer, ranked by the number of (position, layer)
    cells they appear in. When the rule found nothing (a healthy model) fall
    back to the top-|mean| coordinate per layer of the harmless activations so
    the control still produces a table — flagged `top_mean_fallback`.

    Returns (channels, source, counts)."""
    from collections import Counter

    count: Counter = Counter(int(c) for ch in channels_by_position.values()
                             for cs in ch.values() for c in cs)
    cand = [c for c, _ in count.most_common(max_k)]
    if cand:
        return cand, "nuisance", dict(count)
    mu = acts_neg.float().mean(0).abs()                                   # [L, D]
    top = Counter(int(c) for c in mu.argmax(-1))
    return [c for c, _ in top.most_common(max_k)], "top_mean_fallback", dict(count)


# ---- module-tree walking (needs the loaded model) -----------------------------


def _resolve(obj: Any, dotted: str) -> Any:
    return reduce(getattr, dotted.split("."), obj)


def _is_norm(mod: Any) -> bool:
    return "norm" in type(mod).__name__.lower() and hasattr(mod, "weight") \
        and isinstance(getattr(mod, "weight", None), torch.Tensor)


def hf_module(model: Any):
    """The raw transformers module behind a HarmModel (nnsight keeps it at
    `_model`); accepts a bare nn.Module too, for tests."""
    lm = getattr(model, "lm", model)
    for attr in ("_model", "_module"):
        hf = getattr(lm, attr, None)
        if isinstance(hf, torch.nn.Module):
            return hf
    if isinstance(lm, torch.nn.Module):
        return lm
    raise AttributeError("could not find the underlying torch module on the model wrapper")


def find_layout(hf: torch.nn.Module, layers_path: str | None = None) -> dict:
    """Locate the decoder layers, their d_model-sized norms, the final norm and
    the input embedding. Returns names, not tensors."""
    emb = hf.get_input_embeddings()
    d_model = int(emb.weight.shape[1])
    layers = None
    for path in ((layers_path,) if layers_path else ()) + LAYER_PATH_CANDIDATES:
        try:
            cand = _resolve(hf, path)
            len(cand)
            layers = cand
            break
        except (AttributeError, TypeError):
            continue
    if layers is None:
        raise AttributeError("could not locate the decoder layers on the HF module")
    norm_names: list[str] = []
    for name, child in layers[0].named_children():
        if _is_norm(child) and child.weight.numel() == d_model:
            norm_names.append(name)
    final = None
    for name, mod in hf.named_modules():
        if ".layers." in f".{name}." or not _is_norm(mod) or mod.weight.numel() != d_model:
            continue
        final = name                  # the last d_model norm outside the layers
    readers, writers = classify_norms(norm_names)
    return {"d_model": d_model, "n_layers": len(layers), "norm_names": norm_names,
            "reader_norms": readers, "writer_norms": writers, "final_norm_name": final,
            "layers": layers, "embedding": emb, "final_module": _resolve(hf, final) if final else None}


def norm_gain_report(model: Any, channels: list[int], *, layers_path: str | None = None,
                     top_k: int = 8, embed: bool = True) -> dict:
    """The D7 report for the given channels (typically Stage D's nuisance set).

    JSON-serialisable. Reads parameters only; safe to call while the model is
    on GPU (each gain vector is moved to CPU as a float32 copy).
    """
    hf = hf_module(model)
    lay = find_layout(hf, layers_path)
    offset = gain_offset(getattr(lay["layers"][0], lay["norm_names"][0])) if lay["norm_names"] \
        else 0.0
    report: dict[str, Any] = {
        "d_model": lay["d_model"], "n_layers": lay["n_layers"],
        "gain_offset": offset, "norm_names": lay["norm_names"],
        "reader_norms": lay["reader_norms"], "writer_norms": lay["writer_norms"],
        "final_norm_name": lay["final_norm_name"],
        "channels": [int(c) for c in channels],
        "per_norm": {},
    }
    for n in lay["norm_names"]:
        G = torch.stack([offset + getattr(layer, n).weight.detach().float().cpu()
                         for layer in lay["layers"]])                 # [L, D]
        report["per_norm"][n] = channel_stats(G, channels)
        if n in lay["writer_norms"]:
            report["per_norm"][n]["top"] = top_channels(G, top_k)
    if lay["writer_norms"]:
        report["top_by_writer_gain"] = report["per_norm"][lay["writer_norms"][-1]]["top"]
    if lay["final_module"] is not None:
        gf = (offset + lay["final_module"].weight.detach().float().cpu()).unsqueeze(0)
        st = channel_stats(gf, channels)
        report["final_norm"] = {
            "median_abs": st["median_abs_per_layer"][0],
            "max_abs": st["max_abs_per_layer"][0], "argmax": st["argmax_per_layer"][0],
            "at": {k: {"gain": v["gain"][0], "rank": v["rank"][0],
                       "abs_over_median": v["abs_over_median"][0]}
                   for k, v in st["at"].items()},
        }
    if embed and channels:
        E = lay["embedding"].weight.detach()
        D = lay["d_model"]
        s = D ** 0.5                       # Gemma scales embeddings by sqrt(D) on the way in
        cols = E[:, channels].float().cpu()                              # [V, k]
        typical = float(E[: min(E.shape[0], 4096)].float().abs().median()) * s
        report["embedding"] = {
            "scaled_by_sqrt_d": True,
            "typical_abs_entry": typical,
            "at": {str(c): {"col_mean": float(cols[:, i].mean()) * s,
                            "col_sd": float(cols[:, i].std()) * s,
                            "col_abs_max": float(cols[:, i].abs().max()) * s}
                   for i, c in enumerate(channels)},
        }
        bos = getattr(getattr(model, "tokenizer", None), "bos_token_id", None)
        if bos is not None:
            row = E[bos].float().cpu()
            for i, c in enumerate(channels):
                report["embedding"]["at"][str(c)]["bos_row"] = float(row[c]) * s
            report["embedding"]["bos_typical_abs"] = float(row.abs().median()) * s
    report["verdict"] = verdict(report, channels)
    return report


def format_report(report: dict, nuisance_channels: list[int] | None = None) -> str:
    """Compact log block for 23's stdout."""
    lines = [f"D7 norm gains: offset {report['gain_offset']:.0f} | norms {report['norm_names']} "
             f"| writers {report['writer_norms'] or 'NONE (pre-norm family)'} "
             f"| final {report['final_norm_name']}"]
    top = report.get("top_by_writer_gain")
    if top:
        pairs = ", ".join(f"{c}:{g:.1f}" for c, g in zip(top["channels"], top["mean_abs_gain"]))
        overlap = sorted(set(top["channels"]) & set(nuisance_channels or []))
        lines.append(f"  top-{len(top['channels'])} channels by mean |writer gain|: {pairs} "
                     f"(median channel {top['median_channel']:.2f}) | overlap with the "
                     f"activation nuisance set: {overlap or 'none'}")
    for c in report["channels"]:
        k = str(c)
        v = report["verdict"][k]
        parts = [f"  ch {c:5d}:"]
        for n in report["writer_norms"]:
            a = report["per_norm"][n]["at"][k]
            parts.append(f"{n[:8]} g mean {a['mean']:8.1f} rank1 {a['frac_rank1']:.2f} "
                         f"sign {'ok' if a['sign_consistent'] else 'MIXED'} |")
        if "writer_attn_over_ffn_gain_median" in v:
            parts.append(f"attn/ffn gain {v['writer_attn_over_ffn_gain_median']:.2f} |")
        if "reader_abs_over_median" in v:
            parts.append(f"reader |g|/med {v['reader_abs_over_median']:.2f} "
                         f"({'hidden' if v['reader_gain_hidden'] else 'bias-input' if v['reader_gain_bias_input'] else 'ordinary'}) |")
        if "final_norm_rank" in v:
            parts.append(f"final rank {v['final_norm_rank']} "
                         f"({'writes to logits' if v['writes_to_logits'] else 'no direct logit path'})")
        lines.append(" ".join(parts))
    return "\n".join(lines)
