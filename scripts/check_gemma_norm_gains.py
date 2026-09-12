#!/usr/bin/env python3
"""Weight-level check of the Gemma massive channel.

The activation-side measurement says a channel is loud; this says the weights
build it, without loading the model.

Reads ONLY the RMSNorm gain vectors (4 per layer + final norm, 3840 bf16 each)
straight out of the safetensors shards with a tiny numpy reader — no torch,
no safetensors package, no model load, no GPU; ~1 s once the shards are in the
HF cache (needs `huggingface_hub` only when `--path` is not given). Optional:
the embedding column of the channel (`--embed`, reads the 2 GB embed_tokens).

Gemma-3 decoder block (transformers modeling_gemma3.py):
    x = x + post_attn_norm( attn( input_norm(x) ) )
    x = x + post_ffn_norm ( mlp ( pre_ffn_norm(x) ) )
    RMSNorm(v) = v / rms(v) * (1 + w)            # gain is (1 + w), not w
so a channel c receives, per layer, `(1+w_post)[c] * n[c]` where n has unit
RMS. A large fixed-sign (1+w_post)[c] in every layer is the direct mechanism
for a residual channel that grows monotonically across depth at every token.
The pre-norm gains (1+w_pre)[c] say whether downstream blocks *read* the
channel as content (gain ~1) or hide it (gain ~0), leaving it to act only
through the RMS denominator.

Usage:
    python scripts/check_gemma_norm_gains.py --model google/gemma-3-12b-it --channel 2339
    python scripts/check_gemma_norm_gains.py --path /path/to/snapshot --channel 2339 --embed
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

NORMS = ("input_layernorm", "post_attention_layernorm",
         "pre_feedforward_layernorm", "post_feedforward_layernorm")
LAYER_RE = re.compile(r"layers\.(\d+)\.(" + "|".join(NORMS) + r")\.weight$")


def resolve_snapshot(model: str | None, path: str | None) -> Path:
    if path:
        return Path(path)
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors", "*.safetensors.index.json", "config.json"]
    try:  # cached copy first — never triggers a download
        return Path(snapshot_download(model, allow_patterns=pats, local_files_only=True))
    except Exception:
        return Path(snapshot_download(model, allow_patterns=pats))


# ---- minimal safetensors reader (numpy only; no torch / safetensors needed) ----
# format: u64 LE header length, JSON header {name: {dtype, shape, data_offsets}}, raw bytes.
_HEADERS: dict[Path, tuple[dict, int]] = {}


def _header(shard: Path) -> tuple[dict, int]:
    if shard not in _HEADERS:
        with open(shard, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        _HEADERS[shard] = (hdr, 8 + n)
    return _HEADERS[shard]


def tensor_index(snap: Path) -> dict[str, Path]:
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: snap / v for k, v in wm.items()}
    return {k: shard for shard in sorted(snap.glob("*.safetensors")) for k in _header(shard)[0]}


def load(index: dict[str, Path], name: str):
    """Return the tensor as float32 numpy, reading only its bytes from the shard."""
    import numpy as np
    shard = index[name]
    hdr, base = _header(shard)
    meta = hdr[name]
    a, b = meta["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    dt = meta["dtype"]
    if dt == "BF16":  # bf16 -> f32: place the 16 bits in the high half of a u32
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        arr = u16.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype={"F32": "<f4", "F16": "<f2", "F64": "<f8"}[dt]).astype(np.float32)
    return arr.reshape(meta["shape"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--path", default=None, help="local snapshot dir (skips hub lookup)")
    ap.add_argument("--channel", type=int, default=2339)
    ap.add_argument("--others", default="107,670,2551,227,2142,1678,3499",
                    help="comma-separated secondary channels to report alongside")
    ap.add_argument("--embed", action="store_true", help="also read embed_tokens[:, channel]")
    args = ap.parse_args()
    import numpy as np

    snap = resolve_snapshot(args.model, args.path)
    index = tensor_index(snap)
    c = args.channel
    others = [int(x) for x in args.others.split(",") if x.strip()]

    # ---- locate per-layer norms (name prefix differs across transformers versions)
    per_layer: dict[int, dict[str, str]] = defaultdict(dict)
    for name in index:
        m = LAYER_RE.search(name)
        if m:
            per_layer[int(m.group(1))][m.group(2)] = name
    if not per_layer:
        raise SystemExit(f"no decoder-layer norms found in {snap}; keys look like: {list(index)[:5]}")
    n_layers = max(per_layer) + 1
    final_norm = next((k for k in index if re.search(r"(^|\.)model\.norm\.weight$", k)), None)
    print(f"snapshot: {snap}\nlayers: {n_layers}   channel: {c}   final norm: {final_norm}\n")

    # ---- per-layer gains at channel c, with context (median / max over channels)
    hdr = (f"{'L':>2} | " + " | ".join(f"{n[:9]:>9}" for n in NORMS)
           + " || post_ffn: med|g| max|g| (argmax) rank(c) | pre_ffn: med|g| rank(c)")
    print("gain g = (1 + w)[channel]   (rank = rank of |g[c]| among all channels, 1 = largest)")
    print(hdr)
    print("-" * len(hdr))
    gains = {n: np.zeros(n_layers) for n in NORMS}
    gain_vecs_post_ffn = []
    for L in range(n_layers):
        row = []
        ctx = ""
        for n in NORMS:
            g = 1.0 + load(index, per_layer[L][n])
            gains[n][L] = g[c]
            row.append(f"{g[c]:9.2f}")
            ag = np.abs(g)
            rank = int((ag > ag[c]).sum()) + 1
            if n == "post_feedforward_layernorm":
                gain_vecs_post_ffn.append(g)
                ctx += (f" || {np.median(ag):6.2f} {ag.max():8.1f} ({int(ag.argmax()):4d}) {rank:5d}")
            if n == "pre_feedforward_layernorm":
                ctx += f" | {np.median(ag):6.2f} {rank:5d}"
        print(f"{L:2d} | " + " | ".join(row) + ctx)

    print("\nsummary at channel", c)
    for n in NORMS:
        g = gains[n]
        print(f"  {n:26s}: mean {g.mean():9.2f}  min {g.min():9.2f}  max {g.max():9.2f}  "
              f"sign-consistent over layers: {np.all(np.sign(g) == np.sign(g[g != 0][0])) if np.any(g != 0) else 'n/a'}")
    if final_norm:
        gf = 1.0 + load(index, final_norm)
        agf = np.abs(gf)
        print(f"  final norm                 : g[c] = {gf[c]:9.3f}   median |g| = {np.median(agf):.3f}   "
              f"rank(c) = {int((agf > agf[c]).sum()) + 1}")

    # ---- which channels do the post-FFN gains single out overall? (model-wide top-8)
    G = np.stack(gain_vecs_post_ffn)              # [L, D]
    score = np.abs(G).mean(0)
    top = np.argsort(-score)[:8]
    print("\nchannels with largest mean |post_ffn gain| across layers:")
    for ch in top:
        print(f"  ch {int(ch):4d}: mean |g| = {score[ch]:8.2f}   (median channel: {np.median(score):.2f})")
    print("secondary channels from the activation cache:",
          {ch: round(float(score[ch]), 2) for ch in others})

    # ---- optional: embedding column (tied to lm_head in Gemma)
    if args.embed:
        ename = next((k for k in index if k.endswith("embed_tokens.weight")), None)
        if ename is None:
            print("\nembed_tokens not found")
        else:
            E = load(index, ename)                # [V, D]
            D = E.shape[1]
            col = E[:, c]
            print(f"\nembed_tokens[:, {c}] * sqrt(D={D}): mean {col.mean()*math.sqrt(D):8.2f}  "
                  f"sd {col.std()*math.sqrt(D):8.2f}  |max| {np.abs(col).max()*math.sqrt(D):8.2f}")
            cfg = json.loads((snap / "config.json").read_text())
            tc = cfg.get("text_config", cfg)
            bos = tc.get("bos_token_id", cfg.get("bos_token_id"))
            if bos is not None:
                print(f"  BOS row at channel {c}: {E[bos, c]*math.sqrt(D):8.2f}   "
                      f"(typical |entry| * sqrt(D) over all channels of the BOS row: "
                      f"{np.median(np.abs(E[bos]))*math.sqrt(D):.2f})")
            print("  (lm_head is tied to embed_tokens in Gemma, so this column, times the final-norm "
                  "gain, is also the channel's direct write into the logits)")


if __name__ == "__main__":
    main()
