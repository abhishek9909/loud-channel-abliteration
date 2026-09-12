"""Does every Gemma size build the channel? Weights only, no GPU, no model load.

The write-up claims you can tell in advance which models need the correction,
by reading the norm gains rather than by running an abliteration and watching
it fail. This script is that claim, executed across a ladder of models.

It reuses the reader in `check_gemma_norm_gains.py` — which pulls only the
RMSNorm gain vectors straight out of the safetensors shards, a few KB per layer
— and adds two things that a per-model run does not give you:

  * channel DISCOVERY. The per-model script takes `--channel` (defaulting to
    2339, which is the 12B's index and meaningless elsewhere). Here the channel
    is found per model as the modal argmax of the post-FFN gain across layers,
    so a family sweep needs no prior knowledge of where the channel sits.
  * a comparison table with the quantities that decide whether the correction
    is needed: is the channel rank-1 in most layers, is its sign consistent
    across depth, do the readers hide it (reader gain / layer median ~ 0), and
    is it excluded from the logits (final-norm rank near d_model).

Pre-norm families (Llama, Qwen, Mistral) have no post-block norms at all, so
they produce no writer row — that absence is the control, and the script says
so rather than failing.

    python scripts/scan_norm_gain_ladder.py
    python scripts/scan_norm_gain_ladder.py --models google/gemma-3-4b-it google/gemma-3-27b-it
    python scripts/scan_norm_gain_ladder.py --out artifacts/norm_gain_ladder.json

SAFETY: this script NEVER downloads. `check_gemma_norm_gains.resolve_snapshot`
falls back to a full `snapshot_download` on a cache miss, which with
allow_patterns=["*.safetensors", ...] means pulling the entire weights — across
a ladder of nine models, on whatever node you happened to run it from. Here a
miss is an error that names the cache it searched, and `--allow-download` is
required to change that.

ENVIRONMENT: the lookup uses HF_HOME (falling back to ~/.cache/huggingface),
so point it at whichever cache holds the weights before running:

    HF_HOME=/path/to/hf-cache python scripts/scan_norm_gain_ladder.py

The script prints the cache root it is using, so a wrong HF_HOME is visible in
the first line rather than as nine confusing "not cached" rows. Anything outside
the cache can be pointed at directly with --path-for MODEL=/dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    # the Gemma-3 ladder
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    # does it predate instruction tuning?
    "google/gemma-3-12b-pt",
    # is it Gemma-3 or the whole family?
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    # pre-norm controls: expected to have no writer norms at all
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# A "writer" norm is one applied to a sublayer's OUTPUT before it is added back
# to the residual — that is the thing whose per-channel gain can grow a massive
# coordinate. Only `post_feedforward_layernorm` is that. Llama and Qwen also
# carry a `post_attention_layernorm`, but in a pre-norm block that name means
# the norm applied to the residual BEFORE the MLP reads it: a reader, not a
# writer. Treating it as a writer made the pre-norm controls scan as post-norm
# models and report "no dominant writer channel" instead of being recognised as
# the architectural control they are.
WRITERS = ("post_feedforward_layernorm",)
READERS = ("input_layernorm", "pre_feedforward_layernorm")


def hf_cache_root() -> str:
    """Where the Hub cache actually is, for the banner and the error message."""
    import os
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        if os.environ.get(var):
            root = os.environ[var]
            return root if var == "HF_HUB_CACHE" else str(Path(root) / "hub")
    return str(Path.home() / ".cache" / "huggingface" / "hub")


def resolve_local(model: str, path: str | None, *, allow_download: bool) -> Path:
    """Snapshot path WITHOUT the silent full-weights download on a miss."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"--path-for points at {p}, which does not exist")
        return p
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors", "*.safetensors.index.json", "config.json"]
    try:
        return Path(snapshot_download(model, allow_patterns=pats, local_files_only=True))
    except Exception as e:
        if not allow_download:
            raise FileNotFoundError(
                f"not in the Hub cache at {hf_cache_root()} "
                f"(set HF_HOME to the cache that holds it). "
                f"Refusing to download; pass --allow-download to override.") from e
        return Path(snapshot_download(model, allow_patterns=pats))


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "check_gemma_norm_gains", REPO / "scripts" / "check_gemma_norm_gains.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_model(hlp, model: str, path: str | None, *, allow_download: bool = False) -> dict:
    import numpy as np

    snap = resolve_local(model, path, allow_download=allow_download)
    index = hlp.tensor_index(snap)
    per_layer: dict[int, dict[str, str]] = defaultdict(dict)
    for name in index:
        m = hlp.LAYER_RE.search(name)
        if m:
            per_layer[int(m.group(1))][m.group(2)] = name
    if not per_layer:
        return {"model": model, "error": "no decoder-layer norms found"}
    n_layers = max(per_layer) + 1

    has_writer = any(w in per_layer[0] for w in WRITERS)
    if not has_writer:
        return {"model": model, "n_layers": n_layers, "family": "pre-norm",
                "writer_norms": [], "note": "no post-block norms — the control case"}

    writer = "post_feedforward_layernorm"
    argmax_per_layer, gains = [], []
    for L in range(n_layers):
        g = 1.0 + hlp.load(index, per_layer[L][writer])
        gains.append(g)
        argmax_per_layer.append(int(np.abs(g).argmax()))
    channel = Counter(argmax_per_layer).most_common(1)[0][0]
    d_model = len(gains[0])

    g_c = np.array([g[channel] for g in gains])
    ranks = [int((np.abs(g) > abs(g[channel])).sum()) + 1 for g in gains]
    med = np.array([float(np.median(np.abs(g))) for g in gains])

    reader_ratio = None
    rn = next((r for r in READERS if r in per_layer[0]), None)
    if rn:
        rr = []
        for L in range(n_layers):
            g = 1.0 + hlp.load(index, per_layer[L][rn])
            rr.append(abs(float(g[channel])) / max(float(np.median(np.abs(g))), 1e-9))
        reader_ratio = float(np.median(rr))

    final_name = next((k for k in index if re.search(r"(^|\.)model\.norm\.weight$", k)), None)
    final_rank = final_ratio = None
    if final_name is not None:
        gf = 1.0 + hlp.load(index, final_name)
        final_rank = int((np.abs(gf) > abs(gf[channel])).sum()) + 1
        final_ratio = abs(float(gf[channel])) / max(float(np.median(np.abs(gf))), 1e-9)

    return {
        "model": model, "family": "post-norm", "n_layers": n_layers,
        "d_model": d_model, "writer_norm": writer,
        "channel": channel,
        "channel_modal_share": Counter(argmax_per_layer).most_common(1)[0][1] / n_layers,
        "writer_gain_median": float(np.median(np.abs(g_c))),
        "writer_gain_max": float(np.abs(g_c).max()),
        "layer_median_gain": float(np.median(med)),
        "gain_over_median": float(np.median(np.abs(g_c) / np.maximum(med, 1e-9))),
        "frac_rank1": float(sum(r == 1 for r in ranks) / n_layers),
        "frac_rank_le8": float(sum(r <= 8 for r in ranks) / n_layers),
        "sign_consistent": bool(len(set(np.sign(g_c[np.abs(g_c) > 1e-6]).tolist())) == 1),
        "reader_gain_over_median": reader_ratio,
        "final_norm_rank": final_rank,
        "final_norm_over_median": final_ratio,
        "argmax_per_layer": argmax_per_layer,
    }


def verdict(r: dict) -> str:
    """Does this model need the correction, judged from weights alone?"""
    if r.get("family") == "pre-norm":
        return "no writer norms (control)"
    if "error" in r:
        return "unreadable"
    # `x or 1.0` is wrong here: a channel that is PERFECTLY muted has gain
    # exactly 0.0, and `0.0 or 1.0` is 1.0, so the cleanest cases scored as
    # "not muted" (gemma-3-4b, -27b and 12b-pt all did). Missing measurements
    # are None and must be the only thing that falls back to the neutral 1.0.
    def _ratio(key: float | None) -> float:
        return 1.0 if key is None else float(key)

    # GAIN RATIO IS THE PRIMARY SIGNAL; everything else is confirmatory.
    # Measured across eight models against whether raw difference-of-means
    # actually yields a usable cell:
    #
    #   gemma-2-2b   2.9x  ->  2/156 raw cells, median ablation KL 0.57  WORKS
    #   gemma-2-9b   3.4x  -> 33/252 raw cells, KL 0.20                  WORKS
    #   gemma-3-1b   7.1x  ->  0/156 raw cells, KL 20.95                 BREAKS
    #   gemma-3-4b   9.5x  ->  0/204,  KL 29.28                          BREAKS
    #   gemma-3-27b 10.7x  ->  0/186,  KL 22.09                          BREAKS
    #   gemma-3-12b 11.7x  ->  0/288,  KL 30.06                          BREAKS
    #
    # A cut anywhere in 3.5-7 separates them; 5.0 is the midpoint of the gap.
    # This is a separating statistic fitted on eight models, not a validated
    # threshold -- treat a model landing near the cut as undecided and measure.
    #
    # An earlier version made `hidden` and `mute` mandatory conjuncts, which
    # demoted gemma-3-1b to "loud writer, but read/logit path not hidden" on a
    # reader ratio of 0.1051 against a hand-set 0.1 cutoff. The 1B is in fact
    # the most broken model on the ladder by KL per candidate. A hand-set
    # cutoff on a secondary check must not overturn a gain ratio 7x the median.
    rank = r.get("final_norm_rank")
    d_model = r.get("d_model") or 0
    loud = r["frac_rank1"] >= 0.8 and r["sign_consistent"]
    dominant = r["gain_over_median"] >= 5.0
    hidden = _ratio(r["reader_gain_over_median"]) < 0.15
    mute = _ratio(r["final_norm_over_median"]) < 0.1 or (
        rank is not None and d_model and rank >= 0.99 * d_model)
    if dominant and (loud or hidden or mute):
        return "EXPECT the channel"
    if dominant:
        return "loud writer, nothing else matches - measure it"
    return "no dominant writer channel"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--path-for", nargs="*", default=[],
                    help="MODEL=/local/snapshot overrides for models not in HF_HOME")
    ap.add_argument("--out", default=None, help="write the full JSON here")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching weights that are not in the cache. Off by "
                         "default: a full-weights download of this ladder is hundreds "
                         "of GB and must never happen by accident.")
    args = ap.parse_args()
    print(f"Hub cache: {hf_cache_root()}"
          f"{'' if __import__('os').environ.get('HF_HOME') or __import__('os').environ.get('HF_HUB_CACHE') else '   (HF_HOME unset — using the default cache)'}")
    paths = dict(kv.split("=", 1) for kv in args.path_for)

    hlp = _load_helper()
    rows = []
    for m in args.models:
        try:
            r = scan_model(hlp, m, paths.get(m), allow_download=args.allow_download)
        except Exception as e:                     # a missing snapshot is normal
            r = {"model": m, "error": f"{type(e).__name__}: {e}"}
        r["verdict"] = verdict(r)
        rows.append(r)

    print(f"\n{'model':34s} {'L':>3s} {'d_model':>7s} {'chan':>6s} {'rank1':>6s} "
          f"{'gain':>8s} {'/med':>6s} {'sign':>5s} {'read/med':>9s} {'finalrk':>8s}  verdict")
    for r in rows:
        if "error" in r:
            print(f"{r['model'][:34]:34s} {'':>3s} {'':>7s} {'':>6s} {'':>6s} "
                  f"{'':>8s} {'':>6s} {'':>5s} {'':>9s} {'':>8s}  {r['error'][:60]}")
            continue
        if r.get("family") == "pre-norm":
            print(f"{r['model'][:34]:34s} {r['n_layers']:3d} {'':>7s} {'':>6s} {'':>6s} "
                  f"{'':>8s} {'':>6s} {'':>5s} {'':>9s} {'':>8s}  {r['verdict']}")
            continue
        print(f"{r['model'][:34]:34s} {r['n_layers']:3d} {r['d_model']:7d} "
              f"{r['channel']:6d} {r['frac_rank1']:6.2f} {r['writer_gain_median']:8.1f} "
              f"{r['gain_over_median']:6.1f} {str(r['sign_consistent']):>5s} "
              f"{(r['reader_gain_over_median'] if r['reader_gain_over_median'] is not None else float('nan')):9.4f} "
              f"{(r['final_norm_rank'] if r['final_norm_rank'] is not None else -1):8d}  {r['verdict']}")

    print("\n  chan     = modal argmax of the post-FFN gain across layers (discovered, not assumed)"
          "\n  rank1    = fraction of layers where that channel has the largest |gain|"
          "\n  /med     = its gain over the layer's median gain"
          "\n  read/med = the READER gain at that channel over the layer median; ~0 means no"
          "\n             sublayer reads it as a coordinate, so it acts only through the RMS norm"
          "\n  finalrk  = its rank in the final norm; near d_model means no path to the logits"
          "\n\n  A model with 'EXPECT the channel' will have a contaminated difference-of-means"
          "\n  direction and needs the class-blind correction. A pre-norm model has no such"
          "\n  writer at all. This is decidable before running any abliteration.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"models": rows}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
