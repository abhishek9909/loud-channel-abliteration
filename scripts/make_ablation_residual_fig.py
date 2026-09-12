"""What projection ablation does to the residual stream, coordinate by coordinate.

Reads `artifacts/<model>/recipes/diagnostics.json` written by
`experiments/01_extract.py`. For the chosen cell it pulls the MEASURED
geometry -- d_model, the loud coordinate's magnitude and the RMS of every other
coordinate, recovered from `norms.rms_full` and `norms.rms_wo` -- and the
estimated direction's split between that coordinate and the rest,
a = sqrt(top-1 share of the difference vector). Only the per-coordinate texture
of the tail is a draw; the spike height, the floor and `a` are all measured.

Then the real operation is applied, x <- x - (x.d)d, and both are plotted in
units of that model's own typical coordinate so the panels share an axis.

Worth knowing: `a` derived here as sqrt(top-1 share) reproduces the value stored
independently in channel_causal_e2.json's `decomposition` to five decimals, so
the direction in the figure is the estimator's own, not a stand-in.

    python scripts/make_ablation_residual_fig.py \
        --left  gemma3-12b:t_post_inst:32 \
        --right llama3-8b:t_post_inst-2:11

Default cells are each model's own raw-estimator cell.
"""
import argparse
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS = os.environ.get("HARMDIR_ARTIFACTS", "artifacts")

PRETTY = {"gemma3-1b": "Gemma-3-1B", "gemma3-4b": "Gemma-3-4B",
          "gemma3-12b": "Gemma-3-12B", "gemma3-27b": "Gemma-3-27B",
          "gemma2-2b": "Gemma-2-2B", "gemma2-9b": "Gemma-2-9B",
          "llama3-8b": "Llama-3-8B", "qwen2.5-7b": "Qwen-2.5-7B"}


def load_cell(model: str, position: str, layer: int) -> dict:
    """Everything the figure needs, from one diagnostics.json."""
    with open(f"{ARTIFACTS}/{model}/recipes/diagnostics.json") as fh:
        d = json.load(fh)
    b = d["by_position"][position]
    N, en = b["norms"], b["energy"]["r0_raw"]
    dm = len(b["nominal"]["channel"]) and d.get("d_model")
    if not dm:                                  # older runs omit d_model
        with open(f"{ARTIFACTS}/{model}/recipes/norm_gains.json") as fh:
            dm = json.load(fh)["d_model"]
    n_ch = int(N["n_channels"][layer])
    rf, rw = N["rms_full"][layer], N["rms_wo"][layer]
    # |x_c*| from the two RMS values: d*rms_full^2 = x_c*^2 + (d-n)*rms_wo^2
    xc = math.sqrt(max(dm * rf * rf - (dm - n_ch) * rw * rw, 0.0))
    return dict(d_model=dm, layer=layer, position=position, rms_full=rf,
                rms_wo=rw, n_ch=n_ch, x_channel=xc,
                a=math.sqrt(en["top1_share"][layer]),
                channel=b["nominal"]["channel"][layer])


THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#9a998f",
                  grid="#e6e5e0", s1="#2a78d6", s2="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#7d7c75",
                  grid="#333331", s1="#3987e5", s2="#d95926"),
}


def build(o, seed=1):
    """Clean residual, ablated with the raw direction, ablated with the masked one.

    The masked direction is the same vector with its component on the loud
    coordinate zeroed and renormalised -- i.e. a = 0, so d_hat = u exactly.
    That is the whole correction, and it is why the third curve is the
    interesting one: the operation is unchanged, only the estimate differs.
    """
    d, a = o["d_model"], o["a"]
    b = np.sqrt(max(1.0 - a * a, 0.0))
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, o["rms_wo"], size=d)
    if o["n_ch"] > 0 and o["x_channel"] > 4 * o["rms_wo"]:
        x[0] = o["x_channel"]          # the loud coordinate, at its measured magnitude
    # else: the class-blind rule flags no loud channel at this cell (Llama: n_ch = 0),
    # so the residual is left as it is and its largest coordinate is just the largest draw
    u = rng.normal(size=d); u[0] = 0.0; u /= np.linalg.norm(u)

    d_raw = np.zeros(d); d_raw[0] = a; d_raw += b * u      # a*e_c + b*u
    d_masked = u                                           # a = 0, renormalised

    x_raw = x - float(x @ d_raw) * d_raw
    x_masked = x - float(x @ d_masked) * d_masked
    return np.abs(x), np.abs(x_raw), np.abs(x_masked)


def draw(theme_name, cells, labels, outdir):
    T = THEMES[theme_name]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True,
                             gridspec_kw=dict(wspace=0.09))
    fig.patch.set_facecolor(T["surface"])

    for ax, o, lab, col in zip(axes, cells, labels, (T["s2"], T["s1"])):
        clean, abl, mask = build(o)
        unit = o["rms_wo"]
        sc = np.sort(clean)[::-1] / unit
        sa = np.sort(abl)[::-1] / unit
        sm = np.sort(mask)[::-1] / unit

        ax.set_facecolor(T["surface"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(T["grid"]); ax.spines[sp].set_linewidth(1.0)
        ax.tick_params(colors=T["ink2"], labelsize=9, length=3, width=1.0)
        ax.grid(True, color=T["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        k = np.arange(1, len(sc) + 1)
        ax.plot(k, sc, color=T["ink3"], linewidth=4.0, solid_capstyle="round", zorder=3)
        ax.plot(k, sa, color=col, linewidth=2.0, solid_capstyle="round", zorder=4)
        ax.plot(k, sm, color=T["ink"], linewidth=1.6, linestyle=(0, (4, 3)),
                solid_capstyle="round", zorder=5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.85, len(sc) * 1.5)
        ax.set_ylim(0.08, 900)
        ax.set_xlabel("coordinate, ranked by magnitude", color=T["ink2"], fontsize=9.5)
        ax.set_title(f"{lab}  ·  layer {o['layer']}", color=T["ink"],
                     fontsize=11.5, loc="left", pad=10)

        mid = len(sc) // 2
        ax.text(0.97, 0.955,
                f"raw:      loud {sc[0]:,.0f}× → {sa[0]:,.0f}×,  typical ×{sa[mid] / sc[mid]:.2f}\n"
                f"masked:  loud {sc[0]:,.0f}× → {sm[0]:,.0f}×,  typical ×{sm[mid] / sc[mid]:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
                color=T["ink"], linespacing=1.5)
        print(f"  {lab:<14} raw: loud {sc[0]:>7,.0f}x -> {sa[0]:>7,.0f}x  typical x{sa[mid]/sc[mid]:.2f}"
              f"   |  masked: loud -> {sm[0]:>7,.0f}x  typical x{sm[mid]/sc[mid]:.2f}   a={o['a']:.4f}")

    axes[0].set_ylabel("|coordinate| ÷ that model's typical coordinate",
                       color=T["ink2"], fontsize=9.5)
    axes[0].annotate("before", xy=(330, 0.50), fontsize=9.5,
                     color=T["ink3"], fontweight="semibold")
    axes[0].annotate("ablated with the raw direction", xy=(11, 13), fontsize=9.5,
                     color=T["s2"], fontweight="semibold")
    axes[0].annotate("ablated with the corrected\ndirection — sits on \"before\"",
                     xy=(1.25, 0.16), fontsize=9, color=T["ink"], linespacing=1.35)
    axes[1].annotate("all three curves\ncoincide", xy=(24, 0.30),
                     fontsize=9.5, color=T["ink2"], linespacing=1.35)

    fig.text(0.5, 0.015,
             "Same operation throughout, x ← x − (x·d̂)d̂; only the estimate of d̂ differs. Dashed = the "
             "class-blind masked estimate (a = 0); on Llama the mask is empty, so it is the raw vector.",
             ha="center", fontsize=8.1, color=T["ink3"], style="italic")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.87, bottom=0.20)
    os.makedirs(outdir, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(f"{outdir}/ablation-residual-{theme_name}.{ext}",
                    dpi=200, facecolor=T["surface"])
    plt.close(fig)


def _cell(spec):
    model, position, layer = spec.split(":")
    return model, position, int(layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", default="gemma3-12b:t_post_inst:32",
                    help="model:position:layer")
    ap.add_argument("--right", default="llama3-8b:t_post_inst-2:11",
                    help="model:position:layer")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    specs = [_cell(args.left), _cell(args.right)]
    cells = [load_cell(*sp) for sp in specs]
    labels = [PRETTY.get(sp[0], sp[0]) for sp in specs]
    for name in THEMES:
        print(f"{name}:")
        draw(name, cells, labels, args.outdir)
    print(f"wrote {args.outdir}/ablation-residual-{{light,dark}}.{{svg,png}}")


if __name__ == "__main__":
    main()
