"""Stage I — steering-onset sweep: does the cleaned vector fire at alpha ~ ||x_perp||?

Prediction 4 (README.md): on Gemma the original vector needed a
recalibrated alpha = 2048 at the pinned L12 cell because only the b-fraction
of the unit vector (d_hat = a e_c* + b u, b ~ 0.47 at L12) points anywhere but
the bias coordinate — the useful dose is alpha*b, so onset sits at ~1/b times
||x_perp||. A vector with the nuisance coordinates removed has b = 1 and should
show its onset at ~1x ||x_perp|| (perp_norm ~ 1e3 at L12). Stages 23-25 never
sweep alpha (Stage S adds once, at the raw norm), so this stage does.

Instrument, shared by every arm so the comparison is internal:
  prompts   the HARMLESS val slice 24 used (same subset ordering, from
            directions.pt), so nothing here touched the direction estimate
  dose      alpha = m x perp_norm[layer] for m in --multiples (log2 grid), the
            unit direction ADDED at that one layer at every token position
            (Stage-S `induce` convention)
  readout   selection.refusal_metric — logit of the refusal-opening mass at
            the first response token; `frac` = share of prompts with logit > 0
  onset     smallest m with frac >= 1/2 (selection.dose_onset); reported as a
            grid cell, plus KL(clean || steered) at every dose
  check     one greedy generation pass at the onset dose, three-way scored
            (degeneracy.summarize) — a collapse also lifts the "I"/"Sorry"
            mass, so an "onset" that is really degeneration must be labelled

Arms (each carries b = perp_fraction, so 1/b is its predicted onset multiple):
  <recipe>@sel          each recipe's selected cell from selection.json
  r0_raw@r1cell         the raw vector at r1_masked's cell — same cell, only
                        the estimator differs (the cleanest pairwise contrast)
  legacy_L<l>           the ORIGINAL d_refuse.pt row at --legacy-layer (12 =
                        the cell whose recalibrated dose was 2048)
  random                norm-matched random direction at r1's cell (null)

Note the 2048 figure came from single-position steering with the H1 judgment
readout; this stage does not reproduce that number, it asks whether the ORDER
of onsets (legacy / r0 at ~1/b, r1 at ~1) holds under one instrument.

    python experiments/04_dose.py --model gemma3-12b
    python experiments/04_dose.py --model llama3-8b --no-gen
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.config import artifacts_dir
from loudchannel.data.datasets import load_rows
from loudchannel.data.splits import load_split, source_counts, subset_rows
from loudchannel.degeneracy import summarize
from loudchannel.directions import Direction
from loudchannel.interventions import generate_with_intervention, logits_with_intervention
from loudchannel.positions import positions_for
from loudchannel.selection import (
    broadcast_direction,
    dose_onset,
    first_token_kl,
    induced_fraction,
    perp_fraction,
    refusal_metric,
)

SUB = "recipes"
DEFAULT_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--multiples", nargs="+", type=float, default=list(DEFAULT_MULTIPLES),
                    help="dose grid as multiples of perp_norm at the arm's layer")
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--legacy-sub", default="directions")
    ap.add_argument("--legacy-layer", type=int, default=12,
                    help="layer of the original d_refuse to sweep (12 = the "
                         "pinned Section-B cell recalibrated to alpha=2048)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--no-gen", action="store_true",
                    help="skip the generation check at the onset dose")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    assert args.multiples == sorted(args.multiples), "--multiples must ascend"
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)
    n_layers = model.n_layers

    blob = torch.load(adir / "directions.pt", map_location="cpu", weights_only=False)
    sel = json.loads((adir / "selection.json").read_text())["by_recipe"]
    diag = json.loads((adir / "diagnostics.json").read_text())["by_position"]
    channels = json.loads((adir / "channels.json").read_text())
    assert blob["n_layers"] == n_layers, "directions.pt came from a different model"
    lo, hi = blob["val_slice"]
    subset = blob.get("subset", "head")

    dataset = blob.get("dataset", "default")
    rows = subset_rows(load_rows(dataset, "harmless_extract"), subset)[lo:hi]
    if args.limit:
        rows = rows[: args.limit]
    harmless = instructions(rows)
    pps = positions_for(model, harmless, seed=exp["seed"])
    bs = len(pps)
    print(f"harmless val {len(pps)} | subset={subset} {source_counts(rows)} "
          f"| multiples {args.multiples}")

    clean = logits_with_intervention(model, pps, direction=None, layers=[],
                                     position=None, mode="none", batch_size=bs)
    clean_metric = refusal_metric(clean, model.tokenizer)
    print(f"clean: refusal metric {float(clean_metric.mean()):+.3f} | "
          f"frac>0 {induced_fraction(clean_metric):.3f}")

    # ---- arms -------------------------------------------------------------
    def nuis(pos: str, layer: int) -> list[int]:
        return channels.get(pos, {}).get(str(layer), [])

    arms: dict[str, dict] = {}
    for recipe, r in sel.items():
        s = r.get("selected")
        if not s:
            print(f"  {recipe}: no feasible candidate — arm skipped")
            continue
        arms[f"{recipe}@sel"] = {
            "recipe": recipe, "layer": s["layer"], "position": s["position"],
            "row": blob["directions"][recipe][s["position"]][s["layer"]],
        }
    # the r0@cell / random contrast is anchored on r1's cell if it has one,
    # else the first recipe that does (so the arm is not lost when the
    # selection rule leaves r1 infeasible)
    anchor_recipe = "r1_masked" if sel.get("r1_masked", {}).get("selected") else \
        next((rc for rc, r in sel.items() if r.get("selected")), None)
    anchor = sel.get(anchor_recipe, {}).get("selected") if anchor_recipe else None
    if anchor:
        pos1, l1 = anchor["position"], anchor["layer"]
        arms[f"r0_raw@{anchor_recipe}cell"] = {
            "recipe": "r0_raw", "layer": l1, "position": pos1,
            "row": blob["directions"]["r0_raw"][pos1][l1]}
        ref_row = blob["directions"][anchor_recipe][pos1][l1]
        g = torch.Generator().manual_seed(exp["seed"])
        rnd = torch.randn(ref_row.shape[-1], generator=g)
        rnd = rnd / rnd.norm() * ref_row.norm()
        arms["random"] = {"recipe": "random", "layer": l1, "position": pos1, "row": rnd}
        # random null orthogonal to the nuisance coords at this (position, layer):
        # on Gemma it drops the leak, on Llama/Qwen it equals `random`
        nuis_here = channels.get(pos1, {}).get(str(l1), [])
        rperp = rnd.clone()
        if nuis_here:
            rperp[nuis_here] = 0.0
            rperp = rperp / rperp.norm().clamp_min(1e-12) * float(ref_row.norm())
        arms["random_perp"] = {"recipe": "random_perp", "layer": l1, "position": pos1,
                               "row": rperp}
    else:
        print("  no feasible cell on any recipe — r0_raw@cell / random arms skipped")
    legacy_p = artifacts_dir(exp, args.model, args.legacy_sub) / "d_refuse.pt"
    if legacy_p.exists():
        leg = Direction.load(legacy_p)
        arms[f"legacy_L{args.legacy_layer}"] = {
            "recipe": "legacy", "layer": args.legacy_layer, "position": leg.position,
            "row": leg.vec[args.legacy_layer].float(),
        }
    else:
        print(f"  legacy arm skipped ({legacy_p} not found)")
    if args.arms:
        arms = {k: v for k, v in arms.items() if k in args.arms}
    if not arms:
        raise SystemExit("no arms to sweep")

    # ---- sweep ------------------------------------------------------------
    results: dict[str, dict] = {}
    for name, arm in arms.items():
        pos, layer, row = arm["position"], arm["layer"], arm["row"]
        # perp_norm / nuisance set are class-blind per (position, layer); if
        # the arm's position was not in 23's grid (a legacy vector at a
        # position that was not extracted) use the first grid position and say so
        dpos = pos if pos in diag else next(iter(diag))
        if dpos != pos:
            print(f"  note: no diagnostics for position {pos!r}; using {dpos!r}'s perp_norm")
        pn = diag[dpos]["perp_norm"][layer]
        b = perp_fraction(row, nuis(dpos, layer))
        d = broadcast_direction(row, n_layers, kind=f"dose|{name}",
                                position=pos, model_name=args.model)
        print(f"\n=== arm {name}  L{layer}@{pos}  perp_norm {pn:.0f}  "
              f"||row|| {float(row.norm()):.0f}  b {b:.3f} (1/b {1 / max(b, 1e-6):.2f}) ===")
        sweep, fracs = [], []
        for m in args.multiples:
            alpha = m * pn
            logits = logits_with_intervention(
                model, pps, direction=d, layers=[layer], position="all",
                mode="steer", alpha=alpha, batch_size=bs)
            metric = refusal_metric(logits, model.tokenizer)
            frac = induced_fraction(metric)
            kl = float(first_token_kl(clean, logits).mean())
            fracs.append(frac)
            sweep.append({"multiple": m, "alpha": alpha,
                          "refusal_metric": float(metric.mean()),
                          "frac_refusal": frac, "kl": kl})
            print(f"  m {m:5.2f}  alpha {alpha:9.1f}  metric {float(metric.mean()):+7.3f}  "
                  f"frac {frac:.3f}  kl {kl:.4f}")
        onset = dose_onset(args.multiples, fracs, threshold=args.onset_threshold)
        entry = {
            "recipe": arm["recipe"], "layer": layer, "position": pos,
            "perp_norm": pn, "row_norm": float(row.norm()),
            "perp_fraction_b": b, "predicted_onset_multiple": 1 / max(b, 1e-6),
            "sweep": sweep,
            "onset_multiple": onset,
            "onset_alpha": (onset * pn) if onset is not None else None,
        }
        onset_str = f"m={onset:.2f} (alpha {onset * pn:.0f})" if onset is not None \
            else "NONE in grid"
        print(f"  onset: {onset_str}  | predicted from b: m~{1 / max(b, 1e-6):.2f}")

        if not args.no_gen:
            m_gen = onset if onset is not None else args.multiples[-1]
            alpha = m_gen * pn
            texts = generate_with_intervention(
                model, [model.render(i) for i in harmless], direction=d,
                layers=[layer], mode="steer", alpha=alpha, position="all",
                max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size)
            s = summarize(texts)
            entry["gen_at_onset"] = {
                "multiple": m_gen, "alpha": alpha, "at_onset": onset is not None,
                **{k: v for k, v in s.items() if k != "rows"},
            }
            (adir / f"transcripts_dose_{name.replace('@', '_')}.jsonl").write_text(
                "\n".join(json.dumps({"i": i, "arm": name, "alpha": alpha,
                                      "instruction": ins, "transcript": t, **lab},
                                     ensure_ascii=False)
                          for i, (ins, t, lab) in enumerate(zip(harmless, texts, s["rows"])))
                + "\n")
            print(f"  gen @ m={m_gen:.2f}: refusal {s['refusal_rate_strict']:.3f} "
                  f"compliance {s['compliance_rate_strict']:.3f} "
                  f"degenerate {s['degenerate_rate']:.3f}")
        results[name] = entry

    print(f"\n{'arm':22s} {'L@pos':>16s} {'b':>6s} {'1/b':>6s} {'onset m':>8s} "
          f"{'alpha':>8s} {'gen ref':>8s} {'gen deg':>8s}")
    for name, r in results.items():
        g = r.get("gen_at_onset", {})
        cell = f"L{r['layer']}@{r['position']}"
        om = f"{r['onset_multiple']:.2f}" if r["onset_multiple"] is not None else "none"
        oa = f"{r['onset_alpha']:.0f}" if r["onset_alpha"] is not None else "-"
        print(f"{name:22s} {cell:>16s} "
              f"{r['perp_fraction_b']:6.3f} {r['predicted_onset_multiple']:6.2f} "
              f"{om:>8s} {oa:>8s} "
              f"{g.get('refusal_rate_strict', float('nan')):8.3f} "
              f"{g.get('degenerate_rate', float('nan')):8.3f}")

    dump(exp, args.model, sub, "dose.json", {
        "model": args.model, "subset": subset, "dataset": dataset, "n_val": len(pps),
        "val_sources": source_counts(rows),
        "multiples": args.multiples, "onset_threshold": args.onset_threshold,
        "steer_position": "all", "readout": "refusal_metric@first_response_token",
        "clean": {"refusal_metric": float(clean_metric.mean()),
                  "frac_refusal": induced_fraction(clean_metric)},
        "arms": results,
    })


if __name__ == "__main__":
    main()
