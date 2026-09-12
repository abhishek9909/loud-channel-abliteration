"""E1 + E2 — what the massive channel does causally, separated from the estimator.

Two questions the recipe pipeline cannot answer, because both are about ONE
coordinate rather than a direction. Both are run on the SAME held-out splits
and with the same three-way scoring as 25_recipe_confirm, so the numbers drop
straight into the same table.

E1 — which of the two destructive routes does all-layer projection ablation
     actually collapse Gemma through? Writing the estimated direction as
     d = a*e_c* + b*u, ablation at every layer does

       route 2   scales x_c* by (1 - a^2), raising every downstream RMSNorm
                 gain (a ~ 0.92 at the r0 cell, so ~85% of the channel goes)
       route 1   injects -a*b*x_c* along u: fixed sign, every token, every
                 layer, several times the norm of the ordinary residual

     The existing random-direction null (a ~ 1/sqrt(D)) shows route 1 alone is
     enough to hurt; it says nothing about the split for the real vector. Arms:

       e1_cstar_zero    scale x_c* by 0      route 2 at full strength, no leak
       e1_cstar_route2  scale x_c* by 1-a^2  route 2 at the real vector's strength
       e1_leak_only     inject -a*b*x_c*u    route 1 alone, no scaling
       e1_full_r0       ablate the r0 row    both routes (the original operation,
                        measured with the original instrument, as the reference)

     Reading it: if e1_cstar_zero is clean, the channel's magnitude is not what
     the model needs and the collapse is the leak -> the headline stays "the
     estimator loaded the sink". If e1_cstar_zero collapses too, then projection
     ablation is hostile to Gemma's norm geometry for reasons independent of any
     estimator, which is a stronger and more general claim.

E2 — at the post-instruction positions x_c* carries a large class correlate
     (d' ~ 3.2 at L20@t_post_inst-4, t 23 after covariate control). Reader gain
     0 means no sublayer reads c* AS A COORDINATE; it does NOT mean the value
     is inert, because c* carries most of the residual's squared norm and so
     sets the gain on everything the next block reads. So: move x_c* by the
     measured class gap at the prompt positions and see whether behaviour
     follows.

       e2_to_harmful    harmless prompts, x_c* += delta  (toward the harmful
                        conditional mean) -> does refusal rise?
       e2_to_harmless   harmful prompts,  x_c* -= delta  -> does refusal fall?
       e2_gain_*        the SAME change in residual RMS spread over every
                        coordinate (a global gain). Separates "this channel"
                        from "any equivalent norm change" -- without it a
                        positive result is uninterpretable.
       e2_ordinary_*    the same manoeuvre on the layer's best ORDINARY
                        evidence coordinate, at its own class gap: the positive
                        control that says a prompt-side shift of a coordinate
                        that does carry evidence moves behaviour at all.

     A null result here is a result: it turns "causally inert" from an
     inference off a zero reader gain into a measurement.

    python experiments/05_channel_causal.py --model gemma3-12b --stage e1
    python experiments/05_channel_causal.py --model gemma3-12b --stage e2
    python experiments/05_channel_causal.py --model gemma3-12b --stage e2 --no-gen
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.channel_interventions import (
    generate_with_ops,
    logits_with_ops,
    mmlu_accuracy_with_ops,
    perplexity_with_ops,
)
from loudchannel.channel_ops import (
    decompose_row,
    global_scale,
    inject_leak,
    scale_channel,
    shift_channel,
    shift_vector,
)
from loudchannel.config import artifacts_dir
from loudchannel.data.splits import SUBSETS, load_split, source_counts, subset_rows
from loudchannel.degeneracy import GATE, summarize
from loudchannel.positions import positions_for
from loudchannel.recipes import extract_multi
from loudchannel.selection import broadcast_direction, induced_fraction, refusal_metric

SUB = "recipes"


def _cstar(diag: dict, position: str, layers: list[int], override: int | None) -> int:
    if override is not None:
        return override
    chans = diag[position]["nominal"]["channel"]
    return Counter(int(chans[li]) for li in layers).most_common(1)[0][0]


def _gen_and_score(model, instrs, *, ops, position, pps, max_new_tokens, bs):
    prompts = [model.render(i) for i in instrs]
    texts = generate_with_ops(model, prompts, ops=ops, position=position, pps=pps,
                              max_new_tokens=max_new_tokens, batch_size=bs)
    return texts, summarize(texts)


def _write_transcripts(adir, name, split, rows, texts, s):
    (adir / f"transcripts_{name}_{split}.jsonl").write_text("\n".join(
        json.dumps({"i": i, "arm": name, "split": split, "source": r.get("source"),
                    "instruction": r["instruction"], "transcript": t,
                    **{k: v for k, v in lab.items() if k != "rows"}},
                   ensure_ascii=False)
        for i, (r, t, lab) in enumerate(zip(rows, texts, s["rows"]))) + "\n")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--stage", default="e1", choices=("e1", "e2", "both"))
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--subset", default="stratified", choices=SUBSETS)
    ap.add_argument("--channel", type=int, default=None,
                    help="override c* (default: the modal top-sigma channel)")
    ap.add_argument("--ref-recipe", default="r0_raw",
                    help="whose direction defines a, b, u for the E1 arms")
    ap.add_argument("--ref-cell", default=None,
                    help="layer:position for the reference row (default: the "
                         "cell r1_masked selected, i.e. the dose stage's "
                         "r0_raw@r1cell contrast)")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--skip-capability", action="store_true")
    # E2
    ap.add_argument("--e2-position", default="t_post_inst-4")
    ap.add_argument("--e2-layers", nargs="+", type=int, default=list(range(15, 26)))
    ap.add_argument("--e2-gap", default="adj", choices=("adj", "raw"),
                    help="class gap to move by: covariate-adjusted (default) or raw")
    ap.add_argument("--e2-scale", type=float, default=1.0,
                    help="multiple of the class gap to apply (2.0 = overshoot)")
    ap.add_argument("--e2-positive-recipe", default="r1_masked",
                    help="recipe whose selected direction provides the E2 positive "
                         "control: a DIFFUSE shift at the same position and layers, "
                         "RMS-matched to the c* shift. Without it a null on c* cannot "
                         "be told from an instrument with no sensitivity here. "
                         "'none' disables the arm.")
    ap.add_argument("--no-gen", action="store_true",
                    help="E2: first-token readout only, no generation pass")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)
    n_layers, d_model = model.n_layers, None
    all_layers = list(range(n_layers))

    diag = json.loads((adir / "diagnostics.json").read_text())["by_position"]
    sel = json.loads((adir / "selection.json").read_text())["by_recipe"]
    blob = torch.load(adir / "directions.pt", map_location="cpu", weights_only=False)

    # reference cell for the E1 decomposition
    if args.ref_cell:
        l_ref, p_ref = args.ref_cell.split(":", 1)
        l_ref = int(l_ref)
    else:
        s = (sel.get("r1_masked") or {}).get("selected")
        assert s, "no r1_masked cell; pass --ref-cell layer:position"
        l_ref, p_ref = s["layer"], s["position"]
    ref_row = blob["directions"][args.ref_recipe][p_ref][l_ref].float()
    d_model = ref_row.numel()
    cstar = _cstar(diag, p_ref, all_layers, args.channel)
    dec = decompose_row(ref_row, cstar)
    print(f"c* = {cstar} | reference {args.ref_recipe} @ L{l_ref}@{p_ref} "
          f"| a {dec['a']:+.4f}  b {dec['b']:.4f}  "
          f"route2 gain (1-a^2) {dec['route2_gain']:.4f}  leak coef (-ab) {dec['leak_coef']:+.4f}")

    n_test = args.limit or args.n_test
    harmful_rows = subset_rows(load_split("harmful_eval"), args.subset)[:n_test]
    harmless_rows = subset_rows(load_split("harmless_eval"), args.subset)[:n_test]
    harmful, harmless = instructions(harmful_rows), instructions(harmless_rows)
    print(f"test {len(harmful)} harmful / {len(harmless)} harmless "
          f"| subset={args.subset} {source_counts(harmful_rows)}")

    results: dict[str, dict] = {}
    meta = {"model": args.model, "channel": cstar, "subset": args.subset,
            "n_test": len(harmful), "decomposition":
                {k: v for k, v in dec.items() if k != "u"},
            "reference_cell": {"recipe": args.ref_recipe, "layer": l_ref,
                               "position": p_ref},
            "gate": {k: getattr(GATE, k) for k in GATE.__dataclass_fields__}}

    # ---------------- E1 ----------------------------------------------------
    if args.stage in ("e1", "both"):
        arms: dict[str, dict] = {
            "clean": {"ops": None},
            "e1_cstar_zero": {"ops": scale_channel(
                cstar, 0.0, all_layers, d_model, label="e1_cstar_zero",
                meta={"route": "2 (full): the channel removed entirely"})},
            "e1_cstar_route2": {"ops": scale_channel(
                cstar, dec["route2_gain"], all_layers, d_model, label="e1_cstar_route2",
                meta={"route": "2 (at r0 strength): x_c* scaled by 1-a^2"})},
            "e1_leak_only": {"ops": inject_leak(
                cstar, dec["leak_coef"], dec["u"], all_layers, label="e1_leak_only",
                meta={"route": "1: -a*b*x_c* along u, no scaling of c*"})},
            "e1_full_r0": {"direction": broadcast_direction(
                ref_row, n_layers, kind=f"e1ref|{args.ref_recipe}",
                position=p_ref, model_name=args.model),
                "note": "both routes: the original operation, original instrument"},
        }
        if args.arms:
            arms = {k: v for k, v in arms.items() if k in args.arms}
        for name, arm in arms.items():
            print(f"\n=== E1 arm {name} ===")
            if "direction" in arm:      # reference arm: the untouched code path
                from loudchannel.interventions import generate_with_intervention
                from loudchannel.pipelines import perplexity
                import importlib
                _h2c = importlib.import_module("04_h2c_capability")

                def gen(instrs):
                    texts = generate_with_intervention(
                        model, [model.render(i) for i in instrs],
                        direction=arm["direction"], layers=all_layers, mode="ablate",
                        position="all", max_new_tokens=args.max_new_tokens,
                        batch_size=args.gen_batch_size)
                    return texts, summarize(texts)
                t_h, s_h = gen(harmful)
                t_b, s_b = gen(harmless)
                cap = {}
                if not args.skip_capability:
                    correct = _h2c.mmlu_accuracy(
                        model, load_split("mmlu_slice")[:exp.get("mmlu_slice", 500)],
                        direction=arm["direction"], layers=all_layers, mode="ablate",
                        batch_size=exp["batch_size"])
                    cap["mmlu_acc"] = float(sum(correct) / max(len(correct), 1))
                    cap["nll"] = perplexity(model, harmless[:50],
                                            direction=arm["direction"], layers=all_layers,
                                            mode="ablate", batch_size=args.gen_batch_size)
            else:
                ops = arm["ops"]
                t_h, s_h = _gen_and_score(model, harmful, ops=ops, position="all",
                                          pps=None, max_new_tokens=args.max_new_tokens,
                                          bs=args.gen_batch_size)
                t_b, s_b = _gen_and_score(model, harmless, ops=ops, position="all",
                                          pps=None, max_new_tokens=args.max_new_tokens,
                                          bs=args.gen_batch_size)
                cap = {}
                if not args.skip_capability:
                    correct = mmlu_accuracy_with_ops(
                        model, load_split("mmlu_slice")[:exp.get("mmlu_slice", 500)],
                        ops=ops, batch_size=exp["batch_size"])
                    cap["mmlu_acc"] = float(sum(correct) / max(len(correct), 1))
                    cap["nll"] = perplexity_with_ops(model, harmless[:50], ops=ops,
                                                     batch_size=args.gen_batch_size)
            _write_transcripts(adir, name, "harmful_eval", harmful_rows, t_h, s_h)
            _write_transcripts(adir, name, "harmless_eval", harmless_rows, t_b, s_b)
            results[name] = {
                "stage": "e1",
                "ops": arm["ops"].describe() if arm.get("ops") is not None else
                       ({"note": arm.get("note")} if "direction" in arm else {"note": "no intervention"}),
                "harmful": {k: v for k, v in s_h.items() if k != "rows"},
                "harmless": {k: v for k, v in s_b.items() if k != "rows"},
                "capability": cap,
            }
            print(f"  harmful : refusal {s_h['refusal_rate_strict']:.3f} "
                  f"compliance {s_h['compliance_rate_strict']:.3f} "
                  f"degenerate {s_h['degenerate_rate']:.3f}")
            print(f"  harmless: refusal {s_b['refusal_rate_strict']:.3f} "
                  f"degenerate {s_b['degenerate_rate']:.3f}  {cap}")

        base = results.get("clean", {}).get("harmful", {}).get("refusal_rate_strict")
        if base is not None:
            print(f"\n{'E1 arm':18s} {'refusal':>8s} {'drop':>7s} {'degen':>7s} "
                  f"{'h-less degen':>13s} {'mmlu':>6s} {'nll':>7s}")
            for name, r in results.items():
                if r["stage"] != "e1":
                    continue
                print(f"{name:18s} {r['harmful']['refusal_rate_strict']:8.3f} "
                      f"{r['harmful']['refusal_rate_strict'] - base:+7.3f} "
                      f"{r['harmful']['degenerate_rate']:7.3f} "
                      f"{r['harmless']['degenerate_rate']:13.3f} "
                      f"{r['capability'].get('mmlu_acc', float('nan')):6.3f} "
                      f"{r['capability'].get('nll', float('nan')):7.3f}")

    # ---------------- E2 ----------------------------------------------------
    if args.stage in ("e2", "both"):
        pos = args.e2_position
        assert pos in diag, f"{pos} not in the extracted grid {list(diag)}"
        nom = diag[pos]["nominal"]
        key = "gap_adj_words_punct" if args.e2_gap == "adj" else "raw_gap"
        deltas = {li: float(nom[key][li]) * args.e2_scale for li in args.e2_layers}
        # the layer's best ORDINARY evidence coordinate (positive control), from
        # whichever D2c layer is closest to the band
        abl = diag[pos].get("ablations", {})
        ref_l = str(min((int(k) for k in abl), key=lambda k: abs(k - args.e2_layers[0]))) \
            if abl else None
        ord_ch = abl[ref_l]["evidence"]["best_channel"]["channel"] if ref_l else None

        pps_harmful = positions_for(model, harmful, seed=exp["seed"])
        pps_harmless = positions_for(model, harmless, seed=exp["seed"])

        # exact RMS-matched global gain: measure ||x|| and x_c* at this position
        acts_h = extract_multi(model, harmful, (pos,), batch_size=exp["batch_size"])[pos]
        acts_b = extract_multi(model, harmless, (pos,), batch_size=exp["batch_size"])[pos]
        gains_up, gains_dn, ord_deltas = {}, {}, {}
        for li in args.e2_layers:
            xb, xh = acts_b[:, li].float(), acts_h[:, li].float()
            d = deltas[li]
            shifted_up = xb.clone(); shifted_up[:, cstar] += d
            shifted_dn = xh.clone(); shifted_dn[:, cstar] -= d
            gains_up[li] = float((shifted_up.norm(dim=-1) / xb.norm(dim=-1).clamp_min(1e-9)).mean())
            gains_dn[li] = float((shifted_dn.norm(dim=-1) / xh.norm(dim=-1).clamp_min(1e-9)).mean())
            if ord_ch is not None:
                ord_deltas[li] = float(xh[:, ord_ch].mean() - xb[:, ord_ch].mean()) * args.e2_scale
        print(f"\nE2 @ {pos}, layers {args.e2_layers[0]}-{args.e2_layers[-1]} "
              f"| gap={args.e2_gap} x{args.e2_scale} | c*={cstar} ordinary ctrl={ord_ch}")
        print("  delta / rms-gain per layer: " + " ".join(
            f"L{li}:{deltas[li]:+.0f}({gains_up[li]:.3f})" for li in args.e2_layers))

        e2: dict[str, dict] = {
            "e2_to_harmful": {"ops": shift_channel(cstar, deltas, d_model,
                                                   label="e2_to_harmful"),
                              "on": "harmless"},
            "e2_to_harmless": {"ops": shift_channel(cstar, {k: -v for k, v in deltas.items()},
                                                    d_model, label="e2_to_harmless"),
                               "on": "harmful"},
            "e2_gain_to_harmful": {"ops": global_scale(gains_up, d_model,
                                                       label="e2_gain_to_harmful"),
                                   "on": "harmless"},
            "e2_gain_to_harmless": {"ops": global_scale(gains_dn, d_model,
                                                        label="e2_gain_to_harmless"),
                                    "on": "harmful"},
        }
        # positive control: the same manoeuvre with the DIFFUSE direction, scaled
        # per layer so it changes the residual's RMS by the same factor the c*
        # shift does. This is the arm that says whether a prompt-side edit at
        # this position can move refusal at all.
        pos_sel = (sel.get(args.e2_positive_recipe) or {}).get("selected") \
            if args.e2_positive_recipe != "none" else None
        if pos_sel:
            prow = blob["directions"][args.e2_positive_recipe][pos_sel["position"]][
                pos_sel["layer"]].float()
            pu = prow / prow.norm().clamp_min(1e-12)
            up_v, dn_v = {}, {}
            for li in args.e2_layers:
                xb, xh = acts_b[:, li].float(), acts_h[:, li].float()
                # alpha such that mean ||x + alpha*u|| / ||x|| matches gains_up
                tgt_u = (gains_up[li] - 1.0) * float(xb.norm(dim=-1).mean())
                tgt_d = (1.0 - gains_dn[li]) * float(xh.norm(dim=-1).mean())
                up_v[li] = pu * tgt_u
                dn_v[li] = -pu * tgt_d
            e2["e2_positive_to_harmful"] = {
                "ops": shift_vector(up_v, label="e2_positive_to_harmful",
                                    meta={"recipe": args.e2_positive_recipe,
                                          "cell": pos_sel["layer"],
                                          "cell_position": pos_sel["position"]}),
                "on": "harmless"}
            e2["e2_positive_to_harmless"] = {
                "ops": shift_vector(dn_v, label="e2_positive_to_harmless",
                                    meta={"recipe": args.e2_positive_recipe}),
                "on": "harmful"}
            print(f"  positive control: {args.e2_positive_recipe} @ "
                  f"L{pos_sel['layer']}@{pos_sel['position']}, RMS-matched "
                  f"(alpha L{args.e2_layers[0]} = {float(up_v[args.e2_layers[0]].norm()):.0f})")
        if ord_ch is not None:
            e2["e2_ordinary_to_harmful"] = {
                "ops": shift_channel(ord_ch, ord_deltas, d_model,
                                     label="e2_ordinary_to_harmful"), "on": "harmless"}
            e2["e2_ordinary_to_harmless"] = {
                "ops": shift_channel(ord_ch, {k: -v for k, v in ord_deltas.items()},
                                     d_model, label="e2_ordinary_to_harmless"),
                "on": "harmful"}
        if args.arms:
            e2 = {k: v for k, v in e2.items() if k in args.arms}

        # clean readout baseline on both splits
        clean_m = {}
        for split, pps in (("harmful", pps_harmful), ("harmless", pps_harmless)):
            lg = logits_with_ops(model, pps, ops=None, position="all",
                                 batch_size=exp["batch_size"])
            m = refusal_metric(lg, model.tokenizer)
            clean_m[split] = {"refusal_metric": float(m.mean()),
                              "frac_refusal": induced_fraction(m)}
        print(f"  clean readout: harmful {clean_m['harmful']['refusal_metric']:+.3f} "
              f"(frac {clean_m['harmful']['frac_refusal']:.3f}) | harmless "
              f"{clean_m['harmless']['refusal_metric']:+.3f} "
              f"(frac {clean_m['harmless']['frac_refusal']:.3f})")

        for name, arm in e2.items():
            on = arm["on"]
            pps = pps_harmless if on == "harmless" else pps_harmful
            instrs = harmless if on == "harmless" else harmful
            rows_src = harmless_rows if on == "harmless" else harmful_rows
            print(f"\n=== E2 arm {name}  (on {on} prompts, edit at {pos}) ===")
            lg = logits_with_ops(model, pps, ops=arm["ops"], position=pos,
                                 batch_size=exp["batch_size"])
            m = refusal_metric(lg, model.tokenizer)
            entry = {
                "stage": "e2", "on": on, "position": pos,
                "layers": args.e2_layers,
                "ops": arm["ops"].describe(),
                "readout": {"refusal_metric": float(m.mean()),
                            "frac_refusal": induced_fraction(m),
                            "clean_refusal_metric": clean_m[on]["refusal_metric"],
                            "clean_frac_refusal": clean_m[on]["frac_refusal"],
                            "delta_metric": float(m.mean()) - clean_m[on]["refusal_metric"],
                            "delta_frac": induced_fraction(m) - clean_m[on]["frac_refusal"]},
            }
            print(f"  readout: metric {float(m.mean()):+.3f} "
                  f"(clean {clean_m[on]['refusal_metric']:+.3f}, "
                  f"delta {entry['readout']['delta_metric']:+.3f}) | frac "
                  f"{induced_fraction(m):.3f} (clean {clean_m[on]['frac_refusal']:.3f})")
            if not args.no_gen:
                texts, s = _gen_and_score(model, instrs, ops=arm["ops"], position=pos,
                                          pps=pps, max_new_tokens=args.max_new_tokens,
                                          bs=args.gen_batch_size)
                _write_transcripts(adir, name, f"{on}_eval", rows_src, texts, s)
                entry["generation"] = {k: v for k, v in s.items() if k != "rows"}
                print(f"  gen: refusal {s['refusal_rate_strict']:.3f} "
                      f"compliance {s['compliance_rate_strict']:.3f} "
                      f"degenerate {s['degenerate_rate']:.3f}")
            results[name] = entry

        meta["e2"] = {"position": pos, "layers": args.e2_layers, "gap": args.e2_gap,
                      "scale": args.e2_scale, "deltas": {str(k): v for k, v in deltas.items()},
                      "rms_gains_up": {str(k): v for k, v in gains_up.items()},
                      "rms_gains_dn": {str(k): v for k, v in gains_dn.items()},
                      "ordinary_channel": ord_ch, "clean": clean_m}

        print(f"\n{'E2 arm':26s} {'on':>9s} {'metric':>8s} {'d metric':>9s} "
              f"{'frac':>6s} {'gen ref':>8s} {'gen deg':>8s}")
        for name, r in results.items():
            if r["stage"] != "e2":
                continue
            g = r.get("generation", {})
            print(f"{name:26s} {r['on']:>9s} {r['readout']['refusal_metric']:8.3f} "
                  f"{r['readout']['delta_metric']:+9.3f} "
                  f"{r['readout']['frac_refusal']:6.3f} "
                  f"{g.get('refusal_rate_strict', float('nan')):8.3f} "
                  f"{g.get('degenerate_rate', float('nan')):8.3f}")

    dump(exp, args.model, sub, f"channel_causal_{args.stage}.json",
         {**meta, "arms": results})


if __name__ == "__main__":
    main()
