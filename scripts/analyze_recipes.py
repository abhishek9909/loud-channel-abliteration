"""Aggregate the recipe pipeline across models into the two tables that decide it.

CPU-only, JSON-only (no torch, no model) — run it anywhere the artifacts are.

Table 1 (the claim): per model x arm, refusal rate on held-out harmful prompts,
the drop vs clean, the degenerate rate, the selection KL, and MMLU. A recipe
"works" when refusal falls substantially AND degeneracy stays at clean levels
AND KL < 0.1 AND MMLU is intact. The degenerate column is what stops a
destroyed model from being scored as a successful bypass.

SELECTION prints both the induce feasibility rule in force (frac, the default,
or Arditi's mean) and, for any recipe whose feasibility flips between them,
what the other rule found. TABLE 1 shows `random` and `random_perp` (the null
orthogonal to the nuisance set — they coincide off Gemma) and any `extra_*`
exploratory arm.

Table 2 (the control): per model, how much each recipe moved the vector
relative to raw, and the Stage-D concentration diagnostics. The correction is
only presentable as a measurement correction — rather than a Gemma-specific
patch — if R1/R3 are near-no-ops wherever the raw estimator was already fine.
The `cos(r0,r1)` column is prediction 5's geometric half (min over layers
< 0.8L); R2's cosine is printed but is NOT a no-op criterion (it rotates on
every model by construction).

Table 3 (the dose, prediction 4): per model x arm from 26_recipe_dose —
b = perp fraction of the unit vector, its predicted onset multiple 1/b, the
measured onset multiple of perp_norm, and what the generation at that dose
looked like. Present only where dose.json exists.

    python scripts/analyze_recipes.py
    python scripts/analyze_recipes.py --artifacts artifacts --sub recipes
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAN = float("nan")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sub", default="recipes")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="write the merged JSON here")
    args = ap.parse_args()

    root = (REPO / args.artifacts) if not Path(args.artifacts).is_absolute() \
        else Path(args.artifacts)
    models = args.models or sorted(
        d.name for d in root.iterdir() if (d / args.sub / "confirm.json").exists())
    if not models:
        raise SystemExit(f"no models with {args.sub}/confirm.json under {root}")

    merged = {}
    print("=" * 104)
    print("TABLE 1 — does ablating the selected direction drop refusal, without "
          "destroying the model?")
    print("=" * 104)
    print(f"{'model':14s} {'arm':20s} {'refusal':>8s} {'drop':>7s} {'compl':>7s} "
          f"{'degen':>7s} {'hless-deg':>9s} {'mmlu':>6s} {'kl':>8s}")
    for m in models:
        d = args.sub
        conf = _load(root / m / d / "confirm.json")
        sel = _load(root / m / d / "selection.json")
        diag = _load(root / m / d / "diagnostics.json")
        dose = _load(root / m / d / "dose.json")
        merged[m] = {"confirm": conf, "selection": sel, "diagnostics": diag,
                     "dose": dose}
        if not conf:
            continue
        src = conf.get("test_sources", {}).get("harmful")
        if src:
            print(f"{m:14s} test set: n={conf['n_test']} subset={conf.get('subset', 'head')} "
                  f"{src}")
        arms = conf["arms"]
        base = arms.get("clean", {}).get("harmful", {}).get("refusal_rate_strict", NAN)
        for name, r in arms.items():
            kl = (r.get("cell") or {}).get("kl")
            h, b = r["harmful"], r["harmless"]
            print(f"{m:14s} {name:20s} {h['refusal_rate_strict']:8.3f} "
                  f"{h['refusal_rate_strict'] - base:+7.3f} "
                  f"{h['compliance_rate_strict']:7.3f} {h['degenerate_rate']:7.3f} "
                  f"{b['degenerate_rate']:9.3f} "
                  f"{r['capability'].get('mmlu_acc', NAN):6.3f} "
                  f"{(f'{kl:.4f}' if kl is not None else '-'):>8s}")
        print("-" * 104)

    print()
    print("=" * 104)
    print("TABLE 2 — concentration of the raw estimator, and how far each recipe "
          "had to move it")
    print("=" * 104)
    print(f"{'model':14s} {'position':16s} {'nuis':>5s} {'r0 top1':>8s} "
          f"{'r0 top8':>8s} {'rms gain':>9s} {'r3 ok':>6s} {'cos(r0,r1)':>10s} "
          f"{'cos(r0,r3)':>10s} {'cos(r0,r2)*':>11s}")
    for m in models:
        diag = merged[m].get("diagnostics")
        if not diag:
            continue
        n_layers = diag.get("n_layers") or len(
            next(iter(diag["by_position"].values()))["energy"]["r0_raw"]["top1_share"])
        keep = int(0.8 * n_layers)          # the selectable layers only
        for pos, dd in diag["by_position"].items():
            e = dd["energy"]["r0_raw"]
            t1, t8 = e["top1_share"], e["topk_share"]
            cos = dd.get("cos_to_raw", {})

            def _min(r):
                v = cos.get(r)
                return f"{min(v[:keep]):10.3f}" if v else f"{'-':>10s}"

            print(f"{m:14s} {pos:16s} {dd['n_nuisance_channels_total']:5d} "
                  f"{max(t1):8.3f} {max(t8):8.3f} {max(dd['norms']['gain']):9.1f} "
                  f"{str(dd.get('winsorize_headroom', {}).get('effective', '?')):>6s} "
                  f"{_min('r1_masked')} {_min('r3_winsorized')} {_min('r2_standardized'):>11s}")
        src = diag.get("train_sources", {}).get("harmful")
        if src:
            print(f"{m:14s} train: subset={diag.get('subset', 'head')} {src} | "
                  f"val: {diag.get('val_sources', {}).get('harmful')}")
        print("-" * 104)
    print("* R2's cosine is informational only — standardisation rotates the vector on "
          "every model; its no-op test is behavioural (same cell, same outcome).")

    if any("nominal_summary" in dd for m in models
           for dd in ((merged[m].get("diagnostics") or {}).get("by_position") or {}).values()):
        print()
        print("=" * 104)
        print("TABLE 2c — what the class gap on the top-sigma coordinate is (nominal-error "
              "model): medians over layers")
        print("=" * 104)
        print(f"{'model':14s} {'position':16s} {'c*':>5s} {'rho':>6s} {'E':>6s} "
              f"{'share':>6s} {'pred':>6s} {'noise':>6s} {'null':>6s} {'ncos':>5s} "
              f"{'flip':>5s} | {'@max L':>6s} {'share':>6s} {'t raw':>6s} {'|len':>6s} "
              f"{'|len+p':>6s}")
        for m in models:
            diag = merged[m].get("diagnostics")
            if not diag:
                continue
            cov = diag.get("covariates")
            if cov:
                print(f"{m:14s} covariates pos/neg: " + " | ".join(
                    f"{k} {v['mean_pos']:.2f}/{v['mean_neg']:.2f} (d={v['delta_std']:+.2f})"
                    for k, v in cov.items()))
            for pos, dd in diag["by_position"].items():
                ns = dd.get("nominal_summary")
                if not ns:
                    continue
                am = ns["at_max"]
                print(f"{m:14s} {pos:16s} {ns['channel_mode']:5d} {ns['rho']:6.0f} "
                      f"{ns['E_rest']:6.0f} {ns['measured_share']:6.2f} "
                      f"{ns['predicted_share']:6.2f} {ns['noise_only_share']:6.2f} "
                      f"{ns['null_share']:6.2f} {ns['null_cos']:5.2f} {ns['signflip_p']:5.2f} | "
                      f"{ns['max_share_layer']:6d} {am['measured_share']:6.2f} "
                      f"{am['welch_t']:+6.1f} {am['t_adj_words']:+6.1f} "
                      f"{am['t_adj_words_punct']:+6.1f}")
            print("-" * 104)
        print("share = c*'s energy share of the raw vector; pred = rho^2 kappa^2/(rho^2 kappa^2 + E); "
              "noise = the same with kappa^2 = 2/n;\nnull = c*'s share of a split-half "
              "HARMLESS-only vector (no class at all), ncos = |cos| of that vector with the "
              "raw one;\nflip = bootstrap P(sign change of the gap); t raw -> |len -> |len+p: "
              "the gap's t before and after holding covariates fixed.")

    if any("ablations" in dd for m in models
           for dd in ((merged[m].get("diagnostics") or {}).get("by_position") or {}).values()):
        print()
        print("=" * 104)
        print("TABLE 2d — estimator ablations at one layer: is it the coordinate's scale, "
              "or a special property? (D2c)")
        print("=" * 104)
        print(f"{'model':14s} {'position':14s} {'L':>3s} {'c*':>5s} | {'att1/10':>7s} {'share':>6s} "
              f"{'auroc':>11s} | {'amp':>5s} {'own':>5s} {'x100':>6s} {'auroc':>6s} | "
              f"{'noise':>6s} {'rob':>5s}")
        for m in models:
            diag = merged[m].get("diagnostics")
            if not diag:
                continue
            for pos, dd in diag["by_position"].items():
                for li, abl in (dd.get("ablations") or {}).items():
                    att = {r["gain"]: r for r in abl["dial_attenuate"]["rows"]}
                    amp = {r["gain"]: r for r in abl["dial_amplify"]["rows"]}
                    pick, no = abl["amplify_pick"], abl["noise"]
                    rob_ok = all(r["cos_with_raw"] > 0.97 for r in no["robustness"])
                    print(f"{m:14s} {pos:14s} {int(li):3d} {abl['channel']:5d} | "
                          f"{att[1.0]['share']:7.2f} {att[0.1]['share']:6.2f} "
                          f"{att[1.0]['auroc_heldout']:5.2f}->{att[0.1]['auroc_heldout']:4.2f} | "
                          f"{pick['channel']:5d} {pick['auroc_single_heldout']:5.2f} "
                          f"{amp[100.0]['share']:6.2f} {amp[100.0]['auroc_heldout']:6.2f} | "
                          f"{no['noise_only']['share_mean']:6.2f} {('ok' if rob_ok else 'MOVED'):>5s}")
            print("-" * 104)
        print("att1/10 = c*'s share before and after attenuating it 10x, with the raw projection's "
              "held-out AUROC; amp = the most\nlength-correlated ORDINARY coordinate, its own AUROC, "
              "and share/AUROC after amplifying it 100x; noise = mean share a\nrho=200 pure-noise "
              "coordinate takes; rob = nominal noise on ordinary coordinates left the vector alone.")
        print()
        print(f"{'model':14s} {'position':14s} {'L':>3s} | {'top100/E':>8s} {'ev rho':>6s} "
              f"{'c* rank':>8s} {'raw@ev':>6s} {'ovl':>4s} {'cos e*':>6s} | "
              f"{'AUROC dom/R2/rawP/stdP':>24s} | {'c* std':>7s} {'c* raw':>7s} "
              f"{'cos std':>7s} {'cos raw':>7s} {'d·w':>5s}")
        for m in models:
            diag = merged[m].get("diagnostics")
            if not diag:
                continue
            for pos, dd in diag["by_position"].items():
                for li, abl in (dd.get("ablations") or {}).items():
                    ev, pr = abl["evidence"], abl["probes"]
                    a = pr["auroc_heldout"]
                    print(f"{m:14s} {pos:14s} {int(li):3d} | "
                          f"{ev['cum_share_top_k'].get('100', NAN):8.2f} "
                          f"{ev['evidence_top_k']['rho_median']:6.1f} "
                          f"{ev['cstar']['rank_by_dprime2']:8d} "
                          f"{ev['raw_vs_evidence']['energy_share_on_evidence_top_k']:6.2f} "
                          f"{ev['raw_vs_evidence']['overlap_top_k']:4d} "
                          f"{ev['raw_vs_evidence']['cos_with_e_cstar']:+6.2f} | "
                          f"{a['raw_diff_of_means']:5.3f}/{a['r2_standardized']:5.3f}/"
                          f"{a['raw_feature_probe']:5.3f}/{a['standardized_probe']:5.3f} | "
                          f"{pr['standardized_probe']['cstar_rank']:7d} {pr['raw_probe']['cstar_rank']:7d} "
                          f"{pr['standardized_probe']['cos_with_dprime']:7.2f} "
                          f"{pr['raw_probe']['cos_with_dprime']:7.2f} "
                          f"{pr['raw_probe']['cos_raw_d_with_raw_w']:5.2f}")
            print("-" * 104)
        print("top100/E = share of the layer's evidence (sum d'^2) in its top-100 coordinates; ev rho = "
              "median rho of those; c* rank = c*\nby d'^2; raw@ev = share of the raw vector's energy on "
              "the evidence top-100, ovl = overlap of the raw top-100 with it;\nc* std / c* raw = c*'s "
              "rank by effect^2 for the standardised and (per-sigma) raw probe; cos = cos(effect, d');\n"
              "d·w = cos(raw d, raw-probe w). Standardise before interpreting any probe direction.")

    if any((merged[m].get("diagnostics") or {}).get("norm_gains") for m in models):
        print()
        print("=" * 104)
        print("TABLE 2b — provenance: are the activation-identified nuisance channels the "
              "ones the RMSNorm gains single out?")
        print("=" * 104)
        print(f"{'model':14s} {'ch':>5s} {'#cells':>6s} {'writer rank1':>12s} {'sign':>5s} "
              f"{'reader|g|/med':>13s} {'reads as':>10s} {'final rank':>10s} {'logits':>7s}")
        for m in models:
            ng = (merged[m].get("diagnostics") or {}).get("norm_gains")
            if not ng:
                continue
            writers = ng.get("writer_norms") or []
            top = ng.get("top_by_writer_gain")
            head = (f"writers {writers}" if writers
                    else "pre-norm family: no writer gains (control)")
            print(f"{m:14s} channels from: {ng.get('channels_source')} | {head}")
            if top:
                overlap = sorted(set(top["channels"]) & set(ng.get("channels", [])))
                pairs = ", ".join(f"{c}:{g:.1f}" for c, g in
                                  zip(top["channels"], top["mean_abs_gain"]))
                print(f"{'':14s} top-{len(top['channels'])} by mean |writer gain|: {pairs} "
                      f"(median channel {top['median_channel']:.2f}) | overlap with "
                      f"nuisance set: {overlap or 'none'}")
            counts = ng.get("channel_layer_counts", {})
            for c in ng.get("channels", []):
                v = ng["verdict"].get(str(c), {})
                wr = v.get("writer_frac_rank1")
                sign = v.get("writer_sign_consistent")
                rr = v.get("reader_abs_over_median")
                reads = ("hidden" if v.get("reader_gain_hidden") else
                         "bias-in" if v.get("reader_gain_bias_input") else "ordinary")
                print(f"{'':14s} {c:5d} {counts.get(str(c), 0):6d} "
                      f"{(f'{wr:.2f}' if wr is not None else '-'):>12s} "
                      f"{('ok' if sign else 'MIXED' if sign is not None else '-'):>5s} "
                      f"{(f'{rr:.2f}' if rr is not None else '-'):>13s} {reads:>10s} "
                      f"{str(v.get('final_norm_rank', '-')):>10s} "
                      f"{('yes' if v.get('writes_to_logits') else 'no' if 'writes_to_logits' in v else '-'):>7s}")
            print("-" * 104)
        print("writer rank1 = share of layers where the channel has the largest |post-FFN gain|; "
              "reader|g|/med < 0.5 = hidden from\ncontent reads (acts only via the RMS "
              "denominator), > 2 = a large constant input; 'logits' = final-norm gain in the top 8.")

    print()
    print("=" * 104)
    print("SELECTION — how many candidates survived, and why the rest did not")
    print("=" * 104)
    for m in models:
        sel = merged[m].get("selection")
        if not sel:
            continue
        diag = merged[m].get("diagnostics") or {}
        cfg = sel.get("config", {})
        if "induce_rule" in cfg:
            print(f"{m:14s} induce_rule={cfg['induce_rule']} (frac_min "
                  f"{cfg.get('induce_frac_min', 0.5)}) objective={cfg.get('objective', 'mean')} "
                  f"common_dose={cfg.get('common_dose')}  "
                  f"[feasible = this rule; arditi = mean logit > 0]")
        for recipe, r in sel["by_recipe"].items():
            s = r["selected"]
            cell = (f"L{s['layer']}@{s['position']} kl={s['kl']:.4f} "
                    f"bypass={s['bypass']:+.3f} induce_frac={s.get('induce_frac', float('nan')):.2f}"
                    f"@{s.get('induce_passed_at') or '-'}" if s else "NONE FEASIBLE")
            if s and recipe != "r0_raw":
                c = (diag.get("by_position", {}).get(s["position"], {})
                     .get("cos_to_raw", {}).get(recipe))
                if c:
                    cell += f" cos_raw@cell={c[s['layer']]:.3f}"
            # if this rule found nothing but the OTHER did, say so — the R1
            # feasibility swing between the two rules is a finding in itself
            alt = ""
            na = r.get("n_feasible_arditi")
            if na is not None and (r["n_feasible"] == 0) != (na == 0):
                sa = r.get("selected_arditi")
                alt = (f"  [arditi rule: {na} feasible"
                       + (f", L{sa['layer']}@{sa['position']}" if sa else "") + "]")
            print(f"{m:14s} {recipe:18s} {r['n_feasible']:4d}/{r['n_candidates']:4d} "
                  f"feasible (arditi {r.get('n_feasible_arditi', '?'):>4}) | "
                  f"min kl {r['min_kl']:.4f} | {r['reject_reasons']} | {cell}{alt}")
        print("-" * 104)

    if any(merged[m].get("dose") for m in models):
        print()
        print("=" * 104)
        print("TABLE 3 — steering onset in units of perp_norm (prediction 4): "
              "cleaned ~1x, contaminated ~1/b")
        print("=" * 104)
        print(f"{'model':14s} {'arm':22s} {'cell':>16s} {'b':>6s} {'1/b':>6s} "
              f"{'onset m':>8s} {'alpha':>8s} {'frac@1x':>8s} {'kl@onset':>9s} "
              f"{'gen ref':>8s} {'gen deg':>8s}")
        for m in models:
            dose = merged[m].get("dose")
            if not dose:
                continue
            for name, r in dose["arms"].items():
                sw = {row["multiple"]: row for row in r["sweep"]}
                at1 = sw.get(1.0, {}).get("frac_refusal", NAN)
                om = r["onset_multiple"]
                kl_on = sw.get(om, {}).get("kl", NAN) if om is not None else NAN
                g = r.get("gen_at_onset", {})
                cell = f"L{r['layer']}@{r['position']}"
                om_s = f"{om:.2f}" if om is not None else "none"
                oa_s = f"{r['onset_alpha']:.0f}" if om is not None else "-"
                print(f"{m:14s} {name:22s} {cell:>16s} "
                      f"{r['perp_fraction_b']:6.3f} {r['predicted_onset_multiple']:6.2f} "
                      f"{om_s:>8s} {oa_s:>8s} {at1:8.3f} {kl_on:9.4f} "
                      f"{g.get('refusal_rate_strict', NAN):8.3f} "
                      f"{g.get('degenerate_rate', NAN):8.3f}")
            print("-" * 104)

    if args.out:
        Path(args.out).write_text(json.dumps(merged, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
