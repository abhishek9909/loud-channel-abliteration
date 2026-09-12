"""Stage D+R — one extraction pass, every candidate direction, all diagnostics.

Builds the full candidate grid for the estimator comparison: |positions| x
|layers| vectors under each of R0..R4, from a SINGLE forward pass over the
extraction split. Everything after extraction is a post-hoc transform of the
same cached activations, so adding a recipe costs nothing.

Also emits the Stage-D diagnostics that motivate the whole exercise:
  D1 energy share      how much of ||d||^2 sits in one coordinate
  D2 evidence vs weight  is the loudest coordinate the most discriminative one
  D2b nominal error    what the class gap on the top-sigma coordinate IS: its
                       rho / kappa / E, the share the nominal-error model
                       predicts vs the share measured, the gap conditional on
                       prompt length and terminal punctuation, the bootstrap
                       band on the share, and the no-class split-half null
                       (loudchannel.nominal; the "link 2" analysis)
  D2c estimator ablations  at one layer per position (the max-share layer, or
                       --dial-layers): the scale dial (attenuate c* / amplify
                       the most length-correlated ordinary coordinate and
                       watch the raw estimator heal / break), noise probes on
                       ordinary coordinates, the evidence profile (how diffuse
                       the class evidence is, whether c* is any of it, how much
                       of the raw vector sits on it) and the four-readout probe
                       comparison with effect-per-sigma geometry
  D3 nuisance set      class-blind identification of massive coordinates
  D5 norm decomposition  the RMSNorm gain change ablation would cause
  D7 norm gains        weight-level provenance of the nuisance set: are these
                       the channels the architecture's post-block RMSNorm gains
                       single out? (loudchannel.norm_gains; the audit's last open
                       item). Written to norm_gains.json + a summary here.
  I4 perp norm         the dose scale once the nuisance coordinates are out

Sample sizes follow Arditi et al.: 128 train (direction estimation) + 32 val
(selection, used by 24_recipe_select). Both come from the frozen *_extract
splits, disjoint from the *_eval splits that 25_recipe_confirm scores on.

Sample COMPOSITION (--subset): the frozen harmful split is stored as 150
AdvBench rows then 150 SORRY-Bench rows, so a plain head slice is AdvBench-only
— which shrinks the prompt-length gap R4 exists to control for from ~7 words
to ~2 and makes the raw vector a different sample from the legacy d_refuse
(300 advbench+sorrybench). Default `stratified` interleaves the sources before
slicing (64+64 train, 16+16 val); `head` reproduces the single-source slice.
The choice is recorded in directions.pt and 24 reads it from there, so the
val rows can never come from a different ordering than the train rows.

    python experiments/01_extract.py --model gemma3-12b
    python experiments/01_extract.py --model llama3-8b --save-acts
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.config import artifacts_dir
from loudchannel.data.datasets import DATASET_NAMES, describe as describe_dataset, load_rows
from loudchannel.data.splits import SUBSETS, load_split, source_counts, subset_rows
from loudchannel.nominal import (
    covariate_summary,
    evidence_profile,
    most_confound_sensitive_ordinary_channel,
    noise_probes,
    nominal_decomposition,
    probe_comparison,
    prompt_covariates,
    scale_dial,
    summarize_nominal,
)
from loudchannel.norm_gains import candidate_channels, format_report, norm_gain_report
from loudchannel.recipes import (
    CANDIDATE_POSITIONS,
    RECIPES,
    build_direction,
    cos_to_raw,
    energy_shares,
    evidence_stats,
    extract_multi,
    length_matched_indices,
    norm_decomposition,
    nuisance_channels,
    perp_norm,
    winsorize_headroom,
)

SUB = "recipes"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--n-train", type=int, default=128,
                    help="prompts per class for direction estimation (Arditi: 128)")
    ap.add_argument("--n-val", type=int, default=32,
                    help="held out here, consumed by 24_recipe_select (Arditi: 32)")
    ap.add_argument("--positions", nargs="+", default=list(CANDIDATE_POSITIONS))
    ap.add_argument("--nuisance-ratio", type=float, default=50.0,
                    help="|mean| must exceed this x the layer median to count "
                         "as a nuisance coordinate (class-blind, harmless split)")
    ap.add_argument("--nuisance-max-k", type=int, default=16)
    ap.add_argument("--wins-q", type=float, default=0.995)
    ap.add_argument("--wins-axis", default="within_vector",
                    choices=("within_vector", "within_coordinate"))
    ap.add_argument("--shrinkage", type=float, default=1e-3)
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES))
    ap.add_argument("--save-acts", action="store_true",
                    help="also persist raw activations (large; needed only for "
                         "bootstrap re-analysis)")
    ap.add_argument("--dataset", default="default", choices=DATASET_NAMES,
                    help="which prompt-set pair to estimate the direction from. "
                         "'default' is the frozen AdvBench+SORRY-Bench vs Alpaca "
                         "contrast every published number here used; the others "
                         "are replications on disjoint sources. Recorded in "
                         "directions.pt so 24/25/26 inherit it. Use --tag to keep "
                         "the artifacts separate.")
    ap.add_argument("--subset", default="stratified", choices=SUBSETS,
                    help="row ordering before the head slice: 'stratified' "
                         "interleaves sources (default), 'head' is the frozen "
                         "file order (AdvBench-only for the harmful side)")
    ap.add_argument("--skip-norm-gains", action="store_true",
                    help="skip D7 (the weight-level RMSNorm-gain provenance check)")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="bootstrap resamples for D2b (sign-flip p, share band)")
    ap.add_argument("--dial-layers", nargs="+", type=int, default=None,
                    help="layers for the D2c estimator ablations (scale dial, noise "
                         "probes, evidence profile, probe comparison); default: the "
                         "max-share layer of each position")
    ap.add_argument("--skip-ablations", action="store_true",
                    help="skip D2c (the per-layer estimator ablations)")
    ap.add_argument("--norm-gains-max-k", type=int, default=16,
                    help="how many nuisance channels (by number of layers they "
                         "appear in) D7 reports on")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    bs = exp["batch_size"]
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)

    harmful = subset_rows(load_rows(args.dataset, "harmful_extract"), args.subset)
    harmless = subset_rows(load_rows(args.dataset, "harmless_extract"), args.subset)
    print(f"dataset={args.dataset}: {describe_dataset(args.dataset)['description']}")
    # --limit caps both, so a smoke run is `--limit 8` rather than two flags
    n_tr = min(args.n_train, args.limit) if args.limit else args.n_train
    n_va = min(args.n_val, args.limit) if args.limit else args.n_val
    assert len(harmful) >= n_tr + n_va and len(harmless) >= n_tr + n_va, \
        "extraction splits too small for the requested train/val sizes"
    tr_pos = instructions(harmful[:n_tr])
    tr_neg = instructions(harmless[:n_tr])
    print(f"train {len(tr_pos)}+{len(tr_neg)} | val reserved {n_va}+{n_va} "
          f"(rows {n_tr}:{n_tr + n_va}) | positions {args.positions}")
    print(f"subset={args.subset} | train harmful {source_counts(harmful[:n_tr])} "
          f"harmless {source_counts(harmless[:n_tr])} | val harmful "
          f"{source_counts(harmful[n_tr:n_tr + n_va])}")

    acts_pos = extract_multi(model, tr_pos, tuple(args.positions), batch_size=bs)
    acts_neg = extract_multi(model, tr_neg, tuple(args.positions), batch_size=bs)

    # R4 needs word counts of the SAME rows the activations came from
    len_pos = [len(s.split()) for s in tr_pos]
    len_neg = [len(s.split()) for s in tr_neg]
    idx_p, idx_n = length_matched_indices(len_pos, len_neg, seed=exp["seed"])
    print(f"r4 length matching: kept {len(idx_p)} pairs | mean words "
          f"{sum(len_pos)/len(len_pos):.1f}/{sum(len_neg)/len(len_neg):.1f} -> "
          f"{sum(len_pos[i] for i in idx_p)/max(len(idx_p),1):.1f}/"
          f"{sum(len_neg[j] for j in idx_n)/max(len(idx_n),1):.1f}")
    # R5 and D2b need the per-prompt covariates of the same rows (words,
    # terminal punctuation, question form — the nominal-error analysis)
    cov_p, cov_n = prompt_covariates(tr_pos), prompt_covariates(tr_neg)
    cov_sum = covariate_summary(cov_p, cov_n)
    print("covariates (pos/neg, delta in pooled sd): " + " | ".join(
        f"{k} {v['mean_pos']:.2f}/{v['mean_neg']:.2f} d={v['delta_std']:+.2f}"
        for k, v in cov_sum.items()))

    directions: dict[str, dict[str, torch.Tensor]] = {r: {} for r in args.recipes}
    channels: dict[str, dict[int, list[int]]] = {}
    diag: dict[str, dict] = {}

    for pos in args.positions:
        ap_, an_ = acts_pos[pos], acts_neg[pos]
        ch = nuisance_channels(an_, ratio=args.nuisance_ratio,
                               max_k=args.nuisance_max_k)
        channels[pos] = ch
        n_ch = sum(len(v) for v in ch.values())
        for recipe in args.recipes:
            directions[recipe][pos] = build_direction(
                recipe, ap_, an_, channels=ch, shrinkage=args.shrinkage,
                q=args.wins_q, wins_axis=args.wins_axis,
                idx_pos=idx_p, idx_neg=idx_n, cov_pos=cov_p, cov_neg=cov_n)
        # D2b on the top-sigma coordinate per layer (2339 on Gemma); the
        # bootstrap resamples the cached activations, no model involved
        nom = nominal_decomposition(ap_, an_, cov_p, cov_n, n_boot=args.n_boot,
                                    seed=exp["seed"])
        diag[pos] = {
            "nominal": nom,
            "nominal_summary": summarize_nominal(nom),
            "n_nuisance_channels_total": n_ch,
            "nuisance_channels_per_layer": {str(k): v for k, v in ch.items() if v},
            "energy": {r: energy_shares(directions[r][pos]) for r in args.recipes},
            # prediction 5's geometric half, per layer: how far each recipe
            # rotated the vector away from the raw estimator. Written here (the
            # only place all recipes are in memory) so analyze_recipes.py can
            # stay JSON-only. R2 is expected to rotate on every model.
            "cos_to_raw": cos_to_raw({r: directions[r][pos] for r in args.recipes}),
            "evidence": evidence_stats(ap_, an_),
            "norms": norm_decomposition(torch.cat([ap_, an_]), ch),
            "perp_norm": perp_norm(torch.cat([ap_, an_]), ch),
            # does R3's quantile actually reach the nuisance coordinates at
            # this d_model? (see recipes.winsorize_headroom)
            "winsorize_headroom": winsorize_headroom(an_, ch, q=args.wins_q),
        }
        top1 = diag[pos]["energy"]["r0_raw"]["top1_share"]
        wh = diag[pos]["winsorize_headroom"]
        hr = wh["effective"]
        # per-layer headroom: the global INEFFECTIVE flag is usually an
        # early-layer artefact (many nuisance coords -> the quantile lands
        # inside the group). R3 is only meant to be read where the cap clips
        # the nuisance coordinates, so report which layers fail rather than
        # discarding R3 wholesale on one flag.
        bad_layers = sorted(int(k) for k, v in wh["per_layer"].items()
                            if v["n_nuisance"] and not v["clipped"])
        diag[pos]["winsorize_headroom_ineffective_layers"] = bad_layers
        c01 = diag[pos]["cos_to_raw"].get("r1_masked")
        hr_str = "ok" if hr else (f"INEFFECTIVE@L{bad_layers}" if len(bad_layers) <= 8
                                  else f"INEFFECTIVE@{len(bad_layers)} layers "
                                       f"(L{bad_layers[0]}-{bad_layers[-1]})")
        print(f"  {pos:15s} nuisance coords {n_ch:3d} | r0 top1-share "
              f"med {sorted(top1)[len(top1)//2]:.3f} max {max(top1):.3f} | "
              f"max RMS gain {max(diag[pos]['norms']['gain']):.1f}x | "
              f"r3 headroom {hr_str}"
              + (f" | cos(r0,r1) min {min(c01):.3f}" if c01 else ""))
        ns = diag[pos]["nominal_summary"]
        am = ns["at_max"]
        print(f"  {'':15s} nominal: c*={ns['channel_mode']} rho med {ns['rho']:.0f} "
              f"(rms {ns['rho_rms']:.0f}) E med {ns['E_rest']:.0f} | share meas/pred/noise-only med "
              f"{ns['measured_share']:.2f}/{ns['predicted_share']:.2f}/{ns['noise_only_share']:.2f} "
              f"(exact noise {ns['noise_only_share_exact']:.2f}) "
              f"| null(split-half harmless) share {ns['null_share']:.2f} cos {ns['null_cos']:.2f} "
              f"| L{ns['max_share_layer']} share {am['measured_share']:.2f} t raw {am['welch_t']:+.1f} "
              f"-> |words {am['t_adj_words']:+.1f} -> |words+punct {am['t_adj_words_punct']:+.1f} "
              f"(r_words {am['r_words']:+.2f}, sign-flip p {am['signflip_p']:.2f})")

        # D2c — the estimator ablations at one layer (or --dial-layers): pure
        # tensor maths on the cached activations, a few seconds per layer
        if not args.skip_ablations:
            layers_c = args.dial_layers or [ns["max_share_layer"]]
            diag[pos]["ablations"] = {}
            for li in layers_c:
                xp_, xn_ = ap_[:, li], an_[:, li]
                cstar = int(nom["channel"][li])
                pick = most_confound_sensitive_ordinary_channel(xp_, xn_, cov_p[:, 0], cov_n[:, 0])
                abl = {
                    "channel": cstar,
                    "dial_attenuate": scale_dial(xp_, xn_, cstar, (1.0, 1 / 3, 1 / 10, 1 / 30),
                                                 seed=exp["seed"]),
                    "amplify_pick": pick,
                    "dial_amplify": scale_dial(xp_, xn_, pick["channel"],
                                               (1.0, 10.0, 30.0, 100.0, 300.0), seed=exp["seed"]),
                    "noise": noise_probes(xp_, xn_, seed=exp["seed"]),
                    "evidence": evidence_profile(xp_, xn_, cstar, cov_p[:, 0], cov_n[:, 0]),
                    "probes": probe_comparison(xp_, xn_, cstar),
                }
                diag[pos]["ablations"][str(li)] = abl
                att = {r["gain"]: r for r in abl["dial_attenuate"]["rows"]}
                amp = {r["gain"]: r for r in abl["dial_amplify"]["rows"]}
                ev, pr = abl["evidence"], abl["probes"]
                print(f"  {'':15s} ablations L{li} c*={cstar}: attenuate 1/10 -> share "
                      f"{att[0.1]['share']:.2f} auroc {att[1.0]['auroc_heldout']:.2f}->{att[0.1]['auroc_heldout']:.2f} "
                      f"| amplify ch{pick['channel']} (r {pick['r']:+.2f}, own auroc "
                      f"{pick['auroc_single_heldout']:.2f}) x100 -> share {amp[100.0]['share']:.2f} "
                      f"auroc {amp[100.0]['auroc_heldout']:.2f} | noise-only(rho200) share mean "
                      f"{abl['noise']['noise_only']['share_mean']:.2f}")
                print(f"  {'':15s} evidence L{li}: top-100 hold {ev['cum_share_top_k'].get('100', float('nan')):.2f} "
                      f"of E | evidence coords rho med {ev['evidence_top_k']['rho_median']:.1f} "
                      f"| c* rank {ev['cstar']['rank_by_dprime2']}/{ev['d_model']} share "
                      f"{ev['cstar']['evidence_share']:.4f} | raw energy on evidence top-100 "
                      f"{ev['raw_vs_evidence']['energy_share_on_evidence_top_k']:.2f} overlap "
                      f"{ev['raw_vs_evidence']['overlap_top_k']} cos(d,e_c*) "
                      f"{ev['raw_vs_evidence']['cos_with_e_cstar']:+.2f}")
                a = pr["auroc_heldout"]
                print(f"  {'':15s} probes L{li}: auroc raw-dom {a['raw_diff_of_means']:.3f} R2 "
                      f"{a['r2_standardized']:.3f} raw-probe {a['raw_feature_probe']:.3f} "
                      f"std-probe {a['standardized_probe']:.3f} | c* rank/share: std-probe "
                      f"{pr['standardized_probe']['cstar_rank']}/{pr['standardized_probe']['cstar_share']:.4f}, "
                      f"raw-probe (per sigma) {pr['raw_probe']['cstar_rank']}/{pr['raw_probe']['cstar_share']:.3f} "
                      f"| cos(effect, d'): std {pr['standardized_probe']['cos_with_dprime']:.2f} "
                      f"raw {pr['raw_probe']['cos_with_dprime']:.2f} | cos(raw d, raw w) "
                      f"{pr['raw_probe']['cos_raw_d_with_raw_w']:.2f}")
        if args.save_acts:
            torch.save({"pos": ap_, "neg": an_}, adir / f"acts_{pos}.pt")

    # ---- D7: are the activation-identified nuisance channels the ones the
    # weights single out? Channels = union of the class-blind sets over every
    # position and layer, ranked by how many layers they appear in. On a model
    # where the rule found nothing (the controls) fall back to the top-|mean|
    # coordinate per layer so the report is not empty — flagged as such.
    ng_summary = None
    if args.skip_norm_gains:
        print("D7 norm gains: skipped (--skip-norm-gains)")
    elif args.remote:
        print("D7 norm gains: skipped (weights are not local under --remote)")
    else:
        cand, source, count = candidate_channels(
            channels, acts_neg[args.positions[0]], max_k=args.norm_gains_max_k)
        ng = norm_gain_report(model, cand, layers_path=model.cfg.layers_path)
        ng["channels_source"] = source
        ng["channel_layer_counts"] = {str(c): int(count.get(c, 0)) for c in cand}
        print(format_report(ng, nuisance_channels=list(count)) +
              (f"\n  (channels: {source}; layer counts {ng['channel_layer_counts']})"))
        (adir / "norm_gains.json").write_text(json.dumps(ng, indent=1))
        ng_summary = {k: ng.get(k) for k in (
            "gain_offset", "norm_names", "reader_norms", "writer_norms",
            "final_norm_name", "channels", "channels_source", "channel_layer_counts",
            "top_by_writer_gain", "verdict")}
        if "final_norm" in ng:
            ng_summary["final_norm"] = ng["final_norm"]

    torch.save({"directions": directions,
                "positions": args.positions,
                "recipes": args.recipes,
                "n_layers": model.n_layers,
                "n_train": n_tr, "n_val": n_va,
                "val_slice": [n_tr, n_tr + n_va],
                "subset": args.subset,        # 24/26 re-derive the val rows from this
                "dataset": args.dataset,      # ... and from this
                "wins_axis": args.wins_axis, "wins_q": args.wins_q,
                "nuisance_ratio": args.nuisance_ratio,
                "shrinkage": args.shrinkage},
               adir / "directions.pt")
    print(f"wrote {adir / 'directions.pt'}")
    (adir / "channels.json").write_text(json.dumps(
        {p: {str(k): v for k, v in c.items() if v} for p, c in channels.items()},
        indent=1))
    dump(exp, args.model, sub, "diagnostics.json", {
        "model": args.model, "n_layers": model.n_layers,
        "n_train": n_tr, "n_val": n_va,
        "subset": args.subset,
        "dataset": describe_dataset(args.dataset),
        "train_sources": {"harmful": source_counts(harmful[:n_tr]),
                          "harmless": source_counts(harmless[:n_tr])},
        "val_sources": {"harmful": source_counts(harmful[n_tr:n_tr + n_va]),
                        "harmless": source_counts(harmless[n_tr:n_tr + n_va])},
        "config": {"nuisance_ratio": args.nuisance_ratio,
                   "nuisance_max_k": args.nuisance_max_k,
                   "wins_q": args.wins_q, "wins_axis": args.wins_axis,
                   "shrinkage": args.shrinkage},
        "r4_kept_pairs": len(idx_p),
        "covariates": cov_sum,             # words / terminal punct / question form, per class
        "n_boot": args.n_boot,
        "by_position": diag,
        "norm_gains": ng_summary,          # full report in norm_gains.json
    })


if __name__ == "__main__":
    main()
