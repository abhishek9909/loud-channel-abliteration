"""Stage S — score the candidate grid and select one direction per recipe.

Consumes 23_recipe_extract's directions.pt; for every (recipe, position, layer)
runs Arditi et al.'s three validation measurements on the held-out val slice of
the extraction splits:

    bypass   refusal metric on HARMFUL prompts, direction ablated everywhere
    induce   refusal metric on HARMLESS prompts, raw direction added at its layer
    kl       KL(clean || ablated) of the first response-token distribution on
             HARMLESS prompts

and picks min-bypass subject to kl < 0.1, induce > 0, layer < 0.8L.

The KL constraint is the load-bearing part: a candidate that is mostly a
massive-activation coordinate perturbs harmless prompts just as hard as harmful
ones, so it fails KL no matter how well it "bypasses" refusal. Recording the
rejection reason for every cell is therefore a result in its own right — "all
288 raw candidates rejected for KL" is the quantitative statement that the
original vector was never a feature.

Cost: 3 batched forward passes per candidate. Use --layer-step for a coarse
grid first (a full 48-layer x 6-position x 5-recipe sweep is 1440 candidates).

    python experiments/02_select.py --model gemma3-12b --layer-step 2
    python experiments/02_select.py --model llama3-8b
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.config import artifacts_dir
from loudchannel.data.datasets import load_rows
from loudchannel.data.splits import load_split, source_counts, subset_rows
from loudchannel.interventions import logits_with_intervention
from loudchannel.positions import positions_for
from loudchannel.selection import (
    INDUCE_FRAC_MIN,
    INDUCE_RULES,
    KL_THRESHOLD,
    OBJECTIVES,
    apply_constraints,
    refusal_metric,
    score_candidate,
    selection_report,
)

SUB = "recipes"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--layer-step", type=int, default=1,
                    help="subsample the layer grid (1 = every layer)")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="explicit layer list, overrides --layer-step")
    ap.add_argument("--recipes", nargs="+", default=None,
                    help="subset of the recipes present in directions.pt")
    ap.add_argument("--positions", nargs="+", default=None)
    ap.add_argument("--kl-threshold", type=float, default=KL_THRESHOLD)
    ap.add_argument("--induce-rule", default="frac", choices=INDUCE_RULES,
                    help="feasibility rule for induce: 'frac' (>= --induce-frac-min "
                         "of harmless prompts flip, robust to bf16 mean tails; default) "
                         "or 'mean' (Arditi's mean logit > 0). Both are always recorded.")
    ap.add_argument("--induce-frac-min", type=float, default=INDUCE_FRAC_MIN)
    ap.add_argument("--objective", default="mean", choices=OBJECTIVES,
                    help="argmin objective among feasible cells: 'mean' bypass (Arditi, "
                         "pre-registered) or 'frac' (share of harmful prompts still refusing)")
    ap.add_argument("--no-common-dose", action="store_true",
                    help="skip the induce measurement at alpha = perp_norm (one extra "
                         "forward pass per candidate)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)

    blob = torch.load(adir / "directions.pt", map_location="cpu", weights_only=False)
    directions = blob["directions"]
    recipes = args.recipes or blob["recipes"]
    positions = args.positions or blob["positions"]
    n_layers = model.n_layers
    assert blob["n_layers"] == n_layers, "directions.pt came from a different model"
    lo, hi = blob["val_slice"]
    # the val rows must come from the SAME ordering 23 trained on, or they
    # could overlap the train rows; the blob carries it (pre-subset blobs
    # were written under the frozen head order)
    subset = blob.get("subset", "head")
    # perp_norm[position][layer] -> the common dose for the induce readout, so
    # every recipe's addition is scored at the same effective alpha rather than
    # at its own ||row|| (which differs by construction: R2 is renormalised to
    # the raw norm, R1 keeps the masked norm)
    perp = {}
    if not args.no_common_dose:
        diag_p = adir / "diagnostics.json"
        if diag_p.exists():
            bp = json.loads(diag_p.read_text()).get("by_position", {})
            perp = {pos: bp[pos]["perp_norm"] for pos in bp if "perp_norm" in bp.get(pos, {})}
        else:
            print("  (no diagnostics.json; common-dose induce skipped)")

    dataset = blob.get("dataset", "default")
    harmful = subset_rows(load_rows(dataset, "harmful_extract"), subset)[lo:hi]
    harmless = subset_rows(load_rows(dataset, "harmless_extract"), subset)[lo:hi]
    if args.limit:
        harmful, harmless = harmful[:args.limit], harmless[:args.limit]
    pps_harmful = positions_for(model, instructions(harmful), seed=exp["seed"])
    pps_harmless = positions_for(model, instructions(harmless), seed=exp["seed"])
    bs = max(len(pps_harmful), len(pps_harmless))   # val sets fit one batch
    print(f"val {len(pps_harmful)} harmful / {len(pps_harmless)} harmless "
          f"| subset={subset} {source_counts(harmful)} "
          f"| recipes {recipes} | positions {positions}")

    # clean baselines: one pass each, reused by every candidate
    clean_harmful = logits_with_intervention(
        model, pps_harmful, direction=None, layers=[], position=None,
        mode="none", batch_size=bs)
    clean_harmless = logits_with_intervention(
        model, pps_harmless, direction=None, layers=[], position=None,
        mode="none", batch_size=bs)
    cr_harmful = float(refusal_metric(clean_harmful, model.tokenizer).mean())
    cr_harmless = float(refusal_metric(clean_harmless, model.tokenizer).mean())
    print(f"clean refusal metric: harmful {cr_harmful:+.3f} | harmless {cr_harmless:+.3f}")

    layers = args.layers if args.layers is not None else \
        list(range(0, n_layers, args.layer_step))
    todo = [(r, p, l) for r in recipes for p in positions for l in layers]
    print(f"scoring {len(todo)} candidates ({len(layers)} layers x "
          f"{len(positions)} positions x {len(recipes)} recipes)")

    cands = []
    for recipe, pos, layer in tqdm(todo, desc="candidates"):
        row = directions[recipe][pos][layer]
        if float(row.norm()) < 1e-8:            # e.g. a fully masked layer
            continue
        ca = perp.get(pos, [None] * n_layers)[layer] if perp.get(pos) else None
        cands.append(score_candidate(
            model, row, layer, pps_harmful=pps_harmful, pps_harmless=pps_harmless,
            clean_harmless_logits=clean_harmless,
            clean_harmful_refusal=cr_harmful, clean_harmless_refusal=cr_harmless,
            recipe=recipe, position=pos, batch_size=bs, common_alpha=ca))

    apply_constraints(cands, n_layers, kl_threshold=args.kl_threshold,
                      induce_rule=args.induce_rule, induce_frac_min=args.induce_frac_min)
    report = selection_report(cands, n_layers, objective=args.objective)
    report["clean"] = {"harmful_refusal_metric": cr_harmful,
                       "harmless_refusal_metric": cr_harmless,
                       "n_val": len(pps_harmful),
                       "val_sources": source_counts(harmful)}
    report["config"] = {"layers": layers, "kl_threshold": args.kl_threshold,
                        "subset": subset, "induce_rule": args.induce_rule,
                        "induce_frac_min": args.induce_frac_min,
                        "objective": args.objective,
                        "common_dose": bool(perp)}

    for recipe in recipes:
        r = report["by_recipe"].get(recipe)
        if not r:
            continue
        s = r["selected"]
        head = (f"L{s['layer']}@{s['position']} bypass {s['bypass']:+.3f} "
                f"(clean {cr_harmful:+.3f}) kl {s['kl']:.4f} induce_frac {s['induce_frac']:.2f}"
                f" @{s['induce_passed_at'] or '-'}" if s
                else f"NO FEASIBLE CANDIDATE  rejects={r['reject_reasons']}")
        print(f"  {recipe:18s} feasible {r['n_feasible']:4d}/{r['n_candidates']:4d} "
              f"(arditi {r['n_feasible_arditi']:4d}) | min kl {r['min_kl']:.4f} | {head}")

    dump(exp, args.model, sub, "selection.json", report)


if __name__ == "__main__":
    main()
