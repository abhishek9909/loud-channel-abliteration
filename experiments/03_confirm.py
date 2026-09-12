"""Stage E+C — confirm the selected directions by generation, three-way scored.

The headline measurement of the recipe experiment. For each arm we ablate ONE
direction at every layer and every token position, generate on the held-out
*_eval splits, and label every response

    refusal / compliance / degenerate

Three-way is mandatory, not fastidious. Degenerate output contains no refusal
substrings, so under binary scoring a vector whose ablation destroys the model
scores as a 100% refusal bypass — which is exactly what the original Gemma
arms did (transcripts_ablate_refuse.jsonl is 200/200 'pairs pairs pairs...').
The claim we are testing is "refusal fell AND the remaining outputs are real
answers", so the second half needs its own class. See loudchannel/degeneracy.py.

Arms:
  clean                  no intervention
  <recipe>               the direction 24_recipe_select chose for that recipe
  legacy_layerlocal      the ORIGINAL pipeline's d_refuse, applied per-layer
                         (48 different vectors) — the arm we expect to degenerate
  random                 norm-matched random direction (the raw null)
  random_perp            norm-matched random direction with the nuisance
                         coordinates zeroed — the null that does NOT leak the
                         massive coordinate under ablation (identical to
                         `random` on models with no nuisance set: Llama/Qwen)

Also reruns the capability pair (MMLU slice + NLL) per arm, so "did we fix it
or just break it more gently" is answered in the same file.

Test-set composition (--subset): harmful_eval is stored 150 AdvBench then 150
SORRY-Bench, so the default --n-test 150 under the frozen order is AdvBench
only (Gemma's clean refusal rate there is 0.89, vs 0.69 on the mixed set the
paper's Table 1 reports). Default `stratified` interleaves the sources first
(75+75 at n=150); `head` keeps the frozen order. Recorded in confirm.json.

    python experiments/03_confirm.py --model gemma3-12b
    python experiments/03_confirm.py --model llama3-8b --n-test 100
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.config import artifacts_dir
from loudchannel.data.datasets import describe as describe_dataset, load_rows
from loudchannel.data.splits import SUBSETS, load_split, source_counts, subset_rows
from loudchannel.degeneracy import GATE, summarize
from loudchannel.directions import Direction, random_directions, zero_channels
from loudchannel.interventions import generate_with_intervention
from loudchannel.mmlu import mmlu_accuracy
from loudchannel.pipelines import perplexity
from loudchannel.selection import broadcast_direction

SUB = "recipes"


def gen_and_score(model, instrs, *, direction, layers, mode, max_new_tokens, bs):
    prompts = [model.render(i) for i in instrs]
    texts = generate_with_intervention(
        model, prompts, direction=direction, layers=layers, mode=mode,
        position="all", max_new_tokens=max_new_tokens, batch_size=bs)
    return texts, summarize(texts)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--n-test", type=int, default=150,
                    help="held-out prompts per class from the *_eval splits")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of arms to run (default: all)")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument("--legacy-sub", default="directions",
                    help="artifacts sub-dir holding the original d_refuse.pt")
    ap.add_argument("--extra-cells", nargs="+", default=None,
                    help="exploratory ablation arms as recipe:layer:position "
                         "(e.g. r1_masked:24:t_post_inst-4) — confirmed even when "
                         "the selection rule did not pick them; labelled extra_*")
    ap.add_argument("--subset", default="stratified", choices=SUBSETS,
                    help="row ordering of the *_eval splits before the head "
                         "slice (stratified = source-balanced, head = frozen order)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)
    n_layers = model.n_layers
    all_layers = list(range(n_layers))

    blob = torch.load(adir / "directions.pt", map_location="cpu", weights_only=False)
    sel = json.loads((adir / "selection.json").read_text())

    n_test = args.limit or args.n_test
    # the dataset travels in directions.pt, so the arms can never be scored on a
    # different contrast than the directions were estimated from
    dataset = blob.get("dataset", "default")
    harmful_rows = subset_rows(load_rows(dataset, "harmful_eval"), args.subset)[:n_test]
    harmless_rows = subset_rows(load_rows(dataset, "harmless_eval"), args.subset)[:n_test]
    print(f"dataset={dataset}: {describe_dataset(dataset)['description']}")
    harmful = instructions(harmful_rows)
    harmless = instructions(harmless_rows)
    print(f"test {len(harmful)} harmful / {len(harmless)} harmless "
          f"| subset={args.subset} {source_counts(harmful_rows)} "
          f"| ablating at all {n_layers} layers, all positions")

    # ---- assemble the arms --------------------------------------------------
    arms: dict[str, dict] = {"clean": {"direction": None, "layers": [], "mode": "none"}}
    for recipe, r in sel["by_recipe"].items():
        s = r.get("selected")
        if not s:
            print(f"  {recipe}: no feasible candidate — arm skipped")
            continue
        row = blob["directions"][recipe][s["position"]][s["layer"]]
        arms[recipe] = {
            "direction": broadcast_direction(
                row, n_layers, kind=f"sel|{recipe}",
                position=s["position"], model_name=args.model),
            "layers": all_layers, "mode": "ablate",
            "cell": {"layer": s["layer"], "position": s["position"],
                     "bypass": s["bypass"], "kl": s["kl"]},
        }
    legacy_p = artifacts_dir(exp, args.model, args.legacy_sub) / "d_refuse.pt"
    if legacy_p.exists():
        arms["legacy_layerlocal"] = {
            "direction": Direction.load(legacy_p),   # per-layer vectors, as before
            "layers": all_layers, "mode": "ablate",
            "cell": {"note": "original pipeline: a different vector at each layer"},
        }
    else:
        print(f"  legacy arm skipped ({legacy_p} not found)")
    # union of the class-blind nuisance coordinates over positions, per layer
    # — what the null must avoid to not leak the massive coordinate (Gemma);
    # empty on Llama/Qwen, so random_perp == random there
    nuis_by_layer: dict[int, list[int]] = {}
    ch_p = adir / "channels.json"
    if ch_p.exists():
        chans = json.loads(ch_p.read_text())
        for _pos, per_layer in chans.items():
            for lstr, cs in per_layer.items():
                nuis_by_layer.setdefault(int(lstr), set()).update(cs)
        nuis_by_layer = {k: sorted(v) for k, v in nuis_by_layer.items()}
    n_nuis = sum(len(v) for v in nuis_by_layer.values())

    ref = arms.get("r0_raw", {}).get("direction") or \
        next((a["direction"] for a in arms.values() if a.get("direction") is not None), None)
    if ref is not None:
        rand = random_directions(ref, 1, seed=exp["seed"])[0]
        arms["random"] = {
            "direction": rand,
            "layers": all_layers, "mode": "ablate",
            "cell": {"note": "norm-matched random direction (raw null)"},
        }
        # the null drawn orthogonal to the nuisance set: on Gemma this removes
        # the RMSNorm leak that makes plain `random` destructive; on a model
        # with no nuisance set it is the SAME vector (n_nuis == 0)
        rand_perp = zero_channels(rand, nuis_by_layer) if n_nuis else rand
        arms["random_perp"] = {
            "direction": rand_perp,
            "layers": all_layers, "mode": "ablate",
            "cell": {"note": f"random null orthogonal to {n_nuis} nuisance coords "
                             f"({'== random; no nuisance set' if not n_nuis else 'leak removed'})"},
        }
    # exploratory extra cells: recipe:layer:position, confirmed regardless of
    # the selection rule (so R1's best-bypass cell can be seen even if `induce`
    # did not select it) — read the recorded Stage-S stats for the label
    for spec in (args.extra_cells or []):
        rc, lstr, pos = spec.split(":", 2)
        layer = int(lstr)
        row = blob["directions"][rc][pos][layer]
        stat = next((c for c in sel.get("candidates", [])
                     if c["recipe"] == rc and c["layer"] == layer and c["position"] == pos), {})
        arms[f"extra_{rc}_L{layer}_{pos}"] = {
            "direction": broadcast_direction(row, n_layers, kind=f"extra|{rc}",
                                             position=pos, model_name=args.model),
            "layers": all_layers, "mode": "ablate",
            "cell": {"layer": layer, "position": pos, "exploratory": True,
                     "kl": stat.get("kl"), "bypass": stat.get("bypass"),
                     "induce_frac": stat.get("induce_frac"),
                     "note": "exploratory cell (not selected)"},
        }
    if args.arms:
        # --extra-cells arms are always kept: they are named by cell, not by
        # recipe, so they can never appear in --arms, and filtering them out
        # silently turns `--arms clean --extra-cells r0_raw:17:...` into a
        # clean-only run that exits 0 having measured nothing.
        arms = {k: v for k, v in arms.items()
                if k in args.arms or k.startswith("extra_")}

    # ---- run ---------------------------------------------------------------
    results: dict[str, dict] = {}
    for name, arm in arms.items():
        print(f"\n=== arm {name} ===")
        t_h, s_h = gen_and_score(
            model, harmful, direction=arm["direction"], layers=arm["layers"],
            mode=arm["mode"], max_new_tokens=args.max_new_tokens,
            bs=args.gen_batch_size)
        t_b, s_b = gen_and_score(
            model, harmless, direction=arm["direction"], layers=arm["layers"],
            mode=arm["mode"], max_new_tokens=args.max_new_tokens,
            bs=args.gen_batch_size)
        cap = {}
        if not args.skip_capability and arm["mode"] != "none":
            correct = mmlu_accuracy(
                model, load_split("mmlu_slice")[:exp.get("mmlu_slice", 500)],
                direction=arm["direction"], layers=arm["layers"],
                mode="ablate", batch_size=exp["batch_size"])
            cap["mmlu_acc"] = float(sum(correct) / max(len(correct), 1))
            cap["nll"] = perplexity(model, harmless[:50], direction=arm["direction"],
                                    layers=arm["layers"], mode="ablate",
                                    batch_size=args.gen_batch_size)
        elif not args.skip_capability:
            correct = mmlu_accuracy(
                model, load_split("mmlu_slice")[:exp.get("mmlu_slice", 500)],
                direction=None, layers=[], mode="none",
                batch_size=exp["batch_size"])
            cap["mmlu_acc"] = float(sum(correct) / max(len(correct), 1))
            cap["nll"] = perplexity(model, harmless[:50], mode="none",
                                    batch_size=args.gen_batch_size)

        for split, texts, s in (("harmful_eval", t_h, s_h), ("harmless_eval", t_b, s_b)):
            src_rows = harmful_rows if split == "harmful_eval" else harmless_rows
            (adir / f"transcripts_{name}_{split}.jsonl").write_text("\n".join(
                json.dumps({"i": i, "arm": name, "split": split,
                            "source": r0.get("source"),
                            "instruction": r0["instruction"], "transcript": t,
                            **{k: v for k, v in row.items() if k != "rows"}},
                           ensure_ascii=False)
                for i, (r0, t, row) in enumerate(zip(src_rows, texts, s["rows"]))) + "\n")

        results[name] = {
            "cell": arm.get("cell"),
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
        print(f"\n{'arm':22s} {'refusal':>8s} {'drop':>7s} {'degen':>7s} "
              f"{'mmlu':>6s} {'kl(sel)':>8s}")
        for name, r in results.items():
            kl = (r["cell"] or {}).get("kl")
            print(f"{name:22s} {r['harmful']['refusal_rate_strict']:8.3f} "
                  f"{r['harmful']['refusal_rate_strict'] - base:+7.3f} "
                  f"{r['harmful']['degenerate_rate']:7.3f} "
                  f"{r['capability'].get('mmlu_acc', float('nan')):6.3f} "
                  f"{(f'{kl:.4f}' if kl is not None else '-'):>8s}")

    dump(exp, args.model, sub, "confirm.json", {
        "model": args.model, "n_test": len(harmful),
        "subset": args.subset,
        "dataset": describe_dataset(dataset),
        "test_sources": {"harmful": source_counts(harmful_rows),
                         "harmless": source_counts(harmless_rows)},
        "max_new_tokens": args.max_new_tokens,
        "gate": {k: getattr(GATE, k) for k in GATE.__dataclass_fields__},
        "arms": results,
    })


if __name__ == "__main__":
    main()
