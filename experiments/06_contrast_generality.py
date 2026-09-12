"""E3 (GPU half) — is the massive channel's class correlate specific to THIS contrast?

At the post-instruction positions channel c* carries a large class gap on the
harmful/harmless contrast (d' ~ 3.2 at L20@t_post_inst-4, t 23 after covariate
control). Three readings of that, which the refusal data alone cannot separate:

  (i)   the model's refusal-relevant state changes its global gain, and c* is
        the gain knob -> the gap is functional (E2 tests this causally);
  (ii)  a position/length artefact that covariate control did not catch;
  (iii) a generic "what kind of task is this" signal that would show up for ANY
        binary text contrast at these positions -> nothing to do with refusal.

(iii) is decided cheaply and offline-ish: run the SAME estimator and the same
diagnostics on an unrelated binary contrast (SST-2 positive vs negative, the
repo's existing semantic control) at the SAME positions and layers. If c* shows
a comparable d' there, the post-instruction correlate is generic; if it is flat
for sentiment and large for harm, it is contrast-specific and (i) or (ii).

One extraction pass, no generation, no intervention. The refusal-side numbers
are already in diagnostics.json, so this script only computes the sentiment
side and writes it in the same shape.

    python experiments/06_contrast_generality.py --model gemma3-12b
    python experiments/06_contrast_generality.py --model gemma3-12b --n-train 128
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, instructions, setup

from loudchannel.config import artifacts_dir
from loudchannel.data.splits import load_split
from loudchannel.nominal import (
    evidence_profile,
    nominal_decomposition,
    probe_comparison,
    prompt_covariates,
    summarize_nominal,
)
from loudchannel.recipes import CANDIDATE_POSITIONS, energy_shares, extract_multi

SUB = "recipes"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--positions", nargs="+", default=list(CANDIDATE_POSITIONS))
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="layers for the evidence/probe profile (default: each "
                         "position's max-share layer in the REFUSAL diagnostics, "
                         "so the two contrasts are read at the same cells)")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)
    diag = json.loads((adir / "diagnostics.json").read_text())["by_position"]

    n = args.limit or args.n_train
    pos_rows = load_split("sst2_pos")[:n]
    neg_rows = load_split("sst2_neg")[:n]
    tr_pos, tr_neg = instructions(pos_rows), instructions(neg_rows)
    print(f"sentiment contrast: {len(tr_pos)}+{len(tr_neg)} | positions {args.positions}")

    acts_pos = extract_multi(model, tr_pos, tuple(args.positions), batch_size=exp["batch_size"])
    acts_neg = extract_multi(model, tr_neg, tuple(args.positions), batch_size=exp["batch_size"])
    cov_p, cov_n = prompt_covariates(tr_pos), prompt_covariates(tr_neg)

    out: dict[str, dict] = {}
    for p in args.positions:
        ap_, an_ = acts_pos[p], acts_neg[p]
        nom = nominal_decomposition(ap_, an_, cov_p, cov_n, n_boot=args.n_boot,
                                    seed=exp["seed"])
        d_raw = ap_.mean(0) - an_.mean(0)
        ref_nom = diag.get(p, {}).get("nominal", {})
        # read BOTH contrasts at the same channel: the refusal side's c* (so a
        # flat sentiment gap there is evidence, not a different coordinate)
        cstar = args.channel if args.channel is not None else (
            Counter(int(c) for c in ref_nom.get("channel", nom["channel"])).most_common(1)[0][0])
        if args.layers:
            layers = args.layers
        elif p in diag:
            layers = [int(diag[p]["nominal_summary"]["max_share_layer"])]
        else:
            layers = [nom_max(nom)]
        entry = {"channel_read": cstar,
                 "energy": energy_shares(d_raw),
                 "nominal": nom, "nominal_summary": summarize_nominal(nom),
                 "at_layers": {}}
        for li in layers:
            xp_, xn_ = ap_[:, li], an_[:, li]
            entry["at_layers"][str(li)] = {
                "evidence": evidence_profile(xp_, xn_, cstar, cov_p[:, 0], cov_n[:, 0]),
                "probes": probe_comparison(xp_, xn_, cstar),
            }
            e = entry["at_layers"][str(li)]["evidence"]
            ref = (diag.get(p, {}).get("ablations", {}) or {}).get(str(li), {})
            ref_d = ref.get("evidence", {}).get("cstar", {}).get("dprime")
            print(f"  {p:15s} L{li:<3d} c*={cstar}: sentiment d' {e['cstar']['dprime']:+.2f} "
                  f"(rank {e['cstar']['rank_by_dprime2']}, share {e['cstar']['evidence_share']:.4f}, "
                  f"E {e['E']:.0f}) | refusal d' "
                  f"{('%+.2f' % ref_d) if ref_d is not None else '   n/a'}")
        out[p] = entry

    dump(exp, args.model, sub, "contrast_generality.json", {
        "model": args.model, "contrast": "sst2_pos vs sst2_neg",
        "n_per_class": len(tr_pos), "positions": args.positions,
        "by_position": out,
    })


def nom_max(nom: dict) -> int:
    shares = nom["measured_share"]
    return int(max(range(len(shares)), key=lambda i: shares[i]))


if __name__ == "__main__":
    main()
