"""Decision tables for E1/E2/E3 — the causal status of the massive channel.

Four tables, each with the reading rule written next to it so the result can be
read without re-deriving the argument:

  T1  E3 evidence accounting (OFFLINE, from diagnostics.json — needs no GPU).
      Per position, at its max-share layer: how much of the raw difference-of-
      means vector sits on c*, how much of the layer's class EVIDENCE sits on
      c*, and what each readout achieves. The pair (share of vector, share of
      evidence) is the whole quantitative claim: the estimator overweights c*
      by two to three orders of magnitude relative to its evidence, at EVERY
      position — including the positions where c*'s own d' is large.
  T2  E3 contrast generality (06_contrast_generality.py): is the post-
      instruction class correlate on c* specific to harm/harmless, or does an
      unrelated binary contrast put the same signal there?
  T3  E1 route separation (27 --stage e1).
  T4  E2 causal test (27 --stage e2).

Plus a transcript spot check: induced/remaining refusals are counted by a
lexical classifier, so "did it really refuse" needs eyes on the text at least
once. --spot prints the distinct opening lines per arm and the share of the
most common one (a collapsed distribution = one templated line = the readout
was measuring a mode, not refusal behaviour).

    python scripts/analyze_channel_causal.py --model gemma3-12b
    python scripts/analyze_channel_causal.py --model gemma3-12b --spot
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_STOP = set("the a to of and in is it you that for on with this can be are as or "
            "your i not have will if how what we they there here from at by an".split())


def fluency(path: Path) -> dict:
    """Two cheap English-fluency statistics per transcript file.

    The three-way gate catches REPETITION (word_run, char_run, top_word_share)
    and nothing else, so it scores high-entropy multilingual word salad as
    `compliance`. E1's e1_cstar_zero arm is exactly that: 100/100 "compliance",
    0% degenerate, MMLU 0.262 and NLL x5.3. `stopword` (share of tokens that
    are common English function words) separates the two populations with a
    wide margin -- healthy arms sit at 0.31-0.38, broken ones at 0.00-0.12 --
    and `ascii` catches the script-salad variant. Reported here rather than
    folded into DegeneracyGate: a new gate rule has to be calibrated against
    the stored known-good/known-degenerate transcript populations the way the
    existing rules were (tests/test_degeneracy.py), not bolted on mid-analysis.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}
    asc, stop = [], []
    for r in rows:
        t = r.get("transcript") or ""
        al = [c for c in t if c.isalpha()]
        asc.append(sum(c.isascii() for c in al) / max(len(al), 1))
        w = [x.strip(".,:;!?'\"()[]").lower() for x in t.split()]
        stop.append(sum(x in _STOP for x in w) / max(len(w), 1))
    return {"n": len(rows), "ascii": sum(asc) / len(asc), "stopword": sum(stop) / len(stop)}


def _load(adir: Path, name: str):
    p = adir / name
    return json.loads(p.read_text()) if p.exists() else None


def t1_evidence(diag: dict) -> None:
    print("\n== T1  E3 evidence accounting (offline) "
          "== share of the VECTOR vs share of the EVIDENCE ==")
    print(f"{'position':15s} {'L':>3s} {'vec share':>9s} {'d_prime':>8s} {'rank':>6s} "
          f"{'ev share':>9s} {'ratio':>7s} {'E':>6s} {'best ord':>8s} "
          f"{'raw':>6s} {'r2':>6s} {'probe':>6s} {'null sh':>7s}")
    for pos, v in diag.items():
        nom = v["nominal"]
        for L, a in (v.get("ablations") or {}).items():
            li = int(L)
            e, p = a["evidence"], a["probes"]["auroc_heldout"]
            vec = nom["measured_share"][li]
            ev = e["cstar"]["evidence_share"]
            print(f"{pos:15s} {li:3d} {vec:9.3f} {e['cstar']['dprime']:+8.2f} "
                  f"{e['cstar']['rank_by_dprime2']:6d} {ev:9.4f} "
                  f"{vec / max(ev, 1e-12):7.0f} {e['E']:6.0f} "
                  f"{e['best_channel']['dprime']:+8.2f} "
                  f"{p['raw_diff_of_means']:6.3f} {p['r2_standardized']:6.3f} "
                  f"{p['standardized_probe']:6.3f} {nom['null_share'][li]:7.2f}")
    print("  read: `ratio` is the overweighting factor (vector share / evidence share);"
          "\n        `raw` vs `probe` is the readout cost of letting scale set the weights;"
          "\n        `null sh` is c*'s share of a split-half of the HARMLESS set alone —"
          "\n        high null share = the estimator finds this axis with no class at all."
          "\n  NOTE: `d_prime` large with `ev share` tiny is NOT a contradiction — E is the"
          "\n        sum over 3840 coordinates, so even a top-10 coordinate is <1% of it.")


def t2_generality(gen: dict | None, diag: dict) -> None:
    if not gen:
        print("\n== T2  E3 contrast generality == (no contrast_generality.json — run 28)")
        return
    print("\n== T2  E3 contrast generality == same channel, unrelated contrast ==")
    print(f"{'position':15s} {'L':>3s} {'sent d_prime':>12s} {'sent rank':>10s} "
          f"{'harm d_prime':>12s} {'harm rank':>10s} {'verdict':>18s}")
    for pos, v in gen["by_position"].items():
        for L, a in v["at_layers"].items():
            e = a["evidence"]
            ref = ((diag.get(pos, {}).get("ablations") or {}).get(L, {})
                   .get("evidence", {}).get("cstar", {}))
            hd, hr = ref.get("dprime"), ref.get("rank_by_dprime2")
            verdict = "-"
            if hd is not None:
                verdict = ("generic" if abs(e["cstar"]["dprime"]) > 0.5 * abs(hd)
                           else "contrast-specific")
            print(f"{pos:15s} {int(L):3d} {e['cstar']['dprime']:+12.2f} "
                  f"{e['cstar']['rank_by_dprime2']:10d} "
                  f"{(('%+.2f' % hd) if hd is not None else 'n/a'):>12s} "
                  f"{(str(hr) if hr is not None else 'n/a'):>10s} {verdict:>18s}")
    print("  read: a large sentiment d' on the SAME channel means the post-instruction"
          "\n        correlate is a generic task signal, not a refusal one.")


def t3_e1(e1: dict | None, adir: Path | None = None) -> None:
    if not e1:
        print("\n== T3  E1 route separation == (no channel_causal_e1.json — run 27 --stage e1)")
        return
    d = e1["decomposition"]
    print(f"\n== T3  E1 route separation == c*={e1['channel']}  a={d['a']:+.4f} "
          f"b={d['b']:.4f}  route2 gain={d['route2_gain']:.4f}  leak coef={d['leak_coef']:+.4f} ==")
    arms = {k: v for k, v in e1["arms"].items() if v.get("stage") == "e1"}
    base = arms.get("clean", {}).get("harmful", {}).get("refusal_rate_strict")
    print(f"{'arm':18s} {'h refusal':>9s} {'drop':>7s} {'h degen':>8s} {'b degen':>8s} "
          f"{'mmlu':>6s} {'nll':>7s} {'stopword':>9s} {'ascii':>6s}")
    for name, r in arms.items():
        cap = r.get("capability", {})
        drop = (r["harmful"]["refusal_rate_strict"] - base) if base is not None else float("nan")
        fl = fluency(adir / f"transcripts_{name}_harmful_eval.jsonl") \
            if adir and (adir / f"transcripts_{name}_harmful_eval.jsonl").exists() else {}
        print(f"{name:18s} {r['harmful']['refusal_rate_strict']:9.3f} {drop:+7.3f} "
              f"{r['harmful']['degenerate_rate']:8.3f} {r['harmless']['degenerate_rate']:8.3f} "
              f"{cap.get('mmlu_acc', float('nan')):6.3f} {cap.get('nll', float('nan')):7.3f} "
              f"{fl.get('stopword', float('nan')):9.3f} {fl.get('ascii', float('nan')):6.3f}")
    print("  read: e1_cstar_zero clean  -> the collapse is the LEAK; the headline stays"
          "\n          'the estimator loaded the sink', and removing the channel is harmless."
          "\n        e1_cstar_zero broken -> projection ablation is hostile to Gemma's norm"
          "\n          geometry independently of any estimator (a stronger, more general claim)."
          "\n        e1_leak_only ~ e1_full_r0 -> route 1 accounts for the whole effect."
          "\n        both partial -> report the split; neither route alone is the story."
          "\n  WARNING: refusal 0 + degen 0 is NOT a clean bypass on its own. The gate"
          "\n        scores fluent multilingual word salad as `compliance`; read `mmlu`,"
          "\n        `nll` and `stopword` (healthy 0.31-0.38, broken 0.00-0.12) together.")


def t4_e2(e2: dict | None) -> None:
    if not e2:
        print("\n== T4  E2 causal test == (no channel_causal_e2.json — run 27 --stage e2)")
        return
    m = e2.get("e2", {})
    print(f"\n== T4  E2 causal test == c*={e2['channel']} @ {m.get('position')} "
          f"L{m.get('layers', [None])[0]}-{m.get('layers', [None])[-1]} "
          f"gap={m.get('gap')} x{m.get('scale')} ==")
    print(f"{'arm':26s} {'on':>9s} {'metric':>8s} {'d metric':>9s} {'d frac':>7s} "
          f"{'gen ref':>8s} {'gen deg':>8s}")
    for name, r in e2["arms"].items():
        if r.get("stage") != "e2":
            continue
        g = r.get("generation", {})
        ro = r["readout"]
        print(f"{name:26s} {r['on']:>9s} {ro['refusal_metric']:8.3f} "
              f"{ro['delta_metric']:+9.3f} {ro['delta_frac']:+7.3f} "
              f"{g.get('refusal_rate_strict', float('nan')):8.3f} "
              f"{g.get('degenerate_rate', float('nan')):8.3f}")
    print("  read: e2_to_* moves refusal and e2_gain_* does not  -> c* is causally used"
          "\n          for this contrast, via the norm; 'causally inert' must go, and the"
          "\n          paper gains a mechanism result alongside the measurement correction."
          "\n        e2_to_* and e2_gain_* move together -> it is the norm change, not the"
          "\n          channel's identity; still causal, but the channel is interchangeable."
          "\n        neither moves (and e2_ordinary_* does) -> 'inert for this contrast'"
          "\n          becomes a measurement rather than an inference from reader gain 0."
          "\n        nothing moves at all, e2_ordinary_* included -> the instrument is not"
          "\n          sensitive at this position; re-run with --e2-scale 2 before concluding.")


def _content_words(text: str) -> set:
    return {w.strip(".,:;!?'\"()[]").lower() for w in text.split()
            if len(w.strip(".,:;!?'\"()[]")) > 3}


def spot(adir: Path, pattern: str = "transcripts_*") -> None:
    """Is a scored refusal a real, request-aware refusal or boilerplate?

    Two numbers, because the obvious one is misleading on its own. `top open`
    is the share of rows sharing the most common first six words -- steered
    refusals often share an opener ("I am programmed to be a...") while the
    BODY still addresses the specific request, which is potency, not collapse.
    `on-topic` is the share of refusals whose text reuses at least two content
    words from their own instruction: high means the model tracked the request
    and then declined it; low with a high `top open` means one canned string
    regardless of input, and the lexical refusal classifier was counting a mode.
    """
    print("\n== spot check: are the scored refusals real, request-aware refusals? ==")
    print(f"{'file':46s} {'n':>4s} {'refuse':>7s} {'degen':>6s} {'distinct':>8s} "
          f"{'top open':>9s} {'on-topic':>9s}  top opening")
    for p in sorted(adir.glob(f"{pattern}.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        ref = [r for r in rows if r.get("label") == "refusal"]
        deg = [r for r in rows if r.get("label") == "degenerate"]
        pool = ref or rows
        opens = [" ".join((r.get("transcript") or "").strip().split()[:6]) for r in pool]
        c = Counter(opens)
        top, n_top = (c.most_common(1)[0] if c else ("", 0))
        on = [r for r in pool
              if len(_content_words(r.get("transcript", ""))
                     & _content_words(r.get("instruction", ""))) >= 2]
        print(f"{p.name[:46]:46s} {len(rows):4d} {len(ref):7d} {len(deg):6d} {len(c):8d} "
              f"{n_top / max(len(pool), 1):9.2f} {len(on) / max(len(pool), 1):9.2f}  {top[:44]!r}")
    print("  read: on-topic high (>~0.6) = the steered model understood the request and"
          "\n        refused it -- genuine induced refusal, even if the opener repeats."
          "\n        on-topic low with top open high = one canned string; the readout was"
          "\n        measuring a mode, and the 'induce' criterion is not evidence of potency."
          "\n        degen > 0 alongside refusals scored on the same file = read those rows:"
          "\n        repetition that still contains 'I cannot' can score as refusal.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--artifacts", default=None,
                    help="artifacts dir (default: artifacts/<model>/recipes)")
    ap.add_argument("--spot", action="store_true", help="run the transcript spot check")
    ap.add_argument("--spot-pattern", default="transcripts_*")
    args = ap.parse_args()

    if args.artifacts:
        adir = Path(args.artifacts)
    else:
        adir = REPO / "artifacts" / args.model / "recipes"
    print(f"artifacts: {adir}")

    diag = (_load(adir, "diagnostics.json") or {}).get("by_position", {})
    if diag:
        t1_evidence(diag)
    t2_generality(_load(adir, "contrast_generality.json"), diag)
    t3_e1(_load(adir, "channel_causal_e1.json"), adir)
    t4_e2(_load(adir, "channel_causal_e2.json"))
    if args.spot:
        spot(adir, args.spot_pattern)


if __name__ == "__main__":
    main()
