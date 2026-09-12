"""Causal selection of a refusal direction (Stage S), after Arditi et al. 2024.

The estimator recipes in `recipes.py` decide HOW a candidate vector is built.
This module decides WHICH candidate to use, and it is the part the original
pipeline never had: `03_h2_grid` pinned a cell from a steering-onset sweep and
then ablated 48 layer-local vectors at every token. That procedure has no way
to notice that a candidate is a global bias rather than a feature.

Arditi's selection does notice, and the reason is worth stating because it is
the whole point of adopting it. Ablating a direction that is mostly a
massive-activation coordinate rescales the RMSNorm denominator and dumps a
fixed-sign push along the vector's remainder — on EVERY prompt, harmless ones
included. So the first-token distribution on harmless prompts moves a lot, and
`kl_score` is large. A genuine refusal direction, ablated, should leave a
harmless prompt essentially untouched. The KL filter is therefore a specificity
test, and "is this vector secretly a bias?" is the specificity failure it is
most sensitive to.

Protocol (their App. C, reproduced):
  candidates   r = mu_harmful - mu_harmless for every (layer, post-instruction
               position), here x every recipe;
  bypass       mean refusal metric on held-out HARMFUL prompts with r ablated
               at every layer and every token position  (minimize);
  induce       mean refusal metric on held-out HARMLESS prompts with the raw r
               added at its own layer, all token positions  (require > 0);
  kl           mean KL(clean || ablated) of the first response-token
               distribution on HARMLESS prompts  (require < 0.1);
  layer prune  l < 0.8 * L, so nothing adjacent to the unembedding is picked.
  choose       min bypass subject to the three constraints.

Deviation from Arditi, deliberate: their bypass/induce use a substring
classifier over generated text. On Gemma that classifier is the same fragile
instrument that made Table 1' swing from 0.487 to 0.76 on a markdown fix, and
generating for every candidate would cost |I| x L x n generations. We use their
own efficiency metric instead -- refusal_metric(p) := logit P(refusal) at the
first response position -- and keep generation for the confirmation stage,
where it is scored three ways (see degeneracy.py).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from .directions import Direction

# Only `score_candidate` touches the model; the readouts, constraints and
# selection rule are pure tensor maths and stay importable without nnsight.

KL_THRESHOLD = 0.1          # Arditi App. C
INDUCE_MIN = 0.0
LAYER_FRACTION = 0.8
INDUCE_FRAC_MIN = 0.5       # "half the harmless prompts flip" — the same onset rule as 26_recipe_dose
INDUCE_RULES = ("frac", "mean")
OBJECTIVES = ("mean", "frac")


# ---- one direction, applied everywhere (S3) ---------------------------------


def broadcast_direction(
    row: Tensor, n_layers: int, *, kind: str, position: str, model_name: str,
) -> Direction:
    """A [D] vector repeated across all layers as a [L, D] Direction.

    The repo's intervention code reads `direction.at(layer)`, i.e. it applies a
    DIFFERENT vector at each layer. Arditi applies ONE vector at every layer
    (equivalently: orthogonalizes every matrix that writes into the residual
    stream against it). Repeating the row here gets that behaviour with no
    change to interventions.py — and it matters: with 48 layer-local vectors a
    single contaminated coordinate is re-removed 48 times over.
    """
    assert row.ndim == 1, f"expected [D], got {tuple(row.shape)}"
    vec = row.detach().float().unsqueeze(0).expand(n_layers, -1).contiguous()
    return Direction(vec=vec, kind=kind, position=position, model_name=model_name)


# ---- readouts ---------------------------------------------------------------


def refusal_metric(logits: Tensor, tokenizer) -> Tensor:
    """Arditi's efficiency metric: logit of the summed refusal-opening mass.

    logits: [N, vocab] at the position where the response begins. Returns [N].
    Uses the repo's existing REFUSAL_PREFIX_WORDS ("I", "Sorry",
    "Unfortunately", "As") so this readout is the same one Table 1' reports as
    its log-prob twin.
    """
    from .readout import refusal_prefix_ids

    refusal_ids, _ = refusal_prefix_ids(tokenizer)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    lp = torch.logsumexp(logprobs[:, refusal_ids], dim=-1)        # log p
    # logit(p) = log p - log(1 - p), stable via log1p(-exp(lp))
    lp = lp.clamp(max=-1e-6)
    return lp - torch.log1p(-lp.exp())


def first_token_kl(clean_logits: Tensor, treated_logits: Tensor) -> Tensor:
    """KL(P_clean || P_treated) per row, over the full next-token distribution."""
    lp_c = torch.log_softmax(clean_logits.float(), dim=-1)
    lp_t = torch.log_softmax(treated_logits.float(), dim=-1)
    return (lp_c.exp() * (lp_c - lp_t)).sum(-1)


# ---- dose sweep (26_recipe_dose, prediction 4) ------------------------------


def induced_fraction(metric: Tensor) -> float:
    """Share of prompts whose refusal-opening mass exceeds 1/2 (logit > 0)."""
    return float((metric > 0).float().mean())


def dose_onset(multiples: list[float], fractions: list[float],
               threshold: float = 0.5) -> float | None:
    """Smallest dose multiple at which the induced-refusal fraction reaches
    `threshold`; None if the sweep never gets there.

    Reported as a grid cell, not interpolated: the claim under test is a
    scale ("~1x perp_norm, not ~2-5x"), and a log2 grid resolves that without
    pretending to more precision than 32 val prompts carry. Requires an
    ascending grid so "first crossing" is well defined.
    """
    assert len(multiples) == len(fractions), (len(multiples), len(fractions))
    assert all(a < b for a, b in zip(multiples, multiples[1:])), "multiples must ascend"
    for m, f in zip(multiples, fractions):
        if f >= threshold:
            return float(m)
    return None


def perp_fraction(row: Tensor, channels: list[int]) -> float:
    """b = ||row with the nuisance coordinates zeroed|| / ||row||: the share of
    the unit vector that points anywhere except the bias coordinates. The
    notes' algebra predicts a contaminated vector's onset at ~1/b x perp_norm
    and a cleaned one at ~1x, so 1/b is the quantitative version of
    prediction 4 for each arm."""
    r = row.float()
    n = float(r.norm())
    if n < 1e-12:
        return 0.0
    keep = r.clone()
    if channels:
        keep[channels] = 0.0
    return float(keep.norm()) / n


# ---- candidate scoring ------------------------------------------------------


@dataclasses.dataclass
class Candidate:
    recipe: str
    position: str
    layer: int
    norm: float
    bypass: float = float("nan")     # lower = ablation suppresses refusal more
    induce: float = float("nan")     # higher = addition induces refusal
    kl: float = float("nan")         # lower = intervention is specific
    bypass_delta: float = float("nan")   # vs clean harmful
    induce_delta: float = float("nan")   # vs clean harmless
    # Robust readouts. The mean logit is hostage to bf16 tails: on Gemma a
    # complying prompt sits at ~-35 nats, so one such prompt drags the mean of
    # eight below zero however many others flipped. The fraction with
    # P(refusal-prefix) > 1/2 is the prompt-level form of Arditi's intent and
    # the onset rule stage 04_dose uses; the median is the robust central value.
    bypass_frac: float = float("nan")    # share of harmful prompts still opening with a refusal prefix
    bypass_median: float = float("nan")
    induce_frac: float = float("nan")    # share of harmless prompts flipped, at alpha = ||row||
    induce_median: float = float("nan")
    # The same addition at a COMMON dose (alpha = perp_norm at the layer) so
    # recipes are compared at equal effective doses: ||row|| is the raw norm
    # for R0/R2 (R2 is renormalised to it) but the masked norm for R1, and
    # the induced-refusal response is non-monotone in alpha.
    induce_common: float = float("nan")
    induce_common_frac: float = float("nan")
    induce_common_median: float = float("nan")
    common_alpha: float = float("nan")
    feasible: bool = False
    reject: str = ""
    feasible_arditi: bool = False        # the original rule: mean logit > 0 at alpha = ||row||
    reject_arditi: str = ""
    induce_passed_at: str = ""           # "row" | "common" | "row+common" | ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def score_candidate(
    model,
    row: Tensor,
    layer: int,
    *,
    pps_harmful: list,
    pps_harmless: list,
    clean_harmless_logits: Tensor,
    clean_harmful_refusal: float,
    clean_harmless_refusal: float,
    recipe: str,
    position: str,
    batch_size: int = 32,
    common_alpha: float | None = None,
) -> Candidate:
    """Three forward passes: ablate@harmful, ablate@harmless, add@harmless —
    plus a fourth (add@harmless at `common_alpha`) when a common dose is given."""
    from .interventions import logits_with_intervention

    n_layers = model.n_layers
    all_layers = list(range(n_layers))
    d = broadcast_direction(row, n_layers, kind=f"cand|{recipe}",
                            position=position, model_name=model.cfg.name)
    norm = float(row.norm())

    abl_harmful = logits_with_intervention(
        model, pps_harmful, direction=d, layers=all_layers,
        position="all", mode="ablate", batch_size=batch_size)
    abl_harmless = logits_with_intervention(
        model, pps_harmless, direction=d, layers=all_layers,
        position="all", mode="ablate", batch_size=batch_size)
    # addition: the RAW vector at its own layer only (Arditi's implicit 1x dose)
    add_harmless = logits_with_intervention(
        model, pps_harmless, direction=d, layers=[layer],
        position="all", mode="steer", alpha=norm, batch_size=batch_size)

    m_bypass = refusal_metric(abl_harmful, model.tokenizer)
    m_induce = refusal_metric(add_harmless, model.tokenizer)
    bypass = float(m_bypass.mean())
    induce = float(m_induce.mean())
    kl = float(first_token_kl(clean_harmless_logits, abl_harmless).mean())
    cand = Candidate(
        recipe=recipe, position=position, layer=layer, norm=norm,
        bypass=bypass, induce=induce, kl=kl,
        bypass_delta=bypass - clean_harmful_refusal,
        induce_delta=induce - clean_harmless_refusal,
        bypass_frac=induced_fraction(m_bypass), bypass_median=float(m_bypass.median()),
        induce_frac=induced_fraction(m_induce), induce_median=float(m_induce.median()),
    )
    if common_alpha is not None and common_alpha > 0:
        add_common = logits_with_intervention(
            model, pps_harmless, direction=d, layers=[layer],
            position="all", mode="steer", alpha=float(common_alpha), batch_size=batch_size)
        m_common = refusal_metric(add_common, model.tokenizer)
        cand.induce_common = float(m_common.mean())
        cand.induce_common_frac = induced_fraction(m_common)
        cand.induce_common_median = float(m_common.median())
        cand.common_alpha = float(common_alpha)
    return cand


def apply_constraints(
    cands: list[Candidate],
    n_layers: int,
    *,
    kl_threshold: float = KL_THRESHOLD,
    induce_min: float = INDUCE_MIN,
    layer_fraction: float = LAYER_FRACTION,
    induce_rule: str = "frac",
    induce_frac_min: float = INDUCE_FRAC_MIN,
) -> list[Candidate]:
    """Mark feasibility in place and return the list (reason recorded on reject).

    Reasons are kept because the interesting result may be that EVERY raw
    candidate is rejected for KL — that is the quantitative form of "the
    original vector is a global bias, not a feature".

    Two induce rules, BOTH always recorded (`feasible` / `feasible_arditi`):
      mean   Arditi's: mean refusal logit > induce_min at alpha = ||row||
      frac   at least `induce_frac_min` of the harmless prompts flip
             (P(refusal-prefix) > 1/2) at alpha = ||row|| OR at the common
             dose, when one was measured — the prompt-level form of the same
             criterion, immune to a single -35-nat prompt vetoing a cell
    `induce_rule` decides which one sets `feasible`; `induce_passed_at` says
    which dose carried it.
    """
    assert induce_rule in INDUCE_RULES, induce_rule
    max_layer = layer_fraction * n_layers
    for c in cands:
        base = []
        if not (c.layer < max_layer):
            base.append("layer")
        if not (c.kl < kl_threshold):
            base.append("kl")
        ok_mean = c.induce > induce_min
        at = []
        if c.induce_frac >= induce_frac_min:
            at.append("row")
        if c.induce_common_frac == c.induce_common_frac and c.induce_common_frac >= induce_frac_min:
            at.append("common")
        c.induce_passed_at = "+".join(at)
        ok_frac = bool(at)
        c.reject_arditi = "+".join(base + ([] if ok_mean else ["induce"]))
        c.feasible_arditi = not c.reject_arditi
        ok = ok_frac if induce_rule == "frac" else ok_mean
        c.reject = "+".join(base + ([] if ok else ["induce"]))
        c.feasible = not c.reject
    return cands


def select(cands: list[Candidate], objective: str = "mean",
           feasible_attr: str = "feasible") -> Candidate | None:
    """Minimum bypass among feasible candidates; None if the grid is empty.

    objective "mean": Arditi's min mean refusal logit (pre-registered);
    "frac": min share of harmful prompts still refusing, mean as tie-break.
    `feasible_attr` selects which feasibility flag to honour."""
    assert objective in OBJECTIVES, objective
    ok = [c for c in cands if getattr(c, feasible_attr)]
    if not ok:
        return None
    if objective == "frac":
        return min(ok, key=lambda c: (c.bypass_frac, c.bypass))
    return min(ok, key=lambda c: c.bypass)


def selection_report(cands: list[Candidate], n_layers: int, objective: str = "mean") -> dict:
    """Grid summary: what was chosen, and why everything else was not."""
    from collections import Counter

    by_recipe: dict[str, dict] = {}
    for recipe in sorted({c.recipe for c in cands}):
        sub = [c for c in cands if c.recipe == recipe]
        best = select(sub, objective)
        best_frac = select(sub, "frac")
        best_arditi = select(sub, "mean", feasible_attr="feasible_arditi")
        by_recipe[recipe] = {
            "n_candidates": len(sub),
            "n_feasible": sum(c.feasible for c in sub),
            "reject_reasons": dict(Counter(c.reject for c in sub if c.reject)),
            "min_kl": min(c.kl for c in sub),
            "median_kl": float(sorted(c.kl for c in sub)[len(sub) // 2]),
            "selected": best.as_dict() if best else None,
            # the alternatives, always visible: the frac-objective pick under
            # the same feasibility, and the original Arditi rule end to end
            "selected_by_frac": best_frac.as_dict() if best_frac else None,
            "n_feasible_arditi": sum(c.feasible_arditi for c in sub),
            "reject_reasons_arditi": dict(Counter(c.reject_arditi for c in sub if c.reject_arditi)),
            "selected_arditi": best_arditi.as_dict() if best_arditi else None,
            "induce_passed_at": dict(Counter(c.induce_passed_at for c in sub if c.induce_passed_at)),
        }
    return {"n_layers": n_layers, "objective": objective, "by_recipe": by_recipe,
            "candidates": [c.as_dict() for c in cands]}
