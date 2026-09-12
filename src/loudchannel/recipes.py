"""Estimator variants for difference-of-means directions (Stage D + R).

Why this module exists
----------------------
`extract.diff_of_means` gives d = mean(harmful) - mean(harmless) per layer.
That estimator weights coordinate c by

    s_c = d_c^2 / ||d||^2 = sigma_c^2 * d'_c^2 / sum_k sigma_k^2 * d'_k^2

i.e. by *variance x discriminability^2* — so a coordinate that is merely huge
can dominate the vector without carrying any class evidence. On Gemma-3-12B a
single residual coordinate (2339, a bias-like "massive activation") holds
91-96% of ||d_refuse||^2 in layers 13-19 while its own class gap is
statistically zero (|t| < 2) and partly a prompt-length artifact. Ablating such
a vector is not a feature removal: it collapses the RMSNorm denominator (the
other coordinates get amplified 8-21x) and leaks -a*b*M along the vector's
remainder. Hence NLL x7.76, MMLU 0.68 -> 0.22 and degenerate generations.

The recipes below are the candidate fixes. All of them are post-hoc transforms
of ONE cached extraction pass, so the whole grid is free after extraction.

    R0  raw            baseline: the current recipe
    R1  masked         zero the nuisance coordinates, then difference
    R2  standardized   Delta_c / (sigma_c^2 + lambda)  (diagonal LDA)
    R3  winsorized     clip activations at a quantile, then difference
    R4  length-matched resample so mean prompt length matches, then difference
    R5  covariate-adj. per-coordinate OLS class coefficient holding word count
                       and terminal punctuation fixed (the two covariates the
                       nominal-error analysis found to carry the gap); the
                       conditional gap rather than the marginal one

Design rules that keep this honest:
  * the nuisance set is chosen from CLASS-BLIND statistics (harmless split
    only), never from the direction itself — otherwise we would remove signal
    by construction;
  * every recipe is applied to every model, so "it changes nothing on Llama and
    Qwen" is a measured result rather than an assumption;
  * R2 is expected to be the best *readout* and NOT the best intervention
    vector (Sigma^-1 Delta rescales the feature's own coordinates, so it is a
    biased estimate of the direction the model writes along). Pre-registered.
"""

from __future__ import annotations


import torch
from torch import Tensor

from .directions import Direction

# `extract_multi` needs the model stack; everything else in this module is pure
# tensor maths on cached activations. Those imports are therefore deferred into
# the function so the recipes and diagnostics stay importable (and unit
# testable) on a box with no nnsight/transformers install.

# Arditi et al. (2024) select over post-instruction token positions; the repo's
# PromptPositions.get() resolves "<name><+/-offset>", so the last five template
# tokens are t_post_inst-4 .. t_post_inst. t_inst is Zhao et al.'s position and
# is kept so R0 reproduces the original recipe inside the same grid.
CANDIDATE_POSITIONS: tuple[str, ...] = (
    "t_inst",
    "t_post_inst-4", "t_post_inst-3", "t_post_inst-2", "t_post_inst-1",
    "t_post_inst",
)

RECIPES: tuple[str, ...] = ("r0_raw", "r1_masked", "r2_standardized",
                            "r3_winsorized", "r4_length_matched",
                            "r5_covariate_adjusted")


# ---- extraction (one pass, every candidate position) ------------------------


def extract_multi(
    model,
    instructions: list[str],
    positions: tuple[str, ...] = CANDIDATE_POSITIONS,
    *,
    batch_size: int = 16,
    system: str | None = None,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Hidden states at EVERY named position, in a single forward pass per batch.

    Returns {position: float32 CPU tensor [n_prompts, n_layers, d_model]}.

    This is `extract.extract_activations` generalized over positions: the whole
    Arditi candidate grid needs |I| positions x L layers, and running one
    forward pass per position would multiply the extraction cost by |I| for no
    reason. Same NDIF-whitelist discipline as extract.py — the trace body must
    not reference loudchannel objects, so envoys and index tensors are hoisted.
    """
    from .positions import positions_for
    from .remote import with_retries

    layers = list(range(model.n_layers))
    pps = positions_for(model, instructions, system=system, seed=seed)
    tup = model.output_is_tuple  # probe BEFORE opening traces
    layer_envoys = [model.layers[li] for li in layers]
    n_pos = len(positions)

    from tqdm import tqdm

    rows: list[Tensor] = []
    for start in tqdm(range(0, len(pps), batch_size), desc="extract@multi"):
        chunk = pps[start : start + batch_size]
        prompts = [p.prompt for p in chunk]
        b = torch.arange(len(chunk))
        # [n_pos, B] — one index row per candidate position
        idx = torch.stack([torch.tensor([p.get(name) for p in chunk])
                           for name in positions])

        def run_batch():
            with model.trace(prompts):
                saved = list().save()
                for env in layer_envoys:
                    out = env.output
                    h = out[0] if tup else out       # [B, T, D]
                    for pi in range(n_pos):
                        saved.append(h[b, idx[pi]].detach().cpu())
            return saved

        saved = with_retries(run_batch, what=f"extract@multi batch {start}")
        rows.append(regroup(saved, len(layers), n_pos, len(chunk)))

    cat = torch.cat(rows, dim=1)                                   # [n_pos, N, L, D]
    return {name: cat[i].contiguous() for i, name in enumerate(positions)}


def regroup(saved: list, n_layers: int, n_pos: int, n_batch: int) -> Tensor:
    """[L*n_pos] tensors of [B, D], appended (layer-major, position-minor),
    -> [n_pos, B, L, D].

    Factored out of the trace loop because getting this transpose wrong is
    silent: every downstream number would still be computed, just from
    activations attributed to the wrong position. tests/test_recipes.py pins it
    against an explicitly-constructed reference.
    """
    assert len(saved) == n_layers * n_pos, (len(saved), n_layers, n_pos)
    stacked = torch.stack([s.float() for s in saved])              # [L*n_pos, B, D]
    stacked = stacked.reshape(n_layers, n_pos, n_batch, -1)
    return stacked.permute(1, 2, 0, 3).contiguous()                # [n_pos, B, L, D]


# ---- D3: class-blind nuisance coordinates -----------------------------------


def nuisance_channels(
    acts_neg: Tensor,
    *,
    ratio: float = 50.0,
    max_k: int = 16,
) -> dict[int, list[int]]:
    """Coordinates whose |mean| on the HARMLESS split dwarfs the layer median.

    acts_neg: [n, L, D] activations of the harmless class only. Using one class
    keeps the choice independent of the harmful/harmless contrast we are about
    to estimate; picking coordinates by looking at the difference would be
    circular ("remove whatever the direction points at").

    A coordinate is nuisance at layer l iff
        |mean_n x_c| > ratio * median_k |mean_n x_k|
    capped at the `max_k` largest, so a model with no massive activations
    (Llama, Qwen) yields empty or near-empty sets and R1 degenerates to R0.
    """
    assert acts_neg.ndim == 3, f"expected [n, L, D], got {tuple(acts_neg.shape)}"
    mu = acts_neg.float().mean(0)                          # [L, D]
    med = mu.abs().median(dim=-1, keepdim=True).values     # [L, 1]
    score = mu.abs() / med.clamp_min(1e-12)
    out: dict[int, list[int]] = {}
    for l in range(mu.shape[0]):
        idx = torch.where(score[l] > ratio)[0]
        if len(idx) > max_k:                                # keep the loudest
            order = score[l][idx].argsort(descending=True)[:max_k]
            idx = idx[order]
        out[l] = sorted(int(c) for c in idx)
    return out


def mask_channels(vec: Tensor, channels: dict[int, list[int]]) -> Tensor:
    """Zero the given per-layer coordinates of a [L, D] tensor (copy)."""
    out = vec.clone()
    for l, cs in channels.items():
        if cs:
            out[l, cs] = 0.0
    return out


# ---- R3: winsorization ------------------------------------------------------


def winsorize(acts: Tensor, *, q: float = 0.995, axis: str = "within_vector") -> Tensor:
    """Clip extreme magnitudes before averaging.

    axis="within_vector"     quantile of |x| across COORDINATES, per (prompt,
                             layer). A coordinate that is huge on every prompt
                             is clipped on every prompt, so its class gap
                             collapses -> behaves like R1.
    axis="within_coordinate" quantile of |x| across PROMPTS, per (layer,
                             coordinate). A uniformly-huge coordinate is NOT
                             clipped (its 99.5th percentile is itself), so this
                             axis does NOT fix the massive-activation case.

    Both are implemented because the practitioner write-up this recipe comes
    from (grimjim, "Projected Abliteration") does not say which axis it used,
    and the two have opposite effects. Default is the one that works.

    LIMITATION worth knowing before preferring R3 to R1: "within_vector" only
    clips a nuisance coordinate if the nuisance coordinates are a smaller
    fraction of D than (1 - q). At D = 3840 and q = 0.995 the cap is the ~19th
    largest coordinate, so one or two massive coordinates are clipped hard; but
    a model carrying >19 of them would set the cap INSIDE the massive group and
    clip nothing that matters. R1 has no such failure mode because it names the
    coordinates. `_winsorize_headroom` reports the margin so a run can assert
    it rather than assume it.
    """
    assert axis in ("within_vector", "within_coordinate"), axis
    x = acts.float()
    out = torch.empty_like(x)
    n, L, _ = x.shape
    for l in range(L):                       # loop keeps torch.quantile in range
        sl = x[:, l, :]                      # [n, D]
        if axis == "within_vector":
            cap = torch.quantile(sl.abs(), q, dim=-1, keepdim=True)   # [n, 1]
        else:
            cap = torch.quantile(sl.abs(), q, dim=0, keepdim=True)    # [1, D]
        out[:, l, :] = sl.clamp(min=-cap, max=cap)
    return out


# ---- R4: length matching ----------------------------------------------------


def winsorize_headroom(acts: Tensor, channels: dict[int, list[int]],
                       *, q: float = 0.995, min_reduction: float = 0.5) -> dict:
    """Will within-vector Winsorization actually clip the nuisance coordinates?

    Measures the cap directly rather than reasoning about quantile arithmetic:
    per layer, the median within-vector cap against the median |value| of the
    nuisance coordinates, reported as `reduction` = 1 - cap/magnitude. A cap
    that merely shaves a few percent off a massive coordinate is a no-op
    dressed as a fix, so `effective` requires the cap to remove at least
    `min_reduction` of it. This fails when the nuisance coordinates are
    numerous enough (relative to (1-q)*D) that the quantile lands inside them —
    R1 has no such failure mode because it names the coordinates.
    """
    x = acts.float()
    per_layer, ok = {}, True
    for l in range(x.shape[1]):
        cs = channels.get(l, [])
        sl = x[:, l, :]
        cap = float(torch.quantile(sl.abs(), q, dim=-1).median())
        mag = float(sl[:, cs].abs().median()) if cs else 0.0
        reduction = (1.0 - cap / mag) if mag > 0 else 0.0
        clipped = bool(cs) and reduction >= min_reduction
        per_layer[l] = {"cap": cap, "nuisance_magnitude": mag,
                        "reduction": reduction, "n_nuisance": len(cs),
                        "clipped": clipped}
        if cs and not clipped:
            ok = False
    return {"q": q, "min_reduction": min_reduction, "effective": ok,
            "per_layer": per_layer}


def length_matched_indices(
    len_pos: list[int], len_neg: list[int], *, seed: int = 0, tol: float = 0.25,
) -> tuple[list[int], list[int]]:
    """Greedy nearest-length pairing so the two classes share a length profile.

    The harmful splits (AdvBench/SORRY-Bench) run ~6.5 words longer than the
    harmless one (Alpaca), and on Gemma the massive coordinate moves ~-160 per
    word, which manufactures a class gap of ~1000 out of nothing. This removes
    the confound in the DATA rather than in the vector, so it is the cleanest
    control for that specific alternative explanation.

    Returns (idx_pos, idx_neg) of equal length; pairs whose lengths differ by
    more than `tol` (relative) are dropped.
    """
    import random

    rng = random.Random(seed)
    pool = sorted(range(len(len_neg)), key=lambda i: len_neg[i])
    lens = sorted(len_neg)
    order = list(range(len(len_pos)))
    rng.shuffle(order)
    keep_p, keep_n = [], []
    used = set()
    for i in order:
        target = len_pos[i]
        best, best_d = None, None
        for j_rank, j in enumerate(pool):
            if j in used:
                continue
            d = abs(lens[j_rank] - target)
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is None:
            break
        if best_d <= tol * max(target, 1):
            used.add(best)
            keep_p.append(i)
            keep_n.append(best)
    return keep_p, keep_n


# ---- the recipes ------------------------------------------------------------


def build_direction(
    recipe: str,
    acts_pos: Tensor,
    acts_neg: Tensor,
    *,
    channels: dict[int, list[int]] | None = None,
    shrinkage: float = 1e-3,
    q: float = 0.995,
    wins_axis: str = "within_vector",
    idx_pos: list[int] | None = None,
    idx_neg: list[int] | None = None,
    cov_pos: Tensor | None = None,
    cov_neg: Tensor | None = None,
) -> Tensor:
    """One [L, D] direction (raw scale, NOT unit-normalized) per recipe.

    Raw scale is kept because Arditi's activation-addition arm adds the
    unnormalized difference-of-means vector (an implicit 1x local dose); the
    unit vector and an explicit alpha are recovered downstream.
    """
    p, n = acts_pos.float(), acts_neg.float()

    if recipe == "r0_raw":
        return p.mean(0) - n.mean(0)

    if recipe == "r1_masked":
        assert channels is not None, "r1_masked needs the nuisance channel set"
        return mask_channels(p.mean(0) - n.mean(0), channels)

    if recipe == "r2_standardized":
        delta = p.mean(0) - n.mean(0)
        var = 0.5 * (p.var(0, unbiased=True) + n.var(0, unbiased=True))
        # shrink toward the layer's median variance so near-constant
        # coordinates cannot explode; lambda is relative, not absolute.
        floor = shrinkage * var.median(dim=-1, keepdim=True).values
        w = delta / (var + floor).clamp_min(1e-12)
        # rescale per layer to the raw vector's norm: the DIRECTION is what
        # differs between recipes, the dose is calibrated separately.
        return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12) * \
            delta.norm(dim=-1, keepdim=True)

    if recipe == "r3_winsorized":
        return (winsorize(p, q=q, axis=wins_axis).mean(0)
                - winsorize(n, q=q, axis=wins_axis).mean(0))

    if recipe == "r4_length_matched":
        assert idx_pos is not None and idx_neg is not None, \
            "r4_length_matched needs matched index lists"
        return p[idx_pos].mean(0) - n[idx_neg].mean(0)

    if recipe == "r5_covariate_adjusted":
        # R4 removes the length confound by matching; R5 removes length AND
        # terminal punctuation by regression, on the full sample. Sampling
        # noise on a rho ~ 200 coordinate is NOT removed by either (see
        # nominal.py) — that part needs R1.
        assert cov_pos is not None and cov_neg is not None, \
            "r5_covariate_adjusted needs per-prompt covariates (nominal.prompt_covariates)"
        from .nominal import ols_class_gap
        return ols_class_gap(p, n, cov_pos[:, :2], cov_neg[:, :2])

    raise ValueError(f"unknown recipe {recipe!r}; known: {RECIPES}")


def as_direction(vec: Tensor, *, kind: str, position: str, model_name: str,
                 recipe: str, n_pos: int = 0, n_neg: int = 0) -> Direction:
    """Wrap a [L, D] tensor as a Direction, tagging the recipe in `kind`."""
    return Direction(vec=vec, kind=f"{kind}|{recipe}", position=position,
                     model_name=model_name, n_pos=n_pos, n_neg=n_neg)


# ---- D1 / D2 / D5: diagnostics ----------------------------------------------


def energy_shares(vec: Tensor, k: int = 8) -> dict:
    """D1 — how concentrated is the vector? [L, D] -> per-layer shares."""
    e = vec.float() ** 2
    tot = e.sum(-1).clamp_min(1e-30)
    srt, idx = e.sort(-1, descending=True)
    return {
        "top1_share": (srt[:, 0] / tot).tolist(),
        "topk_share": (srt[:, :k].sum(-1) / tot).tolist(),
        "top1_channel": idx[:, 0].tolist(),
        "topk_channels": idx[:, :k].tolist(),
        "norm": vec.float().norm(dim=-1).tolist(),
    }


def cos_to_raw(directions: dict[str, Tensor], raw: str = "r0_raw") -> dict[str, list[float]]:
    """Per-layer cosine of every recipe's [L, D] vector against the raw one.

    The no-op prediction for the control models is geometric for R1/R3/R4
    (cos > 0.95: the recipe barely moved the vector where there was nothing to
    remove) and behavioural only for R2 (standardisation rotates the vector on
    every model by construction, so its cosine is uninformative and must not
    be asserted on). Layers where either vector is ~0 (e.g. a fully masked
    layer) report 0.0 rather than NaN.
    """
    if raw not in directions:
        return {}
    r = directions[raw].float()
    out: dict[str, list[float]] = {}
    for name, v in directions.items():
        if name == raw:
            continue
        v = v.float()
        num = (r * v).sum(-1)
        den = (r.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-12)
        c = torch.where(den > 1e-12, num / den, torch.zeros_like(num))
        out[name] = c.tolist()
    return out


def evidence_stats(acts_pos: Tensor, acts_neg: Tensor) -> dict:
    """D2 — per-coordinate evidence (Cohen's d, Welch t) alongside its weight.

    Reported as summaries plus the per-layer profile of the single
    highest-energy coordinate, which is the one the whole argument turns on.
    """
    p, n = acts_pos.float(), acts_neg.float()
    delta = p.mean(0) - n.mean(0)
    vp, vn = p.var(0, unbiased=True), n.var(0, unbiased=True)
    sd = torch.sqrt(0.5 * (vp + vn)).clamp_min(1e-12)
    dprime = delta / sd
    se = torch.sqrt(vp / p.shape[0] + vn / n.shape[0]).clamp_min(1e-12)  # Welch
    t = delta / se
    e = delta ** 2
    share = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
    top = e.argmax(-1)                                    # [L]
    L = delta.shape[0]
    rows = []
    for l in range(L):
        c = int(top[l])
        rank = int((dprime[l].abs() > dprime[l, c].abs()).sum()) + 1
        rows.append({
            "layer": l, "channel": c,
            "energy_share": float(share[l, c]),
            "cohens_d": float(dprime[l, c]),
            "welch_t": float(t[l, c]),
            "dprime_rank": rank,                # 1 = most discriminative
            "mean_pos": float(p[:, l, c].mean()),
            "mean_neg": float(n[:, l, c].mean()),
            "pooled_sd": float(sd[l, c]),
        })
    return {
        "top_energy_channel_per_layer": rows,
        "max_abs_dprime_per_layer": dprime.abs().max(-1).values.tolist(),
    }


def norm_decomposition(acts: Tensor, channels: dict[int, list[int]]) -> dict:
    """D5 — what ablating the nuisance set would do to the RMSNorm denominator.

    g = rms(x) / rms(x without the nuisance coordinates) is the factor by which
    every OTHER coordinate is amplified downstream when a vector aligned with
    those coordinates is projected out. g ~ 1 means ablation is safe; Gemma
    measures 8-21 in its entangled band.
    """
    x = acts.float()
    out = {"rms_full": [], "rms_wo": [], "gain": [], "n_channels": []}
    for l in range(x.shape[1]):
        cs = channels.get(l, [])
        sl = x[:, l, :]
        rms_full = sl.pow(2).mean(-1).sqrt()
        if cs:
            keep = sl.clone()
            keep[:, cs] = 0.0
        else:
            keep = sl
        rms_wo = keep.pow(2).mean(-1).sqrt().clamp_min(1e-12)
        out["rms_full"].append(float(rms_full.median()))
        out["rms_wo"].append(float(rms_wo.median()))
        out["gain"].append(float((rms_full / rms_wo).median()))
        out["n_channels"].append(len(cs))
    return out


def perp_norm(acts: Tensor, channels: dict[int, list[int]]) -> list[float]:
    """||x_perp|| per layer (median over prompts) — the dose scale that
    matters once the nuisance coordinates are excluded (I4)."""
    x = acts.float()
    out = []
    for l in range(x.shape[1]):
        sl = x[:, l, :].clone()
        cs = channels.get(l, [])
        if cs:
            sl[:, cs] = 0.0
        out.append(float(sl.norm(dim=-1).median()))
    return out
