"""Selection readouts and the constraint/argmin rule — no model needed."""

import pytest
import torch

from loudchannel.selection import (
    Candidate,
    apply_constraints,
    broadcast_direction,
    first_token_kl,
    refusal_metric,
    select,
    selection_report,
)


class _Tok:
    """Minimal tokenizer stub: encodes the refusal/accept prefix words to
    distinct single ids, which is all refusal_prefix_ids needs."""

    VOCAB = {"I": 1, " I": 1, "Sorry": 2, " Sorry": 2, "Unfortunately": 3,
             " Unfortunately": 3, "As": 4, " As": 4, "Sure": 5, " Sure": 5,
             "Here": 6, " Here": 6, "Certainly": 7, " Certainly": 7,
             "Okay": 8, " Okay": 8, "Of": 9, " Of": 9, "To": 10, " To": 10}

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB[text]] if text in self.VOCAB else [11, 12]


def test_broadcast_direction_is_one_vector_everywhere():
    row = torch.randn(16)
    d = broadcast_direction(row, 5, kind="k", position="t_inst", model_name="m")
    assert d.vec.shape == (5, 16)
    for li in range(5):
        assert torch.allclose(d.at(li, unit=True), row / row.norm(), atol=1e-6)


def test_refusal_metric_is_monotone_in_refusal_mass():
    vocab = 32
    lo, hi = torch.zeros(1, vocab), torch.zeros(1, vocab)
    hi[0, [1, 2, 3, 4]] = 8.0            # pile mass onto the refusal openers
    a = float(refusal_metric(lo, _Tok())[0])
    b = float(refusal_metric(hi, _Tok())[0])
    assert b > a
    assert torch.isfinite(torch.tensor([a, b])).all()


def test_refusal_metric_is_finite_at_saturation():
    """logit(p) diverges as p -> 1; the clamp must keep it finite, otherwise a
    single saturated candidate poisons the argmin."""
    vocab = 32
    x = torch.full((1, vocab), -60.0)
    x[0, 1] = 60.0
    assert torch.isfinite(refusal_metric(x, _Tok())).all()


def test_kl_is_zero_for_identical_and_positive_otherwise():
    a = torch.randn(4, 50)
    assert torch.allclose(first_token_kl(a, a), torch.zeros(4), atol=1e-6)
    assert (first_token_kl(a, torch.randn(4, 50)) > 0).all()


def _c(recipe, layer, bypass, induce, kl):
    # induce > 0 stands for "the harmless prompts flip" in these constraint
    # tests, so mirror it into the fraction readout (the new default rule) and
    # the bypass fraction so both rules agree on these synthetic cells
    return Candidate(recipe=recipe, position="t_post_inst", layer=layer,
                     norm=1.0, bypass=bypass, induce=induce, kl=kl,
                     induce_frac=1.0 if induce > 0 else 0.0,
                     bypass_frac=0.0, bypass_median=bypass)


def test_constraints_reject_for_the_right_reason():
    n_layers = 10
    cands = [
        _c("r0_raw", 2, -5.0, 1.0, 0.5),    # great bypass, fails KL  <- the trap
        _c("r0_raw", 3, -1.0, 1.0, 0.01),   # feasible
        _c("r0_raw", 4, -9.0, -1.0, 0.01),  # fails induce
        _c("r0_raw", 9, -9.0, 1.0, 0.01),   # fails layer prune (9 >= 0.8*10)
    ]
    apply_constraints(cands, n_layers)
    assert [c.reject for c in cands] == ["kl", "", "induce", "layer"]
    best = select(cands)
    assert best is not None and best.layer == 3, "argmin ignored the constraints"


def test_select_returns_none_when_everything_is_rejected():
    cands = [_c("r0_raw", 1, -9.0, 1.0, 5.0), _c("r0_raw", 2, -8.0, 1.0, 9.0)]
    apply_constraints(cands, 10)
    assert select(cands) is None
    rep = selection_report(cands, 10)
    assert rep["by_recipe"]["r0_raw"]["selected"] is None
    assert rep["by_recipe"]["r0_raw"]["reject_reasons"] == {"kl": 2}


def test_report_is_per_recipe():
    cands = [_c("r0_raw", 1, -1.0, 1.0, 9.0), _c("r1_masked", 1, -2.0, 1.0, 0.01)]
    apply_constraints(cands, 10)
    rep = selection_report(cands, 10)
    assert rep["by_recipe"]["r0_raw"]["selected"] is None
    assert rep["by_recipe"]["r1_masked"]["selected"]["layer"] == 1


def test_dose_onset_is_first_grid_crossing_or_none():
    from loudchannel.selection import dose_onset

    m = [0.25, 0.5, 1.0, 2.0, 4.0]
    assert dose_onset(m, [0.0, 0.1, 0.6, 0.9, 1.0]) == 1.0
    assert dose_onset(m, [0.0, 0.5, 0.6, 0.9, 1.0]) == 0.5      # >= threshold counts
    assert dose_onset(m, [0.0, 0.1, 0.2, 0.3, 0.4]) is None
    assert dose_onset(m, [0.0, 0.1, 0.2, 0.3, 0.4], threshold=0.3) == 2.0
    with pytest.raises(AssertionError):
        dose_onset([1.0, 0.5], [0.0, 1.0])                        # not ascending


def test_induced_fraction_counts_positive_logits():
    from loudchannel.selection import induced_fraction

    assert induced_fraction(torch.tensor([-1.0, 0.5, 2.0, -0.1])) == 0.5
    assert induced_fraction(torch.tensor([0.0, 0.0])) == 0.0         # logit 0 is p=1/2, not > 1/2


def test_perp_fraction_matches_the_notes_algebra():
    """d_hat = a e_c* + b u  ->  perp_fraction is b; a contaminated vector with
    84.5% of its energy on one coordinate has b = sqrt(0.155) ~ 0.39."""
    from loudchannel.selection import perp_fraction

    row = torch.zeros(100)
    row[7] = 3.0                      # the bias coordinate
    row[:5] = 1.0                     # 5 units of energy elsewhere
    b = perp_fraction(row, [7])
    assert abs(b - (5 ** 0.5) / (14 ** 0.5)) < 1e-6
    assert perp_fraction(row, []) == 1.0                      # nothing masked: all perp
    assert perp_fraction(torch.zeros(100), [7]) == 0.0        # dead row, no NaN


def test_frac_rule_ignores_a_single_negative_mean_prompt():
    """The point of the fraction rule: a cell where 5/8 harmless prompts flip
    but one -35-nat prompt drags the mean below zero must pass the frac rule
    and fail the Arditi mean rule — both recorded."""
    from loudchannel.selection import Candidate, apply_constraints

    c = Candidate(recipe="r1_masked", position="t_post_inst", layer=3, norm=1.0,
                  bypass=-9.0, induce=-4.0, kl=0.01,      # mean < 0 (a bf16 tail)
                  induce_frac=0.625, bypass_frac=0.1, bypass_median=-9.0)
    apply_constraints([c], 10)                             # default: frac rule
    assert c.feasible and c.reject == ""                  # frac >= 0.5 -> passes
    assert not c.feasible_arditi and c.reject_arditi == "induce"
    assert c.induce_passed_at == "row"
    apply_constraints([c], 10, induce_rule="mean")
    assert not c.feasible and c.reject == "induce"        # mean rule vetoes it


def test_common_dose_can_carry_induce_when_the_row_dose_overshoots():
    from loudchannel.selection import Candidate, apply_constraints

    c = Candidate(recipe="r1_masked", position="t_post_inst", layer=3, norm=1.0,
                  bypass=-9.0, induce=-2.0, kl=0.01,
                  induce_frac=0.2,                          # ||row|| dose overshot
                  induce_common_frac=0.75,                  # common dose is in the window
                  bypass_frac=0.1, bypass_median=-9.0)
    apply_constraints([c], 10)
    assert c.feasible and c.induce_passed_at == "common"


def test_report_records_both_rules_and_the_frac_objective():
    from loudchannel.selection import Candidate, apply_constraints, selection_report

    def cc(layer, bypass, bypass_frac, induce_frac, kl, induce_mean):
        return Candidate(recipe="r1_masked", position="t_post_inst", layer=layer,
                         norm=1.0, bypass=bypass, induce=induce_mean, kl=kl,
                         induce_frac=induce_frac, bypass_frac=bypass_frac, bypass_median=bypass)
    cands = [cc(1, -9.0, 0.1, 0.6, 0.01, -4.0),   # frac-feasible, mean-infeasible; best bypass
             cc(2, -3.0, 0.0, 0.6, 0.01, +1.0)]   # feasible under both; best bypass_frac
    apply_constraints(cands, 10)
    rep = selection_report(cands, 10)
    r = rep["by_recipe"]["r1_masked"]
    assert r["n_feasible"] == 2 and r["n_feasible_arditi"] == 1
    assert r["selected"]["layer"] == 1               # min bypass (mean objective)
    assert r["selected_by_frac"]["layer"] == 2       # min bypass_frac
    assert r["selected_arditi"]["layer"] == 2        # only cell passing the mean rule
    assert r["induce_passed_at"].get("row", 0) == 2
