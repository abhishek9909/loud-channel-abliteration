"""D7 weight-level provenance check on toy module trees — no model download.

Two toy architectures, both with the SAME planted channel:

  gemma_like   four d_model norms per layer with the (1 + w) convention; the
               planted channel has a huge fixed-sign post-FFN (writer) gain
               and a ~0 reader gain — the residual-sink mechanism;
  llama_like   two norms per layer, plain w convention, nothing planted —
               the control, where the report must single out nothing.

The assertions are the audit's predictions, so a model that violates them
would fail these tests only if the toy were mis-specified — on a real model
the same flags are reported, never asserted.
"""

import json

import pytest
import torch
from torch import nn

from loudchannel.norm_gains import (
    channel_stats,
    classify_norms,
    find_layout,
    format_report,
    gain_offset,
    hf_module,
    norm_gain_report,
    top_channels,
    verdict,
)

D, L, V = 64, 6, 50
C = 7          # the planted "massive" channel


class GemmaLikeRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * (1.0 + self.weight.float())


class LlamaLikeRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _Layer(nn.Module):
    def __init__(self, norm_cls, names, dim):
        super().__init__()
        for n in names:
            setattr(self, n, norm_cls(dim))
        self.self_attn = nn.Module()
        self.self_attn.q_norm = norm_cls(8)        # head_dim-sized: must be ignored


class _Inner(nn.Module):
    def __init__(self, norm_cls, names):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, D)
        self.layers = nn.ModuleList([_Layer(norm_cls, names, D) for _ in range(L)])
        self.norm = norm_cls(D)


class _HF(nn.Module):
    def __init__(self, norm_cls, names):
        super().__init__()
        self.model = _Inner(norm_cls, names)

    def get_input_embeddings(self):
        return self.model.embed_tokens


class _Wrapper:
    """Mimics HarmModel: .lm._model is the HF module, .tokenizer has a BOS id."""

    def __init__(self, hf):
        self.lm = type("LM", (), {})()
        self.lm._model = hf
        self.tokenizer = type("Tok", (), {"bos_token_id": 2})()


GEMMA_NORMS = ["input_layernorm", "post_attention_layernorm",
               "pre_feedforward_layernorm", "post_feedforward_layernorm"]
LLAMA_NORMS = ["input_layernorm", "post_attention_layernorm"]


def _gemma_like():
    hf = _HF(GemmaLikeRMSNorm, GEMMA_NORMS)
    with torch.no_grad():
        for layer in hf.model.layers:
            layer.post_feedforward_layernorm.weight[C] = 40.0    # writer gain 41
            layer.post_attention_layernorm.weight[C] = 5.0
            layer.input_layernorm.weight[C] = -1.0               # reader gain 0 (hidden)
            layer.pre_feedforward_layernorm.weight[C] = -1.0
        hf.model.norm.weight[C] = -1.0                            # no direct logit path
        hf.model.embed_tokens.weight[2, C] = 3.0                  # BOS row carries it
    return _Wrapper(hf)


def _llama_like():
    return _Wrapper(_HF(LlamaLikeRMSNorm, LLAMA_NORMS))


# ---- pure parts -------------------------------------------------------------

def test_classify_norms_by_count_not_name():
    r, w = classify_norms(GEMMA_NORMS)
    assert w == ["post_attention_layernorm", "post_feedforward_layernorm"]
    assert r == ["input_layernorm", "pre_feedforward_layernorm"]
    r, w = classify_norms(LLAMA_NORMS)
    assert w == [] and r == LLAMA_NORMS          # llama's "post_attention" is a READER


def test_gain_offset_reads_the_convention_from_source():
    assert gain_offset(GemmaLikeRMSNorm(4)) == 1.0
    assert gain_offset(LlamaLikeRMSNorm(4)) == 0.0


def test_channel_stats_rank_sign_and_dominance():
    G = torch.ones(L, D)
    G[:, C] = 30.0
    G[0, C] = -30.0                               # one flipped layer
    st = channel_stats(G, [C, 3])
    a = st["at"][str(C)]
    assert a["rank"] == [1] * L and a["frac_rank1"] == 1.0 and a["frac_rank_le8"] == 1.0
    assert not a["sign_consistent"]               # the flip is caught
    assert st["at"]["3"]["frac_rank1"] == 0.0
    assert abs(a["abs_over_median"][1] - 30.0) < 1e-6
    G[0, C] = 30.0
    assert channel_stats(G, [C])["at"][str(C)]["sign_consistent"]


def test_top_channels_finds_the_planted_one():
    G = torch.randn(L, D) * 0.1 + 1.0
    G[:, C] = 25.0
    t = top_channels(G, k=3)
    assert t["channels"][0] == C and t["mean_abs_gain"][0] == pytest.approx(25.0)
    assert 0.5 < t["median_channel"] < 1.5


# ---- full report on the toy trees --------------------------------------------

def test_layout_detection_ignores_head_dim_norms_and_finds_final_norm():
    lay = find_layout(hf_module(_gemma_like()))
    assert lay["norm_names"] == GEMMA_NORMS and lay["d_model"] == D and lay["n_layers"] == L
    assert lay["final_norm_name"] == "model.norm"
    assert lay["writer_norms"] == ["post_attention_layernorm", "post_feedforward_layernorm"]


def test_gemma_like_report_matches_the_audit_predictions():
    rep = norm_gain_report(_gemma_like(), [C, 3])
    json.dumps(rep)                               # serialisable as written by 23
    assert rep["gain_offset"] == 1.0
    v = rep["verdict"][str(C)]
    assert v["writer_gain_dominant"] and v["writer_sign_consistent"]
    assert v["writer_frac_rank1"] == 1.0
    assert v["reader_gain_hidden"] and not v["reader_gain_bias_input"]
    assert not v["writes_to_logits"]
    assert rep["top_by_writer_gain"]["channels"][0] == C
    assert rep["per_norm"]["post_feedforward_layernorm"]["at"][str(C)]["mean"] == pytest.approx(41.0)
    assert rep["embedding"]["at"][str(C)]["bos_row"] == pytest.approx(3.0 * D ** 0.5)
    # the unplanted channel is ordinary on every count
    u = rep["verdict"]["3"]
    assert not u["writer_gain_dominant"] and not u["reader_gain_hidden"]
    txt = format_report(rep, nuisance_channels=[C])
    assert f"ch {C:5d}" in txt and "overlap with the activation nuisance set: [7]" in txt


def test_llama_like_report_is_the_control():
    rep = norm_gain_report(_llama_like(), [C])
    assert rep["gain_offset"] == 0.0 and rep["writer_norms"] == []
    assert "top_by_writer_gain" not in rep
    v = rep["verdict"][str(C)]
    assert v["writer_gain_dominant"] is None
    assert not v["reader_gain_hidden"] and not v["reader_gain_bias_input"]
    assert "NONE (pre-norm family)" in format_report(rep)


def test_verdict_is_pure_over_the_report_dict():
    rep = norm_gain_report(_gemma_like(), [C])
    again = verdict(rep, [C])
    assert again == rep["verdict"]


def test_empty_channel_list_is_fine():
    rep = norm_gain_report(_gemma_like(), [])
    assert rep["channels"] == [] and rep["verdict"] == {}
    assert "embedding" not in rep


def test_candidate_channels_union_ranked_by_layer_count_with_control_fallback():
    from loudchannel.norm_gains import candidate_channels

    chans = {"t_inst": {11: [2339], 12: [2339, 107], 13: [2339]},
             "t_post_inst": {11: [2339], 12: [2339], 13: [2339, 3499]}}
    acts = torch.randn(10, L, D)
    cand, source, count = candidate_channels(chans, acts, max_k=16)
    assert cand[0] == 2339 and set(cand) == {2339, 107, 3499} and source == "nuisance"
    assert count[2339] == 6 and count[107] == 1
    cand, _, _ = candidate_channels(chans, acts, max_k=1)
    assert cand == [2339]
    # healthy model: nothing found -> per-layer top-|mean| coordinate, flagged
    acts[:, :, C] += 50.0
    cand, source, count = candidate_channels({"t_inst": {li: [] for li in range(L)}}, acts)
    assert cand == [C] and source == "top_mean_fallback" and count == {}


@pytest.mark.parametrize("family", ["gemma3_vlm", "gemma3", "gemma2", "llama", "qwen2"])
def test_layout_on_real_transformers_module_trees(family):
    """Tiny-config instances of the real architectures (no weights): the
    detector must find the right norms, the right reader/writer split, the
    right final norm (model.language_model.norm for the vision-registered
    Gemma-3 class the repo loads) and the right gain convention — under which
    a fresh init has median |gain| = 1 for every norm."""
    tf = pytest.importorskip("transformers")
    tiny = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=300)
    if family == "gemma3":
        hf = tf.Gemma3ForCausalLM(tf.Gemma3TextConfig(
            **tiny, sliding_window=8, layer_types=["full_attention"] * 3))
        path, n_norms, final = "model.layers", 4, "model.norm"
    elif family == "gemma3_vlm":
        hf = tf.Gemma3ForConditionalGeneration(tf.Gemma3Config(
            text_config=tf.Gemma3TextConfig(**tiny, sliding_window=8,
                                            layer_types=["full_attention"] * 3),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8,
                               num_channels=3),
            mm_tokens_per_image=4, image_token_index=299, boi_token_index=297,
            eoi_token_index=298))
        path, n_norms, final = "model.language_model.layers", 4, "model.language_model.norm"
    elif family == "gemma2":
        hf = tf.Gemma2ForCausalLM(tf.Gemma2Config(**tiny, sliding_window=8))
        path, n_norms, final = "model.layers", 4, "model.norm"
    elif family == "llama":
        hf = tf.LlamaForCausalLM(tf.LlamaConfig(**tiny))
        path, n_norms, final = "model.layers", 2, "model.norm"
    else:
        hf = tf.Qwen2ForCausalLM(tf.Qwen2Config(**tiny))
        path, n_norms, final = "model.layers", 2, "model.norm"
    lay = find_layout(hf, path)
    assert len(lay["norm_names"]) == n_norms and lay["final_norm_name"] == final
    assert lay["d_model"] == 64 and lay["n_layers"] == 3
    assert bool(lay["writer_norms"]) == (n_norms == 4)
    w = _Wrapper(hf)
    rep = norm_gain_report(w, [5], layers_path=path)
    assert rep["gain_offset"] == (1.0 if family.startswith("gemma") else 0.0)
    for n in lay["norm_names"]:
        assert rep["per_norm"][n]["median_abs_per_layer"][0] == pytest.approx(1.0)
    assert rep["final_norm"]["median_abs"] == pytest.approx(1.0)


def test_attention_vs_ffn_writer_gain_ratio_at_the_channel():
    """The theory brief's beta_c note: does the class-carrying (attention) path
    pass through the same gain at c* as the sink-generating (FFN) path? The
    toy plants post-FFN gain 41 and post-attention gain 6 -> ratio ~ 0.15."""
    rep = norm_gain_report(_gemma_like(), [C, 3])
    v = rep["verdict"][str(C)]
    assert v["writer_attn_over_ffn_gain_median"] == pytest.approx(6.0 / 41.0, rel=1e-3)
    assert len(v["writer_attn_over_ffn_per_layer"]) == L
    assert rep["verdict"]["3"]["writer_attn_over_ffn_gain_median"] == pytest.approx(1.0)
    assert "attn/ffn gain 0.15" in format_report(rep)
    assert "writer_attn_over_ffn_gain_median" not in norm_gain_report(_llama_like(), [C])["verdict"][str(C)]
