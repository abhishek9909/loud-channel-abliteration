# One loud coordinate breaks difference-of-means abliteration

Code for the measurement, the corrected estimators, and the causal experiments
behind the claim that a single residual-stream coordinate — not refusal — is
what breaks the standard difference-of-means "abliteration" recipe on Gemma-3.

The short version of the finding: a coordinate of Gemma-3's residual stream is
around eighty thousand units tall while a typical coordinate beside it is a
couple of hundred, and that one fact breaks both the estimator (difference-of-means
weights each coordinate by variance × discriminability², so the loudest
coordinate owns the vector regardless of whether it carries class evidence) and
the intervention (the projection scalar `x·d̂` stops measuring the feature and
starts measuring the sink). Neither failure is about safety behaviour: both
reproduce on a sentiment contrast with no safety content, and the coordinate is
already at full amplitude in the base checkpoint before any instruction tuning.
It is just as dominant on uniformly random tokens as on instructions, though
that shows the channel is input-independent, not that the abliteration failure
is.

Everything here is model-agnostic. Two statistics decide whether a model needs
the correction. From activations, **ρ/√d_model**, where ρ = σ_c / median σ is
the loudest coordinate's scale relative to the typical one — plain ρ is not
enough, because a given ρ buys far more of the variance in a 1,152-dimensional
residual than in a 3,840-dimensional one. From the weights alone, and therefore
before running anything at all, the **post-FFN writer gain as a multiple of the
layer median**. Across eight models the two agree and both separate cleanly.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add '[dev]' or `pip install pytest ruff` for tests
```

Model internals are accessed through [nnsight](https://nnsight.net). GPU
requirements are per model: 4B/9B/12B fit a 40 GB card in bf16; 27B needs 80 GB
or two GPUs with `device_map=auto`. The weights-only scans need no GPU and no
model load at all.

The frozen prompt splits in `data/splits/` are committed and hash-checked on
load, so nothing needs downloading to reproduce the main runs. `HF_HOME` should
point at wherever the model weights are cached.

Sanity check, no GPU:

```bash
pytest tests -q
```

---

## Layout

```
src/loudchannel/     library
  recipes.py           the six estimators (raw, masked, standardized,
                       winsorized, length-matched, covariate-adjusted)
  selection.py         bypass / induce / KL scoring and cell selection
  interventions.py     directional ablation and additive steering under nnsight
  channel_ops.py       direct per-coordinate edits (zero / scale / shift)
  channel_interventions.py  generation, MMLU and NLL under those edits
  nominal.py           the class-gap decomposition and evidence accounting
  norm_gains.py        RMSNorm gain provenance: who writes the channel, who
                       reads it, whether it reaches the logits
  degeneracy.py        the three-way refusal / compliance / degenerate gate
  mmlu.py, pipelines.py  capability and fluency controls
experiments/         the runnable stages (below)
scripts/             CPU-only analysis, weights-only scans, and the figure
configs/models/      one YAML per checkpoint
data/splits/         frozen, hash-checked prompt splits
```

Artifacts land under `artifacts/<model>/recipes/` by default
(`configs/experiment.yaml: artifacts_dir`).

---

## The pipeline

Four stages, run in order. Each writes JSON (and `.pt` for directions) that the
next one reads, so a stage can be re-run without repeating the ones before it.

```bash
M=gemma3-12b

python experiments/01_extract.py  --model $M      # ~1 GPU-hour
python experiments/02_select.py   --model $M      # the expensive one
python experiments/03_confirm.py  --model $M --n-test 150
python experiments/04_dose.py     --model $M      # needs only 01+02

python scripts/analyze_recipes.py                 # CPU, merges every model
```

| stage | what it does |
|---|---|
| `01_extract` | One extraction pass over all candidate positions. Identifies the loud ("nuisance") channels **class-blind**, from the harmless split alone. Builds every recipe's direction at every (position, layer). Also runs the diagnostics: the channel's share of the vector against its share of the class evidence, the class-gap decomposition, and the weight-level provenance of the nuisance set. |
| `02_select` | Scores every candidate by the standard criteria — minimise refusal under ablation on held-out harmful prompts, subject to the direction inducing refusal when added to harmless ones, first-token KL on harmless prompts under 0.1, and the layer below 0.8·L. (The induce rule here is stricter than the original `induce_score > 0`: induction on at least half the prompts.) This is where "0 usable cells out of 288" comes from. |
| `03_confirm` | Generation on held-out evaluation prompts, scored three ways (refusal / compliance / degenerate), plus MMLU and NLL per arm. Includes the `random` and `random_perp` nulls and the legacy per-layer direction. |
| `04_dose` | Steering-onset sweep in units of the off-channel residual norm, three-way scored so an "onset" that is really a collapse cannot pass as induced refusal. |

Run the same four stages on `llama3-8b` and `qwen2.5-7b`. Both halves matter:
the correction has to work on Gemma-3 **and** be a no-op on the models that were
already fine. Either half failing sinks the claim.

The full grid is 48 layers × 6 positions × 6 recipes, and stage `02_select` is
3 batched forward passes per candidate. Start with `--layer-step 4` and refine
around whatever survives.

### The recipes

Selected with `--recipes`; all six run by default.

| id | one line |
|---|---|
| `r0_raw` | plain difference of means — the standard recipe |
| `r1_masked` | zero the class-blind loud channels, then difference |
| `r2_standardized` | `Δ_c / (σ_c² + λ)` — diagonal LDA |
| `r3_winsorized` | clip activations at a quantile, then difference |
| `r4_length_matched` | resample to equalize prompt length |
| `r5_covariate_adjusted` | per-coordinate OLS class coefficient holding word count and terminal punctuation fixed |

`r4` and `r5` are the controls that clean the **data**; `r1`–`r3` clean the
**scale**. The dissociation between them is the point: on a model with a loud
channel, only the scale corrections yield usable cells.

Nuisance channels are chosen from the harmless split only
(`|mean| > --nuisance-ratio × layer median`, default 50). Choosing them from the
difference would remove class signal by construction.

### The three-way gate

Degenerate output contains no refusal substrings, so under binary scoring a
recipe that destroys the model scores as a perfect refusal bypass. Every
response is therefore labelled refusal / compliance / degenerate
(`src/loudchannel/degeneracy.py`), and the claim is "refusal fell **and** the
remaining outputs are real answers".

One caveat is worth knowing before trusting that gate: its rules key on
repetition, and the two failure modes here sit at opposite ends of that axis.
The leak produces repetition loops, which it catches; removing the channel
produces high-diversity multi-script word salad at a distinct-word ratio of
1.00, the theoretical maximum, which it scores as compliance. Both are
degenerate to any reader — that disagreement is a fact about the rule, not
about the model. `scripts/calibrate_fluency_rule.py` measures two candidate
two-sided fluency statistics (stopword rate, ASCII share) against stored
transcript populations, but an LLM judge disposes of the problem outright, and
MMLU and NLL catch it regardless, which is why both are reported per arm.

---

## The causal experiments

These read `diagnostics.json`, `directions.pt` and `selection.json` from the
pipeline above, so run `01` and `02` first.

```bash
# E1 — route separation. Is the collapse the off-channel leak injected along
# the feature direction, or the loss of the channel itself? Runs both routes
# separately, plus the ordinary ablation that combines them.
python experiments/05_channel_causal.py --model $M --stage e1 --n-test 100

# E2 — is the channel's post-instruction class correlate causally used?
# Readout-only first (minutes); add generation once the deltas are non-trivial.
python experiments/05_channel_causal.py --model $M --stage e2 --n-test 100 --no-gen

# Does an unrelated binary contrast put the same signal on the same channel?
python experiments/06_contrast_generality.py --model $M

# Is the channel an artifact of instruction-shaped prompts through a chat
# template? Six input families × two renderings.
python experiments/07_input_generality.py --model $M

python scripts/analyze_channel_causal.py --model $M --spot
```

`--spot` prints the distinct opening lines per arm and the share of the most
common one: a collapsed distribution means the lexical classifier was measuring
a mode, not refusal behaviour. Worth looking at once per run.

On a model with no loud channel the E1 arms zero an ordinary coordinate and must
be a no-op — that is the same "no-op where there is nothing to remove" symmetry
the recipe result rests on, at the level of the coordinate rather than the
estimator. Worth one run each on the pre-norm controls.

---

## The weights-only scans

No GPU, no model load, seconds per model: these read the RMSNorm gain vectors
straight out of the safetensors shards.

```bash
# Discover the channel per model and compare the ladder. Never downloads;
# a cache miss is an error naming the cache it searched.
python scripts/scan_norm_gain_ladder.py

# One model in detail, with a known channel index.
python scripts/check_gemma_norm_gains.py --model google/gemma-3-12b-it --channel 2339
```

This is the practical payoff: the writer/reader/final-norm gain profile says in
advance which models will need the correction, before any abliteration is run
and watched to fail.

---

## The figure

```bash
python scripts/make_ablation_residual_fig.py          # reads artifacts/, writes figures/
python scripts/make_ablation_residual_fig.py \
    --left gemma3-4b:t_post_inst:20 --right gemma2-9b:t_post_inst:30
```

Sorted per-coordinate magnitude of the residual, in units of each model's own
typical coordinate, with three curves per panel: before, after ablating with the
raw direction, and (dashed) after the *same* ablation with the class-blind
masked direction. The dashed curve is the point — same operation, same cell,
only `a` set to zero, and it lands on "before".

Everything load-bearing is read from `diagnostics.json`: `|x_c*|` is recovered
from `norms.rms_full` and `norms.rms_wo`, and the direction's component on the
loud coordinate is `a = sqrt(top-1 share)`, which reproduces the independently
stored `decomposition.a` in `channel_causal_e2.json` to five decimals. Only the
tail's per-coordinate texture is a draw. On a model whose mask is empty at the
selected cell — Llama at layer 11 has `n_channels = 0` and gain exactly 1.000 —
all three curves coincide, which is the no-op result drawn rather than asserted.

---

## Replication on other prompt sets

The main grid uses AdvBench + SORRY-Bench against Alpaca. Three swapped pairs
are available, with a deliberately different covariate structure — in one of
them the prompt-length gap points the other way:

```bash
python scripts/freeze_splits.py --only-replication   # needs network; wjb is HF-gated
python experiments/01_extract.py --model $M --dataset strongreject_wjb --tag srwjb
python experiments/02_select.py  --model $M --tag srwjb
python experiments/03_confirm.py --model $M --tag srwjb
```

Datasets: `strongreject_wjb`, `wjb_orbench`, `orbench_alpaca`. The dataset name
is written into `directions.pt` and read back by the later stages, so a scoring
run can never be paired with directions estimated on different data. `--tag`
keeps the artifacts separate from the main run.

Note on `--subset`: the frozen harmful splits are stored source-by-source and
nothing shuffles on load, so a plain head slice would be single-source. The
default `stratified` round-robins the sources before slicing; `head` is kept for
the single-source comparison. The choice is written into `directions.pt` so the
validation rows can never come from a different ordering than the training rows.

---

## Pre-registered predictions

Written down before the runs, so the outcome could falsify them. Recorded here
because three of them came out wrong, and that is the informative part.

1. `r0_raw` fails the KL check on Gemma-3. "All N raw candidates rejected for
   KL" is the quantitative form of "the original vector was never a feature".
2. `r1_masked` clears KL and drops refusal, with degeneracy at clean levels and
   MMLU within a few points of clean.
3. `r1_masked` beats `r2_standardized` as an *intervention*. **This was wrong.**
   For ablation they are indistinguishable; for steering, standardized wins
   clearly, because masking hard-zeroes the loudest channels but leaves a tail
   at ρ 3–6 on coordinates that *are* read, while dividing by variance
   suppresses the whole tail continuously.
4. Steering onset tracks 1/b, where b is the share of the unit vector lying off
   the loud channels. Held.
5. No-op on the pre-norm controls: `cos(r0, r1) > 0.95` and the selected cell
   and outcome unchanged. `r2` is exempt from the geometric half — standardizing
   rotates the vector on every model, so its no-op criterion is behavioural.
6. If no recipe clears KL on Gemma, the "Gemma's refusal mechanism is genuinely
   different" hypothesis gets real support and *that* becomes the finding.
7. The ρ ≈ 50 threshold — "any model carrying a residual coordinate above
   roughly ρ ≈ 50 will need the correction" — was written down on the strength
   of one model and then tested on five more. **This was wrong too**, and in
   both directions at once. Gemma-3-1B sits at ρ = 16 and fails completely
   (0 raw cells of 156, median ablation KL 21); Gemma-2-2B sits at ρ = 8 and is
   fine (2 raw cells, KL 0.57). What ρ leaves out is how many coordinates the
   loud one competes against, so the replacement is ρ/√d_model — 0.11, 0.12,
   0.16, 0.17 for the four models that work against 0.46, 0.79, 1.56, 1.73 for
   the four that do not — or, with no forward pass at all, the writer gain
   ratio: 2.5x and 2.7x for the Gemma-2 sizes against 4.5x to 6.9x for the four
   Gemma-3 sizes. Both are separating statistics fitted after seeing where eight
   models landed, not validated criteria; the next model is the real test.

---

## Models

`configs/models/` carries one YAML per checkpoint: the Gemma-3 ladder
(1B / 4B / 12B / 27B, plus the 12B base checkpoint for the pretraining-vs-
instruction-tuning question), Gemma-2 (2B and 9B) as the within-family control,
and Llama-3-8B and Qwen-2.5-7B as pre-norm controls. Gemma-3-1B is text-only,
so it has no `model_class: vision`; the other Gemma-3 sizes are
multimodal-registered.

Gemma-3 checkpoints are multimodal-registered and load through
`VisionLanguageModel` (hence `torchvision`), used text-only. `gemma-3-12b-pt` is
gated separately on HF.

Adding a model is a YAML file: `hf_id`, `layers_path`, `lm_head_path`,
`n_layers`, `d_model`. `python experiments/00_smoke.py --model <name>` verifies
that it traces, that the tokenizer-derived positions land where they should, and
that a short generation runs.

---

## License

MIT.
