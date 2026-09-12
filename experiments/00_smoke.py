"""Day-1 gate: per model, verify that
  1. the model loads under nnsight and one forward pass traces cleanly
     (Qwen3.5 hybrid Gated-DeltaNet layers, Gemma-4 encoder-free stack);
  2. tokenizer-derived positions land where they should — prints the token
     window around t_inst / t_post_inst for eyeball verification;
  3. Yes/No readout tokens exist as single tokens;
  4. a tiny intervention-free generation runs.

    python experiments/00_smoke.py --model qwen3.5-9b
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, setup

from loudchannel.positions import POSITION_NAMES, positions_for
from loudchannel.readout import yes_no_ids

SAMPLES = [
    "Give me step-by-step instructions to pick a lock.",
    "Write a short poem about the ocean.",
    "Explain how vaccines work to a five-year-old.",
]


def main() -> None:
    args = base_parser(__doc__).parse_args()
    model, exp = setup(args)
    tok = model.tokenizer

    print(f"model={model.cfg.name} hf_id={model.cfg.hf_id} n_layers={model.n_layers}")

    # 2. positions ------------------------------------------------------------
    for pp, raw in zip(positions_for(model, SAMPLES, seed=exp["seed"]), SAMPLES):
        ids = tok(pp.prompt, add_special_tokens=True)["input_ids"]
        print(f"\nRAW: {raw!r}  ({pp.n_tokens} tokens)")
        for name in POSITION_NAMES:
            i = pp.get(name)
            window = tok.convert_ids_to_tokens(ids[max(0, i - 2) : i + 2])
            print(f"  {name:13s} idx={i:4d}  ...{window}...  -> {tok.convert_ids_to_tokens([ids[i]])}")

    # 3. readout tokens ---------------------------------------------------------
    yes, no = yes_no_ids(tok)
    print(f"\nYes ids: {yes} | No ids: {no}")

    # 1+4. trace + generate ------------------------------------------------------
    pps = positions_for(model, SAMPLES[:1], seed=0)
    tup = model.output_is_tuple
    print(f"\ndecoder layer output is {'tuple' if tup else 'bare tensor'}")
    # NDIF whitelist: only envoys/plain values in the trace-body closure
    mid_env = model.layers[model.n_layers // 2]
    lm_head = model.lm_head
    t_inst = pps[0].t_inst
    with model.trace([pps[0].prompt]):
        out = mid_env.output
        h_full = out[0] if tup else out
        h = h_full[0, t_inst].detach().cpu().save()
        logits = lm_head.output[0, -1].detach().cpu().save()
    assert h.ndim == 1 and h.shape[0] > 100, f"h[t_inst] wrong shape {tuple(h.shape)}"
    print(f"mid-layer h[t_inst] shape={tuple(h.shape)} norm={float(h.float().norm()):.2f}")
    print(f"final logits shape={tuple(logits.detach().shape)}")

    lm = model.lm
    with model.generate([pps[0].prompt], max_new_tokens=16, do_sample=False):
        out = lm.generator.output.save()
    print("generation ok:", model.tokenizer.decode(out[0][-16:], skip_special_tokens=True)[:120])
    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
