"""Is the massive channel a property of the MODEL or of the benchmark?

Everything measured so far ran on instruction-shaped prompts from three
datasets (AdvBench + SORRY-Bench, Alpaca, SST-2), all rendered through the chat
template. The weight-level check (D7 in `01_extract`) already says the channel is built
by the architecture -- rank-1 post-FFN writer gain in every layer, reader gain
~0, no path to the logits -- which is input-independent by construction. This
script closes the prompt-side half: does the channel actually show up, at the
same index and the same relative scale, on text that has nothing to do with any
of those benchmarks, and on text that never sees the chat template?

Input families (`--families`, all by default), ~n per family:
  instructions   ordinary task instructions (the familiar case, as the anchor)
  code           source code in several languages
  multilingual   non-English prose (several scripts)
  random_tokens  uniformly sampled vocabulary ids -- no linguistic structure
  long_document   long repetitive-free prose, to reach deep token indices
  short_fragments  a handful of tokens: headings, numbers, punctuation

and two renderings (`--render`): `chat` (the template, comparable with
everything else in the repo) and `raw` (the text alone, no template at all).

Per family x rendering, at EVERY layer, over the final token and a sample of
interior token positions, it records: the top-|mean| and top-sigma coordinate,
whether that coordinate is the reference channel, the reference channel's rho
(sigma over the layer median sigma), its |mean| ratio to the second-largest
coordinate, and its sign consistency across inputs. If the channel is the
model's, every family looks alike; if it is the benchmark's, the instruction
family stands alone.

No intervention, no generation: one forward pass per family. Minutes on one GPU.

    python experiments/07_input_generality.py --model gemma3-12b
    python experiments/07_input_generality.py --model llama3-8b        # control
    python experiments/07_input_generality.py --model gemma3-12b --render raw chat
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import base_parser, dump, setup

from loudchannel.config import artifacts_dir

SUB = "recipes"

CODE = [
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[len(a)//2]\n"
    "    return quicksort([x for x in a if x < p]) + [x for x in a if x == p] + quicksort([x for x in a if x > p])",
    "SELECT u.id, COUNT(o.id) AS n FROM users u LEFT JOIN orders o ON o.user_id = u.id\nGROUP BY u.id HAVING COUNT(o.id) > 3 ORDER BY n DESC LIMIT 20;",
    "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 10; i++) printf(\"%d\\n\", i * i);\n    return 0;\n}",
    "const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };",
    "impl<T: Ord> BinaryHeap<T> {\n    pub fn push(&mut self, item: T) {\n        let old_len = self.len();\n        self.data.push(item);\n        self.sift_up(0, old_len);\n    }\n}",
    "package main\n\nimport \"sync\"\n\nfunc worker(wg *sync.WaitGroup, ch chan int) {\n\tdefer wg.Done()\n\tfor v := range ch {\n\t\t_ = v * 2\n\t}\n}",
    "class Net(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.fc = nn.Linear(d, d)\n    def forward(self, x):\n        return F.relu(self.fc(x))",
    "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>x</title></head>\n<body><div class=\"row\"><span>hello</span></div></body></html>",
]

MULTILINGUAL = [
    "Die Wirtschaft des Landes wuchs im vergangenen Quartal um zwei Prozent, was die Erwartungen der meisten Analysten übertraf.",
    "Le musée présente une collection d'œuvres impressionnistes rassemblées au cours du siècle dernier par plusieurs familles.",
    "経済の回復は緩やかであり、地方の中小企業にとっては依然として厳しい状況が続いていると報告されている。",
    "该项目的目标是在未来三年内将城市的公共交通网络扩展到周边的六个区县。",
    "Экономисты отмечают, что инфляция замедлилась, однако цены на продукты питания продолжают расти.",
    "تشير التقارير إلى أن معدلات هطول الأمطار هذا العام كانت أقل بكثير من المتوسط السنوي المعتاد.",
    "El comité anunció que la votación se aplazará hasta que se complete la revisión de los documentos presentados.",
    "इस अध्ययन में शोधकर्ताओं ने पाया कि नींद की कमी का सीधा असर स्मृति और ध्यान पर पड़ता है।",
]

LONG_DOCUMENT = [
    " ".join([
        "The library had been built in stages, each generation adding a wing that",
        "reflected what it believed a reader needed. The oldest room held maps and",
        "ledgers, the next held novels, and the newest held terminals that indexed",
        "everything the others contained. Visitors rarely noticed the seams, though",
        "the floors changed underfoot from stone to oak to a synthetic composite that",
        "muffled every step. On weekday mornings the reading room filled slowly:",
        "students first, then researchers, then the retired men who came for the",
        "newspapers and stayed for the warmth. The head librarian kept a notebook of",
        "questions she could not answer, and once a year she read it through from the",
        "beginning, marking the ones that time had resolved. Most had. A few had not,",
        "and those she copied into a fresh notebook, so the list was always short and",
        "always old. She said this was the only honest measure of what a library was",
        "for, and that catalogues counted holdings while the notebook counted needs.",
    ] * 3),
    " ".join([
        "Sediment cores drawn from the lakebed record the last eleven thousand years",
        "in bands a few millimetres thick. Each band is a year: pale silt from the",
        "spring melt, dark organic matter from the summer bloom. Counting them is",
        "slow work, and counting them twice is slower, but the arithmetic is simple",
        "and the result is a calendar no instrument had to calibrate. Where the bands",
        "thicken, the melt was heavy; where they vanish, the lake froze through or the",
        "inflow moved. Two intervals interrupt the sequence entirely, and both",
        "coincide with ash layers that can be matched to eruptions dated elsewhere.",
        "That coincidence is what turns a local record into a regional one, and it is",
        "the reason cores from separate basins can be aligned at all.",
    ] * 3),
    " ".join([
        "The protocol specifies that each participant transmits a commitment before",
        "any value is revealed, so that no party can choose its input after seeing",
        "another's. Commitments are binding but hiding: binding, because opening one",
        "to a second value requires solving a problem assumed to be hard; hiding,",
        "because the commitment alone is indistinguishable from a commitment to any",
        "other value of the same length. The reveal phase is then a simple broadcast,",
        "and correctness follows from every participant checking every opening. What",
        "the protocol does not provide is fairness: a participant who dislikes the",
        "outcome can abort before revealing, and the remaining parties learn nothing.",
    ] * 3),
]

SHORT_FRAGMENTS = [
    "Chapter 4", "2019-2024", "TODO:", "###", "Table 2.1", "yes", "N/A",
    "$1,240.00", "Figure 3b", "etc.", "( ii )", "42",
]

INSTRUCTIONS = [
    "Summarize the main argument of the passage in two sentences.",
    "Convert this list of dates into ISO 8601 format.",
    "Explain why the sky appears blue during the day.",
    "Write a polite email declining a meeting invitation.",
    "What is the derivative of x^3 with respect to x?",
    "Suggest three names for a bakery that specializes in rye bread.",
    "Translate the following sentence into formal register.",
    "List the steps for changing a bicycle tire.",
]


def random_token_inputs(tokenizer, n: int, length: int, seed: int) -> list[str]:
    """Uniform draws from the vocabulary — text with no linguistic structure at
    all, the hardest case for 'the channel tracks something about the prompt'."""
    g = torch.Generator().manual_seed(seed)
    vocab = int(getattr(tokenizer, "vocab_size", 32000))
    special = set(tokenizer.all_special_ids or [])
    out = []
    for _ in range(n):
        ids = []
        while len(ids) < length:
            t = int(torch.randint(0, vocab, (1,), generator=g))
            if t not in special:
                ids.append(t)
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


FAMILIES = ("instructions", "code", "multilingual", "random_tokens",
            "long_document", "short_fragments")


def build_family(name: str, tokenizer, n: int, seed: int) -> list[str]:
    pool = {"instructions": INSTRUCTIONS, "code": CODE, "multilingual": MULTILINGUAL,
            "long_document": LONG_DOCUMENT, "short_fragments": SHORT_FRAGMENTS}
    if name == "random_tokens":
        return random_token_inputs(tokenizer, n, 48, seed)
    base = pool[name]
    return [base[i % len(base)] for i in range(n)]


@torch.no_grad()
def residuals(model, texts: list[str], *, render: str, batch_size: int):
    """[N, L, D] at the LAST token, plus [N, L, D] at a mid-sequence token.

    Deliberately not using positions_for/extract_multi: those locate named
    template positions, and half the point here is to run with no template.
    """
    tok = model.tokenizer
    tup = model.output_is_tuple
    layer_envoys = [model.layers[li] for li in range(model.n_layers)]
    last_rows, mid_rows = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        prompts = [model.render(t) for t in chunk] if render == "chat" else list(chunk)
        enc = tok(prompts, return_tensors="pt", padding=True)
        n_tok = enc["attention_mask"].sum(-1)                      # [B] real length
        b = torch.arange(len(chunk))
        last_idx = n_tok - 1
        mid_idx = (n_tok // 2).clamp(min=0)
        saved = []
        with model.trace(prompts):
            for env in layer_envoys:
                o = env.output
                h = o[0] if tup else o
                saved.append(h[b, last_idx].detach().cpu())
                saved.append(h[b, mid_idx].detach().cpu())
        st = torch.stack([s.float() for s in saved])               # [2L, B, D]
        st = st.reshape(model.n_layers, 2, len(chunk), -1)
        last_rows.append(st[:, 0].permute(1, 0, 2))                # [B, L, D]
        mid_rows.append(st[:, 1].permute(1, 0, 2))
    return torch.cat(last_rows), torch.cat(mid_rows)


def channel_stats(x: torch.Tensor, ref: int) -> dict:
    """x: [N, L, D]. Per layer: who dominates, and how loud the reference is."""
    mean = x.mean(0)                                               # [L, D]
    sd = x.std(0).clamp_min(1e-9)                                  # [L, D]
    top_mean = mean.abs().argmax(-1)
    top_sd = sd.argmax(-1)
    rho = (sd[torch.arange(x.shape[1]), ref] / sd.median(-1).values.clamp_min(1e-9))
    srt = mean.abs().sort(-1, descending=True).values
    ratio_12 = (srt[:, 0] / srt[:, 1].clamp_min(1e-9))
    ref_mean = mean[torch.arange(x.shape[1]), ref]
    sign_consistency = (torch.sign(x[:, :, ref]) == torch.sign(ref_mean)).float().mean(0)
    return {
        "top_mean_channel": top_mean.tolist(),
        "top_sd_channel": top_sd.tolist(),
        "ref_is_top_mean_frac": float((top_mean == ref).float().mean()),
        "ref_is_top_sd_frac": float((top_sd == ref).float().mean()),
        "ref_rho": [round(float(v), 2) for v in rho],
        "ref_mean": [round(float(v), 1) for v in ref_mean],
        "ref_sign_consistency": [round(float(v), 3) for v in sign_consistency],
        "top1_over_top2_abs_mean": [round(float(v), 2) for v in ratio_12],
    }


def band_summary(st: dict, lo: int, hi: int) -> dict:
    sl = slice(lo, hi + 1)
    rho = st["ref_rho"][sl]
    return {"ref_is_top_mean_frac_band": round(
                sum(c == st["_ref"] for c in st["top_mean_channel"][sl]) / max(hi - lo + 1, 1), 3),
            "rho_median_band": round(sorted(rho)[len(rho) // 2], 1),
            "rho_max_band": round(max(rho), 1),
            "sign_consistency_min_band": round(min(st["ref_sign_consistency"][sl]), 3)}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--render", nargs="+", default=["chat", "raw"],
                    choices=("chat", "raw"))
    ap.add_argument("--n-per-family", type=int, default=24)
    ap.add_argument("--channel", type=int, default=None,
                    help="reference channel (default: read the modal top-sigma "
                         "channel out of diagnostics.json, i.e. 2339 on gemma3-12b)")
    ap.add_argument("--band", nargs=2, type=int, default=None,
                    help="layer band for the summary (default: 20%%-55%% of depth)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, exp = setup(args)
    sub = f"{SUB}_{args.tag}" if args.tag else SUB
    adir = artifacts_dir(exp, args.model, sub)
    L = model.n_layers
    lo, hi = args.band or (int(0.20 * L), int(0.55 * L))

    ref = args.channel
    if ref is None:
        dp = adir / "diagnostics.json"
        assert dp.exists(), "no diagnostics.json — pass --channel explicitly"
        from collections import Counter
        pos0 = next(iter(json.loads(dp.read_text())["by_position"].values()))
        ref = Counter(int(c) for c in pos0["nominal"]["channel"]).most_common(1)[0][0]
    n = args.limit or args.n_per_family
    print(f"reference channel {ref} | {L} layers | band L{lo}-L{hi} | "
          f"{n} inputs per family | families {args.families} | render {args.render}")

    out: dict = {}
    for render in args.render:
        for fam in args.families:
            texts = build_family(fam, model.tokenizer, n, exp["seed"])
            last, mid = residuals(model, texts, render=render,
                                  batch_size=exp["batch_size"])
            entry = {}
            for where, x in (("last_token", last), ("mid_token", mid)):
                st = channel_stats(x, ref)
                st["_ref"] = ref
                st["band"] = band_summary(st, lo, hi)
                st.pop("_ref")
                entry[where] = st
            out[f"{render}/{fam}"] = entry
            b = entry["last_token"]["band"]
            bm = entry["mid_token"]["band"]
            print(f"  {render:5s} {fam:15s} last: ch{ref} is top-|mean| in "
                  f"{b['ref_is_top_mean_frac_band']*100:5.1f}% of band layers, "
                  f"rho med {b['rho_median_band']:6.1f} max {b['rho_max_band']:7.1f}, "
                  f"sign {b['sign_consistency_min_band']:.2f} | mid: "
                  f"{bm['ref_is_top_mean_frac_band']*100:5.1f}%, rho med {bm['rho_median_band']:6.1f}")

    print("\nread: if the channel is the MODEL's, every family and both renderings look"
          "\n      alike (high top-|mean| share, rho in the hundreds, sign ~1.0). If it is"
          "\n      the BENCHMARK's, `instructions` stands alone and random_tokens/code/raw"
          "\n      collapse toward rho ~ 1 and no dominant coordinate.")

    dump(exp, args.model, sub, "input_generality.json", {
        "model": args.model, "reference_channel": ref, "n_layers": L,
        "band": [lo, hi], "n_per_family": n,
        "families": args.families, "renders": args.render,
        "by_family": out,
    })


if __name__ == "__main__":
    main()
