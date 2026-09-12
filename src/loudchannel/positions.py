"""Token positions, read from each tokenizer — never hard-coded.

Positions (per templated prompt):
  t_inst       last content token of the raw user instruction
  t_post_inst  last token of the fully templated prompt (the token the model
               reads immediately before generating)
  mid_inst     token at the character midpoint of the instruction span
  bos          index 0
  random_early a seeded random token inside the instruction span, != t_inst

Implementation: render the chat template around the raw instruction, locate
the instruction's character span inside the templated string, and map
character offsets to token indices via `return_offsets_mapping`. This is
template-agnostic, so it works unchanged across the Llama, Qwen, Gemma-2 and
Gemma-3 chat templates; 00_smoke.py prints the tokens around each position for
eyeball verification per model.
"""

from __future__ import annotations

import dataclasses
import random
import re

POSITION_NAMES = ("t_inst", "t_post_inst", "mid_inst", "bos", "random_early")


def strip_post_instruction(templated: str, instruction: str) -> str:
    """v2 §3.1 replication: drop everything after the last instruction char.

    The original paper's 'without post-instruction tokens' condition keeps the
    chat template up to and including the user's instruction but removes the
    closing special tokens (e.g. `<|eot_id|><|start_header_id|>assistant...`),
    so generation continues directly from the instruction's last token
    (t_inst == t_post_inst == last prompt token).
    """
    start = templated.rfind(instruction)
    if start == -1:
        instruction = instruction.strip()
        start = templated.rfind(instruction)
    assert start != -1, (
        "instruction text not found inside templated prompt — this chat "
        "template rewrites user content; inspect manually"
    )
    return templated[: start + len(instruction)]


@dataclasses.dataclass
class PromptPositions:
    prompt: str          # templated string
    n_tokens: int
    t_inst: int
    t_post_inst: int
    mid_inst: int
    bos: int
    random_early: int
    t_req: int | None = None       # set only when a request span is located
    t_req_next: int | None = None  # set only when a request span is located
    t_req_first: int | None = None # first token of the request span (H1f
                                   # inversion: steer the whole request span)
    t_fill: int | None = None      # last token of the filler segment
                                   # (composite arrangements only)
    t_fill_next: int | None = None # token after the filler segment

    _OFFSET_RE = re.compile(r"^([a-z_]+?)([+-]\d+)$")

    def get(self, name: str) -> int:
        """Resolve a named position, optionally with a token offset.

        "t_req" -> the stored index; "t_req-2"/"t_req+4" -> the stored index
        shifted by the offset, clamped to [0, n_tokens-1]. Offsets enable the
        dense position-grid experiments without extra char-offset plumbing.
        """
        off = 0
        m = self._OFFSET_RE.match(name)
        if m:
            name, off = m.group(1), int(m.group(2))
        v = getattr(self, name)
        assert v is not None, f"position {name!r} not set for this prompt"
        return min(max(v + off, 0), self.n_tokens - 1)


def _char_to_token(offsets: list[tuple[int, int]], char: int) -> int:
    """Index of the last token whose span contains `char` (start <= char < end)."""
    best = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special tokens have empty spans
            continue
        if s <= char < e:
            best = i
    if best is None:
        # Fall back to the last token starting at or before `char`.
        for i, (s, e) in enumerate(offsets):
            if s != e and s <= char:
                best = i
    assert best is not None, "could not map character offset to a token"
    return best


def _span_tokens(offsets, templated: str, text: str, start: int, end: int,
                 t_inst: int) -> tuple[int, int, int]:
    """(last, next, first) token indices for the located span [start, end)."""
    last = _char_to_token(offsets, end - 1)
    nxt = _char_to_token(offsets, end)
    if nxt <= last:  # token spans the boundary
        nxt = min(last + 1, t_inst)
    first = _char_to_token(offsets, start)
    return last, nxt, first


def find_positions(
    tokenizer,
    templated: str,
    instruction: str,
    *,
    request: str | None = None,   # H2d: raw request text (prefix of instruction)
    filler: str | None = None,    # composite arrangements: benign filler text
    seed: int = 0,
) -> PromptPositions:
    start = templated.rfind(instruction)
    if start == -1:
        # Some templates strip/normalize whitespace; retry on stripped text.
        instruction = instruction.strip()
        start = templated.rfind(instruction)
    assert start != -1, (
        "instruction text not found inside templated prompt — this chat template "
        "rewrites user content; inspect manually"
    )
    end = start + len(instruction)

    enc = tokenizer(templated, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    n = len(enc["input_ids"])

    t_inst = _char_to_token(offsets, end - 1)
    inst_first = _char_to_token(offsets, start)
    mid_inst = _char_to_token(offsets, start + (end - start) // 2)

    rng = random.Random(seed)
    candidates = [i for i in range(inst_first, t_inst) if i != t_inst]
    random_early = rng.choice(candidates) if candidates else max(inst_first, t_inst - 1)

    t_req = t_req_next = t_req_first = None
    if request is not None:
        # locate the request span independently: the request may sit inside a
        # larger wrapper (e.g. the judgment template), not at instruction[0]
        r_start = templated.rfind(request)
        if r_start == -1:
            request = request.strip()
            r_start = templated.rfind(request)
        assert r_start != -1, "request text not found inside templated prompt"
        req_end = r_start + len(request)        # char AFTER the request
        assert start <= r_start and req_end <= end, (
            "request must lie inside the instruction span"
        )
        t_req, t_req_next, t_req_first = _span_tokens(
            offsets, templated, request, r_start, req_end, t_inst)

    t_fill = t_fill_next = None
    if filler is not None:
        f_start = templated.rfind(filler)
        if f_start == -1:
            filler = filler.strip()
            f_start = templated.rfind(filler)
        assert f_start != -1, "filler text not found inside templated prompt"
        f_end = f_start + len(filler)
        assert start <= f_start and f_end <= end, (
            "filler must lie inside the instruction span"
        )
        t_fill, t_fill_next, _ = _span_tokens(
            offsets, templated, filler, f_start, f_end, t_inst)

    return PromptPositions(
        prompt=templated,
        n_tokens=n,
        t_inst=t_inst,
        t_post_inst=n - 1,
        mid_inst=mid_inst,
        bos=0,
        random_early=random_early,
        t_req=t_req,
        t_req_next=t_req_next,
        t_req_first=t_req_first,
        t_fill=t_fill,
        t_fill_next=t_fill_next,
    )


def positions_for_wrapped(model, requests: list[str], wrap, *,
                          system: str | None = None,
                          seed: int = 0) -> list[PromptPositions]:
    """Locate the request span inside an arbitrary wrapper (H1f reply
    inversion: the request sits BEFORE the appended inversion question, so
    t_req/t_req_first bound the span to steer and t_inst != t_req)."""
    out = []
    for i, req in enumerate(requests):
        text = wrap(req)
        templated = model.render(text, system=system)
        out.append(find_positions(model.tokenizer, templated, text,
                                  request=req, seed=seed + i))
    return out


def positions_for(model, instructions: list[str], *, system: str | None = None,
                  seed: int = 0) -> list[PromptPositions]:
    """Render + locate positions for a batch of raw instructions."""
    out = []
    for i, ins in enumerate(instructions):
        templated = model.render(ins, system=system)
        out.append(find_positions(model.tokenizer, templated, ins, seed=seed + i))
    return out


