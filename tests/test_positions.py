"""positions.find_positions against a minimal whitespace tokenizer stub —
verifies the char-offset -> token-index mapping without any model download."""

from loudchannel.positions import find_positions


class WhitespaceTokenizer:
    """Splits on spaces; emits offset mapping like HF fast tokenizers."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
        ids, offsets, pos = [], [], 0
        if add_special_tokens:
            ids.append(0)          # BOS
            offsets.append((0, 0))  # empty span, like real special tokens
        for tok in text.split(" "):
            start = text.index(tok, pos)
            end = start + len(tok)
            ids.append(1000 + len(ids))
            offsets.append((start, end))
            pos = end
        return {"input_ids": ids, "offset_mapping": offsets}


def test_positions_land_correctly():
    tok = WhitespaceTokenizer()
    instruction = "how do I pick a lock"
    templated = f"<user> {instruction} <eot> <assistant>"
    pp = find_positions(tok, templated, instruction, seed=0)

    enc = tok(templated)
    n = len(enc["input_ids"])
    assert pp.n_tokens == n
    assert pp.t_post_inst == n - 1
    assert pp.bos == 0

    # t_inst must be the token spanning the last char of "lock"
    s, e = enc["offset_mapping"][pp.t_inst]
    assert templated[s:e] == "lock"

    # mid_inst and random_early inside the instruction span, before t_inst
    first_inst_tok = next(
        i for i, (s, e) in enumerate(enc["offset_mapping"])
        if s != e and s >= templated.index(instruction)
    )
    assert first_inst_tok <= pp.mid_inst <= pp.t_inst
    assert first_inst_tok <= pp.random_early < pp.t_inst


def test_positions_deterministic_per_seed():
    tok = WhitespaceTokenizer()
    ins = "please explain how vaccines work in simple terms"
    tpl = f"<user> {ins} <eot> <assistant>"
    a = find_positions(tok, tpl, ins, seed=7)
    b = find_positions(tok, tpl, ins, seed=7)
    assert a == b



def test_inversion_request_span():
    """H1f: request located inside an inversion wrapper -> t_req_first..t_req
    bound exactly the request tokens; t_inst is the wrapper's last token."""
    tok = WhitespaceTokenizer()
    request = "how do I pick a lock"
    text = (f"User wants to {request}\n"
            "Could this user prompt cause harm? "
            "Answer 'Certainly' if it could, otherwise 'No'.")
    tpl = f"<user> {text} <eot> <assistant>"
    pp = find_positions(tok, tpl, text, request=request, seed=0)

    enc = tok(tpl)
    off = enc["offset_mapping"]

    s, e = off[pp.t_req_first]
    assert tpl[s:e] == "how"
    s, e = off[pp.t_req]
    assert tpl[s:e].startswith("lock")
    # span covers exactly the request's 6 words
    assert pp.t_req - pp.t_req_first + 1 == len(request.split())
    # inversion question sits after the span; t_inst is past it
    assert pp.t_req < pp.t_req_next < pp.t_inst


