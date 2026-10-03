"""
Charset / special-token invariants for OCRDatasetManager (defects D7 and D2).

D7  the `new_tokens` path grew the charset *after* max_pred_ind and char_only_set were
    derived from it, so end/start/pad were allocated from a stale counter and aliased real
    characters. It did not crash -- ocr_manager.py sizes the output layer from the extended
    charset -- so a model trained through it emitted one id for two different symbols.
    No caller ever passed it; the path is now refused rather than silently wrong.

D2  layout tokens were identified by charset *position* (`ind >= len(char_only_set)`), which
    assumes every non-character sorts to the tail. char_only_set also drops "\n", which sorts
    to index 0, so the boundary lands one index low and exactly one real character per corpus
    is read as a layout token. `layout_token_identity` selects the identity test instead; it
    is off by default so no recorded run's input encoding changes.

Run with:  python -m pytest hand_v2/tests/test_charset_tokens.py -q
"""
import os
import pickle
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hand.OCR.ocr_dataset_manager import OCRDatasetManager, OCRDataset  # noqa: E402
from hand.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_SEM  # noqa: E402

FORMATTED = os.path.join(ROOT, "formatted")
SEM = sorted(set(READ_SEM.values()))
# a charset shaped like a real READ page charset: newline at the front, letters in the
# middle, an em dash just below the semantic block, the semantic tokens at the tail
CHARSET = sorted(["\n"] + [chr(97 + i) for i in range(26)] + ["—"] + SEM)


def make_manager(charset, identity=False, **extra):
    params = {
        "dataset_class": object,
        "datasets": {},
        "charset": charset,
        "batch_size": 1,
        "config": {
            "padding_value": 0,
            "padding_token": None,
            "charset_mode": "seq2seq",
            "constraints": ["add_eot", "add_sot", "hand_encoding"],
            "layout_token_identity": identity,
        },
    }
    params.update(extra)
    return OCRDatasetManager(params)


def as_dataset(m):
    """The attributes apply_specific_treatment_after_dataset_loading copies onto a dataset."""
    d = types.SimpleNamespace(
        charset=m.charset, char_only_set=m.char_only_set, sem_tokens=m.sem_tokens,
        tokens=m.tokens, layout_token_ids=m.layout_token_ids,
        layout_token_identity=m.layout_token_identity)
    d.is_layout_token = lambda ind, _d=d: OCRDataset.is_layout_token(_d, ind)
    return d


# --- D7 -----------------------------------------------------------------------------

def test_specials_never_alias_a_real_character():
    m = make_manager(CHARSET)
    assert min(m.tokens.values()) >= len(m.charset), (
        "a special token was allocated inside the charset: {} with |charset|={}".format(
            m.tokens, len(m.charset)))


def test_new_tokens_is_refused_rather_than_silently_aliasing():
    with pytest.raises(ValueError, match="new_tokens is not supported"):
        make_manager(CHARSET, new_tokens=["⓪", "⓫", "⓬"])


def test_char_only_set_is_derived_from_the_final_charset():
    m = make_manager(CHARSET)
    assert len(m.char_only_set) == len(m.charset) - len(m.sem_tokens) - 1  # minus "\n"
    assert "\n" not in m.char_only_set
    assert not set(m.sem_tokens) & set(m.char_only_set)


# --- D2 -----------------------------------------------------------------------------

def test_layout_token_ids_are_exactly_the_semantic_tokens():
    m = make_manager(CHARSET)
    assert m.layout_token_ids == frozenset(m.charset.index(c) for c in m.sem_tokens)
    assert m.layout_token_ids  # this fixture has semantic tokens


def test_positional_test_misclassifies_the_character_below_the_semantic_block():
    """The defect itself, pinned so it cannot be reintroduced unnoticed."""
    d = as_dataset(make_manager(CHARSET, identity=False))
    em = d.charset.index("—")
    assert em == len(d.char_only_set), "fixture no longer exercises the boundary"
    assert d.is_layout_token(em), "shipped behaviour: the em dash reads as a layout token"


def test_identity_test_keeps_the_character_and_moves_nothing_else():
    off = as_dataset(make_manager(CHARSET, identity=False))
    on = as_dataset(make_manager(CHARSET, identity=True))
    em = off.charset.index("—")
    assert not on.is_layout_token(em)
    moved = [i for i in range(max(on.tokens.values()) + 1)
             if off.is_layout_token(i) != on.is_layout_token(i)]
    assert moved == [em], "exactly one id may change classification, got {}".format(moved)


def test_identity_test_still_covers_semantic_and_special_tokens():
    on = as_dataset(make_manager(CHARSET, identity=True))
    for c in on.sem_tokens:
        assert on.is_layout_token(on.charset.index(c))
    for v in on.tokens.values():
        assert on.is_layout_token(v)
    assert not on.is_layout_token(on.charset.index("a"))


def test_flag_defaults_to_the_shipped_positional_test():
    params_config = make_manager(CHARSET).params["config"]
    assert "layout_token_identity" in params_config
    m = OCRDatasetManager({
        "dataset_class": object, "datasets": {}, "charset": CHARSET, "batch_size": 1,
        "config": {"padding_value": 0, "padding_token": None, "charset_mode": "seq2seq",
                   "constraints": ["add_eot", "add_sot", "hand_encoding"]},
    })
    assert m.layout_token_identity is False, "absent flag must mean shipped behaviour"


# --- the same two invariants against the real corpora, where they exist ---------------

REAL = ["READ_2016_page_sem_dan", "IAM_page", "RIMES_paragraph", "SaintGall_page"]


@pytest.mark.parametrize("name", REAL)
def test_real_corpus_boundary(name):
    lp = os.path.join(FORMATTED, name, "labels.pkl")
    if not os.path.exists(lp):
        pytest.skip("{} not formatted here".format(name))
    with open(lp, "rb") as f:
        charset = sorted(pickle.load(f)["charset"])
    off = as_dataset(make_manager(charset, identity=False))
    on = as_dataset(make_manager(charset, identity=True))
    assert min(off.tokens.values()) >= len(charset)
    moved = [i for i in range(max(off.tokens.values()) + 1)
             if off.is_layout_token(i) != on.is_layout_token(i)]
    # exactly one real character per corpus, and it is the one at the stale boundary
    assert moved == [len(off.char_only_set)], (name, moved)
    assert off.is_layout_token(moved[0]) and not on.is_layout_token(moved[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
