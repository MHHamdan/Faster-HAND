"""
Equivalence tests for the Stage 0 synchronisation removals.

Each replaced function is checked against the reference implementation it replaced,
which is kept in the code base for exactly this purpose. Run with:

    python -m pytest tests -q      or      python tests/test_sync_free_equivalence.py
"""
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.OCR.document_OCR.hand.trainer_hand import Manager as DANManager  # noqa: E402
from hand.OCR.document_OCR.hand.trainer_std_hand import Manager  # noqa: E402
from hand.models.baseline.attention import CustomMultiHeadAttention  # noqa: E402


def fake_manager(charset, n_char_only):
    """A stand-in with the three dataset attributes the functions read.

    NOTE: char_only_set is a *prefix* here, which the real construction never produces --
    OCRDatasetManager removes "\\n" (which sorts to index 0) and the semantic tokens from a
    copy of the charset. A prefix cannot exhibit the D2 off-by-one, so anything that depends
    on where the boundary lands must use realistic_manager() below.
    """
    ds = types.SimpleNamespace(charset=charset, char_only_set=charset[:n_char_only],
                               tokens={"pad": len(charset) + 2, "start": len(charset) + 1, "end": len(charset)})
    return types.SimpleNamespace(dataset=ds)


def realistic_manager(charset, sem_tokens, identity):
    """char_only_set built the way OCRDatasetManager builds it: the charset minus the
    semantic tokens minus "\\n". The boundary then lands one index below the last real
    character, which is the condition D2 addresses."""
    sem = set(sem_tokens)
    ds = types.SimpleNamespace(
        charset=charset,
        char_only_set=[c for c in charset if c not in sem and c != "\n"],
        tokens={"pad": len(charset) + 2, "start": len(charset) + 1, "end": len(charset)},
        layout_token_ids=frozenset(charset.index(c) for c in sem_tokens),
        layout_token_identity=identity)
    return types.SimpleNamespace(dataset=ds)


def random_tokens(rng, b, t, charset, pad, p_pad=0.15, p_layout=0.15, p_nl=0.1):
    nl = charset.index("\n")
    n_char_only = len(charset) - 4
    toks = rng.integers(0, n_char_only, size=(b, t))
    r = rng.random((b, t))
    toks[r < p_layout] = rng.integers(n_char_only, len(charset), size=(r < p_layout).sum())
    toks[(r >= p_layout) & (r < p_layout + p_nl)] = nl
    # padding at the tail (as in real batches) plus a few stray pads inside
    for i in range(b):
        L = rng.integers(1, t + 1)
        toks[i, L:] = pad
    stray = rng.random((b, t)) < p_pad * 0.2
    toks[stray] = pad
    return torch.as_tensor(toks, dtype=torch.long)


def test_line_indices_match_reference():
    rng = np.random.default_rng(0)
    charset = [chr(97 + i) for i in range(20)] + ["\n", "ⓐ", "ⓑ", "ⓒ", "ⓓ"]   # 21 char-only incl. newline? no: newline excluded
    m = fake_manager(charset, n_char_only=20)
    pad = m.dataset.tokens["pad"]
    for trial in range(200):
        b, t = int(rng.integers(1, 6)), int(rng.integers(1, 60))
        toks = random_tokens(rng, b, t, charset, pad)
        li_ref, ii_ref = Manager.get_line_indices_from_tokens_reference(m, toks)
        li, ii = Manager.get_line_indices_from_tokens(m, toks)
        assert torch.equal(li, li_ref), (trial, toks, li, li_ref)
        assert torch.equal(ii, ii_ref), (trial, toks, ii, ii_ref)


def test_line_indices_match_reference_realistic_charset():
    """Same equivalence, but on a charset shaped like a real one (newline at the front,
    semantic tokens at the tail) and under both settings of layout_token_identity."""
    rng = np.random.default_rng(1)
    sem = ["ⓐ", "ⓑ", "ⓝ", "ⓟ", "ⓢ"]
    charset = sorted(["\n"] + [chr(97 + i) for i in range(20)] + ["—"] + sem)
    for identity in (False, True):
        m = realistic_manager(charset, sem, identity)
        pad = m.dataset.tokens["pad"]
        for trial in range(100):
            b, t = int(rng.integers(1, 6)), int(rng.integers(1, 60))
            toks = random_tokens(rng, b, t, charset, pad)
            li_ref, ii_ref = Manager.get_line_indices_from_tokens_reference(m, toks)
            li, ii = Manager.get_line_indices_from_tokens(m, toks)
            assert torch.equal(li, li_ref), (identity, trial, toks, li, li_ref)
            assert torch.equal(ii, ii_ref), (identity, trial, toks, ii, ii_ref)


def test_live_line_indices_are_unchanged_by_the_layout_token_flag():
    """The live path's `<=` test is accidentally correct -- it compensates for exactly the
    one removed front element -- so switching it to the identity test must move nothing.
    This is what licenses leaving the flag off for every existing checkpoint."""
    rng = np.random.default_rng(2)
    sem = ["ⓐ", "ⓑ", "ⓝ", "ⓟ", "ⓢ"]
    charset = sorted(["\n"] + [chr(97 + i) for i in range(20)] + ["—"] + sem)
    off = realistic_manager(charset, sem, False)
    on = realistic_manager(charset, sem, True)
    pad = off.dataset.tokens["pad"]
    # exhaustive over every token id, then randomised batches
    ids = torch.arange(0, pad + 1).unsqueeze(0)
    for a, b in zip(Manager.get_line_indices_from_tokens(off, ids),
                    Manager.get_line_indices_from_tokens(on, ids)):
        assert torch.equal(a, b)
    for _ in range(100):
        toks = random_tokens(rng, int(rng.integers(1, 6)), int(rng.integers(1, 60)), charset, pad)
        for a, b in zip(Manager.get_line_indices_from_tokens(off, toks),
                        Manager.get_line_indices_from_tokens(on, toks)):
            assert torch.equal(a, b)


def test_error_injection_distribution():
    torch.manual_seed(0)
    charset = [chr(97 + i) for i in range(30)]
    m = fake_manager(charset, n_char_only=30)
    b, t = 64, 200
    y = torch.randint(0, len(charset), (b, t))
    y_len = [int(v) for v in torch.randint(2, t + 1, (b,))]
    for rate in (0.0, 0.2, 0.5):
        out = DANManager.add_error_in_gt(m, y, y_len, rate)
        assert out.shape == y.shape and out.dtype == y.dtype
        assert torch.equal(out[:, 0], y[:, 0]), "position 0 must never be corrupted"
        for i in range(b):
            assert torch.equal(out[i, y_len[i]:], y[i, y_len[i]:]), "beyond y_len must be untouched"
        assert int(out.min()) >= 0 and int(out.max()) < len(charset)
        changed = (out != y).float()
        in_range = torch.zeros_like(changed)
        for i in range(b):
            in_range[i, 1:y_len[i]] = 1
        # a replacement drawn uniformly equals the original with prob 1/|charset|
        expected = rate * (1 - 1 / len(charset))
        observed = float(changed.sum() / in_range.sum())
        assert abs(observed - expected) < 0.02, (rate, observed, expected)
        # reference implementation, same expectation
        np.random.seed(0)
        ref = DANManager.add_error_in_gt_reference(m, y, y_len, rate)
        observed_ref = float(((ref != y).float()).sum() / in_range.sum())
        assert abs(observed_ref - expected) < 0.02, (rate, observed_ref, expected)


def test_attention_nan_replacement_is_branch_free_equivalent():
    torch.manual_seed(0)
    att = CustomMultiHeadAttention(embed_dim=16, num_heads=2, proj_value=True, dropout=0.0).eval()
    q = torch.randn(5, 3, 16); k = torch.randn(7, 3, 16)
    # fully masked key row for sample 1 -> softmax row is all NaN -> uniform replacement
    kpm = torch.zeros(3, 7, dtype=torch.bool); kpm[1] = True
    out, w = att(q, k, k, key_padding_mask=kpm, output_weights=True)
    assert torch.isfinite(out).all() and torch.isfinite(w).all()
    assert torch.allclose(w[1], torch.full_like(w[1], 1.0 / 7))
    # unmasked samples are the plain softmax attention
    out2, w2 = att(q, k, k, key_padding_mask=None, output_weights=True)
    assert torch.allclose(w[0], w2[0]) and torch.allclose(out[:, 0], out2[:, 0])


if __name__ == "__main__":
    test_line_indices_match_reference(); print("line indices: ok")
    test_line_indices_match_reference_realistic_charset(); print("line indices, realistic charset: ok")
    test_live_line_indices_are_unchanged_by_the_layout_token_flag(); print("layout flag is a no-op on the live path: ok")
    test_error_injection_distribution(); print("error injection: ok")
    test_attention_nan_replacement_is_branch_free_equivalent(); print("attention nan path: ok")
