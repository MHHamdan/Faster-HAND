"""The exact decode-path optimisations must change cost only, never output.

`HAND_FAST_MASKS` and `HAND_FAST_STEP` replace two pieces of per-step work with closed forms:

  * the (T x T) causal/window mask, built and then sliced to (num_pred, ntk), is built directly
    as a band from two aranges;
  * the embedding and positional encoding of the whole prefix, then sliced to the causal window,
    are computed on the window alone with the positional encoding's `start` advanced.

Both are exact for every (T, num_pred) a decode can produce, and this test holds them to that on
CPU so a regression is caught without a GPU. The end-to-end gate on real pages is
`tools/verify_exact_decoding.py`.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hand.models.baseline import dan_decoder  # noqa: E402
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: E402

WIN = 100


def build(dec_layers=2, dim=64, heads=4, vocab=20):
    return GlobalHTADecoder(dict(
        input_channels=3, dropout=0.5, enc_dim=dim, nb_layers=5, vocab_size=vocab,
        additional_tokens=1, pe_h_max=64, pe_w_max=64, l_max=4000,
        dec_num_layers=dec_layers, dec_num_heads=heads, dec_res_dropout=0.0,
        dec_pred_dropout=0.0, dec_att_dropout=0.0, dec_dim_feedforward=dim,
        attention_win=WIN, use_tokens_from_all_lines=True, use_first_pass_tokens=True,
        use_line_indices=False, two_step_pos_enc_mode="cat", device="cpu")).eval()


def band_reference(T, num_pred, ntk):
    m = torch.logical_not(torch.logical_and(
        torch.tril(torch.ones((T, T), dtype=torch.bool), diagonal=0),
        torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=-WIN + 1)))
    return m[..., -num_pred:, -ntk:]


@pytest.mark.parametrize("T", [1, 2, 5, 99, 100, 101, 150, 477])
@pytest.mark.parametrize("num_pred", [1, 2, 3, 5, 8])
def test_band_mask_matches_the_sliced_square(T, num_pred):
    ntk = min(num_pred + WIN - 1, T)
    if not (num_pred <= ntk <= num_pred + WIN - 1):
        pytest.skip("guarded off; the reference path runs")
    off = ntk - num_pred
    qi = torch.arange(num_pred).unsqueeze(1)
    kj = torch.arange(ntk).unsqueeze(0)
    fast = (kj > qi + off) | (kj < qi + off - WIN + 1)
    assert torch.equal(fast, band_reference(T, num_pred, ntk))


@pytest.mark.parametrize("T", [1, 7, 120, 250])
def test_teacher_forced_output_is_identical_with_and_without_the_fast_paths(T):
    """Full-sequence call, as training makes it: num_pred == T and no cache.

    (num_pred < T with cache=None is not a configuration the model is ever called with -- the
    layers past the first would receive only num_pred entries while the mask still describes the
    whole window -- so it is not tested. The incremental case below covers num_pred < T, with
    the cache the decoder requires there.)
    """
    num_pred = T
    torch.manual_seed(0)
    dec = build()
    s, b, c = 40, 1, 64
    feats = torch.randn(s, b, c)
    tokens = torch.randint(0, 19, (b, T))
    token_len = torch.tensor([T], dtype=torch.int)
    fsize = (b, c, 5, 8)
    reduced = [(5, 8)]
    out = {}
    for fast in (False, True):
        dan_decoder.FAST_MASKS = dan_decoder.FAST_STEP = fast
        with torch.no_grad():
            h, pred, cache, _ = dec(feats, feats, tokens, reduced, token_len, fsize, start=0,
                                    padding_value=19, cache=None, num_pred=num_pred)
        out[fast] = (h, pred)
    dan_decoder.FAST_MASKS = dan_decoder.FAST_STEP = True
    assert torch.equal(out[False][0], out[True][0]), "hidden states differ"
    assert torch.equal(out[False][1], out[True][1]), "logits differ"


@pytest.mark.parametrize("num_pred", [1, 3, 5])
def test_incremental_decode_is_identical_with_and_without_the_fast_paths(num_pred):
    """The case that matters: a cached, step-by-step decode, as inference actually runs."""
    torch.manual_seed(0)
    dec = build()
    s, b, c = 40, 1, 64
    feats = torch.randn(s, b, c)
    fsize, reduced = (b, c, 5, 8), [(5, 8)]
    seqs = {}
    for fast in (False, True):
        dan_decoder.FAST_MASKS = dan_decoder.FAST_STEP = fast
        toks = torch.ones((b, num_pred), dtype=torch.long)
        cache = None
        emitted = []
        with torch.no_grad():
            # Each call consumes num_pred new positions and returns num_pred distributions, so
            # the cache and the sequence stay in step -- the shape a speculative verify pass has.
            for _ in range(130 // num_pred + 1):
                _, pred, cache, _ = dec(feats, feats, toks, reduced,
                                        torch.tensor([toks.size(1)], dtype=torch.int), fsize,
                                        start=0, cache=cache, num_pred=num_pred,
                                        padding_value=19)
                nxt = pred[0].argmax(0)                      # (num_pred,)
                toks = torch.cat([toks, nxt.unsqueeze(0)], dim=1)
                emitted.extend(int(t) for t in nxt)
        seqs[fast] = emitted
    dan_decoder.FAST_MASKS = dan_decoder.FAST_STEP = True
    assert seqs[False] == seqs[True], "incremental decode diverged"


# ---------------------------------------------------------------------------------------------
# E4 -- shared visual memory K/V projections
# ---------------------------------------------------------------------------------------------

def _decoder(share, layers=8, dim=256, vocab=99):
    return GlobalHTADecoder(dict(
        input_channels=3, dropout=0.5, enc_dim=dim, nb_layers=5, vocab_size=vocab,
        additional_tokens=1, pe_h_max=64, pe_w_max=64, l_max=4000,
        dec_num_layers=layers, dec_num_heads=4, dec_res_dropout=0.0, dec_pred_dropout=0.0,
        dec_att_dropout=0.0, dec_dim_feedforward=dim, attention_win=WIN,
        use_tokens_from_all_lines=True, use_first_pass_tokens=True, use_line_indices=False,
        two_step_pos_enc_mode="cat", device="cpu", dec_share_memory_kv=share)).eval()


def test_sharing_memory_kv_removes_exactly_the_expected_parameters():
    """7 layers x (one key + one value projection) at d = 256 -> 921,088 fewer parameters."""
    n = lambda m: sum(p.numel() for p in m.parameters())
    base, shared = n(_decoder(False)), n(_decoder(True))
    per_proj = 256 * 256 + 256
    assert base - shared == 7 * 2 * per_proj == 921088, (base, shared, base - shared)


def test_sharing_is_off_by_default_so_existing_models_are_untouched():
    params = dict(
        input_channels=3, dropout=0.5, enc_dim=256, nb_layers=5, vocab_size=99,
        additional_tokens=1, pe_h_max=64, pe_w_max=64, l_max=4000, dec_num_layers=8,
        dec_num_heads=4, dec_res_dropout=0.0, dec_pred_dropout=0.0, dec_att_dropout=0.0,
        dec_dim_feedforward=256, attention_win=WIN, use_tokens_from_all_lines=True,
        use_first_pass_tokens=True, use_line_indices=False, two_step_pos_enc_mode="cat",
        device="cpu")                                   # no dec_share_memory_kv key at all
    assert sum(p.numel() for p in GlobalHTADecoder(params).parameters()) == 5327460


def test_shared_projections_are_one_object_and_the_model_still_runs():
    dec = _decoder(True, layers=4, dim=64)
    ks = {id(l.att.in_proj_k) for l in dec.att_decoder.decoder_layers}
    vs = {id(l.att.in_proj_v) for l in dec.att_decoder.decoder_layers}
    assert len(ks) == 1 and len(vs) == 1, "projections are not shared across layers"
    assert len({id(l.att.in_proj_q) for l in dec.att_decoder.decoder_layers}) == 4, \
        "queries must stay per-layer, or the layers cannot attend differently"
    feats = torch.randn(30, 1, 64)
    with torch.no_grad():
        _, pred, _, _ = dec(feats, feats, torch.randint(0, 90, (1, 12)), [(5, 6)],
                            torch.tensor([12], dtype=torch.int), (1, 64, 5, 6), start=0,
                            padding_value=98, num_pred=12)
    assert torch.isfinite(pred).all()
