#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.
#
#  Modified by Mohammed Hamdan, 2025-2026. This file is a derivative of the upstream
#  file named in release/NOTICE.md section 1.2, which also records the measured line
#  identity against the pinned upstream commit. The modifications remain governed by
#  CeCILL-C; see release/NOTICE.md and release/licenses/LICENSE-CeCILL-C.md.

import os

import torch
from torch import relu
from torch.nn import Conv1d, Dropout
from torch.nn import Embedding
from torch.nn import Module
from torch.nn.init import xavier_uniform_, zeros_
from hand.models.baseline.attention import FeaturesUpdater, GlobalAttDecoder, PositionalEncoding1D, PositionalEncoding1DOnTheFly


# Debug NaN/Inf probes. Each `tensor.any()` inside an `if` is a host-device synchronisation;
# there were ten per forward pass. They are now opt-in: HAND_DEBUG_CHECKS=1.
DEBUG_CHECKS = os.environ.get("HAND_DEBUG_CHECKS", "0") == "1"


# Exact single-step mask construction (see the comment in forward). Settable to 0 through
# the environment so the A/B can be measured; it changes cost only, never output.
FAST_MASKS = os.environ.get("HAND_FAST_MASKS", "1") != "0"
# Exact single-step token window (see the comment in forward). Same contract as FAST_MASKS:
# it changes cost only, never output, and HAND_FAST_STEP=0 restores the reference path.
FAST_STEP = os.environ.get("HAND_FAST_STEP", "1") != "0"


class GlobalHTADecoder(Module):
    """
    HAND decoder module
    """
    def __init__(self, params):
        super(GlobalHTADecoder, self).__init__()
        self.params = params
        self.enc_dim = params["enc_dim"]
        self.dec_l_max = params["l_max"]

        self.dropout = Dropout(params["dec_pred_dropout"])
        self.dec_att_win = params["attention_win"] if params["attention_win"] is not None else 1

        self.features_updater = FeaturesUpdater(params)
        self.att_decoder = GlobalAttDecoder(params)

        self.emb = Embedding(num_embeddings=params["vocab_size"]+3, embedding_dim=self.enc_dim)

        self.use_line_indices = params["use_line_indices"] if "use_line_indices" in params else False
        if self.use_line_indices:
            if params["two_step_pos_enc_mode"] == "add":
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim, params["device"])
            else:
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim // 2, params["device"])
        else:
            self.pe_1d = PositionalEncoding1D(self.enc_dim, self.dec_l_max, params["device"])

        # Output classes: vocab_size + additional_tokens (default 3 for eot, sot, pad)
        additional_tokens = params.get("additional_tokens", 3)
        output_vocab_size = params["vocab_size"] + additional_tokens
        self.end_conv = Conv1d(self.enc_dim, output_vocab_size, kernel_size=1)

        # Initialize weights to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with xavier uniform to prevent NaN during training"""
        # Initialize embedding with small values
        torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        # Initialize output conv
        xavier_uniform_(self.end_conv.weight)
        zeros_(self.end_conv.bias)

        # Verify no NaN in initialized weights
        if torch.isnan(self.emb.weight).any():
            print("WARNING: NaN in embedding weights after init!", flush=True)
            # Force re-init with zeros + small noise
            self.emb.weight.data.zero_()
            self.emb.weight.data.add_(torch.randn_like(self.emb.weight) * 0.01)
        print(f"Decoder initialized: emb size={self.emb.weight.shape}, end_conv size={self.end_conv.weight.shape}", flush=True)

    def reset_mem_kv_cache(self):
        """Drop the per-document cross-attention memory cache (see attention.py) and the
        memoised encoder-padding mask, which is also constant for one document."""
        self.att_decoder.reset_mem_kv_cache()
        self._enc_mask_cache = None

    def forward(self, raw_features_1d, enhanced_features_1d, tokens, reduced_size, token_len, features_size, start=0, padding_value=None,
                cache=None, num_pred=None, keep_all_weights=False, line_indices=None, index_in_lines=None, depth=None, coverage_bias=None,
                use_mem_cache=False):
        device = raw_features_1d.device

        # Single-step token window. The reference path embeds and positionally-encodes the
        # ENTIRE prefix at every decoding step -- up to `max_char_prediction` tokens -- and then
        # keeps only the last `num_tokens_to_keep <= dec_att_win` of it (the causal window). That
        # is O(T) work per step and O(T^2) per page for a slice that never exceeds 100 entries,
        # and it is the last input whose shape grows with the prefix, which is what blocks
        # static-shape capture of the step.
        #
        # Passing only the kept window with the positional encoding's `start` offset advanced by
        # the number of dropped tokens is ARITHMETICALLY IDENTICAL: PositionalEncoding1D adds
        # pe[:, :, start : start + L], so a window beginning at absolute position start + (T - ntk)
        # receives exactly the encodings the full computation would have given it. Verified by
        # token-identical greedy output (`tools/efficiency_bench.py`).
        fast_step = (FAST_STEP and not self.training
                     and not self.use_line_indices and isinstance(start, int)
                     and self.dec_att_win is not None and num_pred is not None)
        if fast_step:
            ntk_pre = min(num_pred + self.dec_att_win - 1, tokens.size(1), int(max(token_len)))
            start = start + (tokens.size(1) - ntk_pre)
            tokens_win = tokens[:, -ntk_pre:]
        else:
            tokens_win = tokens

        # Token to Embedding
        emb_tokens = self.emb(tokens_win).permute(0, 2, 1)

        # Debug: Check embedding output
        if DEBUG_CHECKS and (torch.isnan(emb_tokens).any() or torch.isinf(emb_tokens).any()):
            print(f"  NaN/Inf in emb_tokens! tokens min/max: {tokens_win.min().item()}/{tokens_win.max().item()}", flush=True)

        if self.use_line_indices:
            pe_line = self.pe_1d(line_indices)
            pe_index = self.pe_1d(index_in_lines)
            # Debug: Check PE outputs
            if DEBUG_CHECKS and (torch.isnan(pe_line).any() or torch.isinf(pe_line).any()):
                print(f"  NaN/Inf in pe_line! line_indices min/max: {line_indices.min().item()}/{line_indices.max().item()}", flush=True)
            if DEBUG_CHECKS and (torch.isnan(pe_index).any() or torch.isinf(pe_index).any()):
                print(f"  NaN/Inf in pe_index! index_in_lines min/max: {index_in_lines.min().item()}/{index_in_lines.max().item()}", flush=True)
            if self.params["two_step_pos_enc_mode"] == "cat":
                pos_tokens = emb_tokens + torch.cat([pe_line, pe_index], dim=1)
            else:
                pos_tokens = emb_tokens + pe_line + pe_index
        else:
            # Add 1D Positional Encoding
            pos_tokens = self.pe_1d(emb_tokens, start=start)

        # Debug: Check pos_tokens
        if DEBUG_CHECKS and (torch.isnan(pos_tokens).any() or torch.isinf(pos_tokens).any()):
            print(f"  NaN/Inf in pos_tokens after PE!", flush=True)

        pos_tokens = pos_tokens.permute(2, 0, 1)

        if num_pred is None:
            num_pred = tokens.size(1)

        # Use cache values to avoid useless computation at eval time
        if self.dec_att_win > 1 and cache is not None:
            cache = cache[:, -self.dec_att_win + 1:]
        else:
            cache = None
        num_tokens_to_keep = num_pred if self.dec_att_win is None else min([num_pred + self.dec_att_win - 1, pos_tokens.size(0), max(token_len)])

        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        memory_mask = None  # Use all feature position

        # Mask construction. The reference path builds three masks from scratch at every call
        # and then throws almost all of them away: `generate_target_mask` allocates two (T, T)
        # bool tensors plus a tril, a triu and two logical ops, only for the last `num_pred`
        # rows and last `num_tokens_to_keep` columns to be kept. Inside an incremental decode
        # that is O(T^2) work per step and O(T^3) per page, for a slice of at most
        # (1, dec_att_win). The single-step path below produces the IDENTICAL slice without
        # building the square:
        #
        #   the kept row is i' = T-1 and the kept columns are j' = T-ntk .. T-1, so the causal
        #   (tril) condition j' <= i' holds for every kept column, and the window (triu,
        #   diagonal -w+1) condition j' >= i'-w+1 holds whenever ntk <= w, which is exactly how
        #   num_tokens_to_keep is defined when num_pred == 1. The slice is therefore all-False.
        #
        # `key_memory_mask` depends only on the feature geometry, which is constant for a
        # document, so it is memoised beside the cross-attention memory cache and dropped by
        # reset_mem_kv_cache(). `key_target_mask` is computed on the kept columns only.
        # Everything here is exact; `tools/efficiency_bench.py --equivalence` checks it by
        # requiring token-identical greedy output.
        fast = (FAST_MASKS and not self.training and self.dec_att_win is not None
                and num_pred <= num_tokens_to_keep <= num_pred + self.dec_att_win - 1)
        if fast:
            # The kept slice is a band. Query i is absolute T-num_pred+i and key j is absolute
            # T-ntk+j, so the causal condition j' <= i' is j <= i + (ntk - num_pred) and the
            # window condition j' >= i'-w+1 is j >= i + (ntk - num_pred) - w + 1. Building those
            # two comparisons costs two aranges instead of a (T x T) square. For num_pred == 1
            # and ntk == w the band covers every key and the slice is all-False, which is the
            # single-token case this path handled before.
            off = num_tokens_to_keep - num_pred
            qi = torch.arange(num_pred, device=device).unsqueeze(1)
            kj = torch.arange(num_tokens_to_keep, device=device).unsqueeze(0)
            target_mask = (kj > qi + off) | (kj < qi + off - self.dec_att_win + 1)
            key_target_mask = torch.eq(tokens[:, -num_tokens_to_keep:], padding_value) \
                if padding_value is not None \
                else torch.zeros((tokens.size(0), num_tokens_to_keep), dtype=torch.bool, device=device)
            key = (tuple(features_size), tuple(tuple(r) for r in reduced_size))
            if getattr(self, "_enc_mask_cache", None) is not None and self._enc_mask_cache[0] == key:
                key_memory_mask = self._enc_mask_cache[1]
            else:
                key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)
                self._enc_mask_cache = (key, key_memory_mask)
        else:
            target_mask = self.generate_target_mask(tokens.size(1), device)
            key_target_mask = self.generate_token_mask(tokens, token_len, device, padding_value)
            key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)
            target_mask = target_mask[..., -num_pred:, -num_tokens_to_keep:]
            key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.att_decoder(pos_tokens, memory_key=enhanced_features_1d,
                                        memory_value=raw_features_1d,
                                        tgt_mask=target_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=key_target_mask,
                                        memory_key_padding_mask=key_memory_mask,
                                        use_cache=True,
                                        cache=cache,
                                        predict_last_n_only=num_pred,
                                        keep_all_weights=keep_all_weights,
                                        depth=depth,
                                        coverage_bias=coverage_bias,
                                        use_mem_cache=use_mem_cache)

        # Debug: Check attention output
        if DEBUG_CHECKS and (torch.isnan(output).any() or torch.isinf(output).any()):
            print(f"  NaN/Inf in attention decoder output!", flush=True)

        dp_output = self.dropout(relu(output))
        preds = self.end_conv(dp_output.permute(1, 2, 0))

        # Numerical stability: clamp logits to prevent overflow in CrossEntropyLoss
        # Very large logits can cause softmax overflow (exp overflow)
        preds = torch.clamp(preds, min=-100.0, max=100.0)

        # Replace NaN/Inf logits with zeros to prevent a cascade. Unconditional (no host
        # sync); identical to the former guarded version whenever no NaN/Inf is present.
        if DEBUG_CHECKS and (torch.isnan(preds).any() or torch.isinf(preds).any()):
            print(f"  NaN/Inf in final preds! dp_output ok: {not torch.isnan(dp_output).any()}", flush=True)
        preds = torch.where(torch.isfinite(preds), preds, torch.zeros_like(preds))

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])
        return output, preds, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        """
        Generate mask for encoded features
        """
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, tokens, token_len, device, padding_value):
        """
        Generate mask for tokens per sample
        """
        batch_size, len_max = tokens.size()
        mask = torch.zeros((batch_size, len_max), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False
        mask[tokens == padding_value] = True
        return mask

    def generate_target_mask(self, target_len, device, correction_pass=False):
        """
        Generate mask for tokens per time step (teacher forcing)
        """
        if correction_pass:
            return torch.zeros((target_len, target_len), dtype=torch.bool, device=device)
        else:
            return torch.logical_not(
                torch.logical_and(torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                  torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_att_win+1)))

