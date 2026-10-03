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
import torch
from torch import relu, softmax
from torch.nn import Dropout,  Linear, LayerNorm
from torch.nn import ModuleList, Module
from torch.nn.init import xavier_uniform_


class PositionalEncoding1D(Module):

    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x


class PositionalEncoding1DOnTheFly(Module):

    def __init__(self, dim, device):
        super(PositionalEncoding1DOnTheFly, self).__init__()
        self.dim = dim
        self.div = torch.exp(-torch.arange(0., dim, 2, device=device) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(0)

    def forward(self, indices):
        # Clamp indices to reasonable range to prevent numerical overflow
        # Max position is 10000 to match the encoding range
        indices = torch.clamp(indices, min=0, max=10000)

        emb_indices = torch.zeros((indices.size(0), self.dim, indices.size(1)), device=indices.device)
        # Convert to float for computation
        indices_float = indices.float().unsqueeze(2)
        emb_indices[:, ::2, :] = torch.sin(indices_float * self.div).permute(0, 2, 1)
        emb_indices[:, 1::2, :] = torch.cos(indices_float * self.div).permute(0, 2, 1)
        return emb_indices


class PositionalEncoding2D(Module):

    def __init__(self, dim, h_max, w_max, device):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class CustomMultiHeadAttention(Module):
    """
    Re-implementation of Multi-head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0, proj_value=True):
        super().__init__()

        self.proj_value = proj_value

        self.in_proj_q = Linear(embed_dim, embed_dim)
        self.in_proj_k = Linear(embed_dim, embed_dim)
        if self.proj_value:
            self.in_proj_v = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = Dropout(dropout)
        # No re-initialisation here: the projections keep nn.Linear's default init, as in DAN.
        # The former self.init_weights() call (xavier on q/k/v) made page-level training from
        # the line-CTC init collapse to an input-independent decoder (Stage 1 arm A1;
        # see docs/ablations.md section 1.1).
        # init_weights() is kept as an unused method for parity with DAN's models_dan.py.

    def reset_mem_kv_cache(self):
        """Drop the cached memory projections. Call once per document."""
        self._mem_kv = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output_weights=True,
                coverage_bias=None, use_mem_cache=False):
        target_len, b, c = query.size()
        source_len = key.size(0)
        q = self.in_proj_q(query)
        # Memory K/V caching (inference only, opt-in). In cross-attention the key and value
        # are the document's visual features: constant for every decoding step of one page,
        # yet re-projected at every step by the reference implementation (and by DAN's).
        # With use_mem_cache the projection and the head reshape are done once per document.
        # Default False, so the training path and the frozen evaluation baseline are
        # untouched; correctness is asserted by token-identical greedy decoding
        # (tools/efficiency_bench.py --equivalence).
        cached = getattr(self, "_mem_kv", None) if use_mem_cache else None
        if cached is not None and cached[0].size(1) == source_len and cached[0].size(0) == b * self.num_heads:
            k, v = cached
            q = q * self.scale_factor
            q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        else:
            k = self.in_proj_k(key)
            v = self.in_proj_v(value) if self.proj_value else value
            q = q * self.scale_factor

            q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
            k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
            v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
            if use_mem_cache:
                self._mem_kv = (k, v)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # Coverage: discourage attending to source positions this decoder has already
        # consumed. `coverage_bias` is (B, source_len), already scaled by the caller, and
        # is subtracted from the logits for every head and every target position. Without
        # it an autoregressive decoder can re-attend to the same region indefinitely and
        # never emit end-of-text -- the failure the multi-page models exhibit, where
        # decoding runs to the character cap on essentially every sample.
        if coverage_bias is not None:
            cb = coverage_bias.unsqueeze(1).unsqueeze(1)                     # (B,1,1,S)
            cb = cb.expand(b, self.num_heads, 1, source_len).reshape(b * self.num_heads, 1, source_len)
            attn_output_weights = attn_output_weights - cb

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = torch.repeat_interleave(attn_mask.unsqueeze(1), self.num_heads, dim=1).reshape(b * self.num_heads, target_len, source_len)

            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(b, self.num_heads, target_len, source_len)

            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(b * self.num_heads, target_len, source_len)

        # Numerical stability: clamp attention weights to prevent overflow/underflow
        # This prevents softmax from producing NaN when attention scores are extreme
        attn_output_weights = torch.clamp(attn_output_weights, min=-1e4, max=1e4)

        attn_output_weights_raw = softmax(attn_output_weights, dim=-1)

        # Handle NaN in softmax output (can occur if all values in a row are -inf after masking)
        # Replace NaN with uniform attention (1/source_len). Done unconditionally: the
        # former `if nan_mask.any()` guard forced a host-device synchronisation on every
        # attention call (16 per decoder forward), and the result is identical either way.
        attn_output_weights_raw = torch.where(torch.isnan(attn_output_weights_raw),
                                              torch.full_like(attn_output_weights_raw, 1.0 / source_len),
                                              attn_output_weights_raw)

        attn_output_weights = self.dropout(attn_output_weights_raw)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if output_weights:
            attn_output_weights_raw = attn_output_weights_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weights_raw.sum(dim=1) / self.num_heads
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class GlobalDecoderLayer(Module):
    """
    Transformer Decoder Layer
    """

    def __init__(self, params):
        super(GlobalDecoderLayer, self).__init__()
        self.emb_dim = params["enc_dim"]
        self.dim_feedforward = params["dec_dim_feedforward"]

        self.self_att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.norm1 = LayerNorm(self.emb_dim)
        self.att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.linear1 = Linear(self.emb_dim, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, self.emb_dim)

        self.dropout = Dropout(params["dec_res_dropout"])

        self.norm2 = LayerNorm(self.emb_dim)
        self.norm3 = LayerNorm(self.emb_dim)

        # Initialize linear layers
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, predict_last_n_only=None,
                coverage_bias=None, use_mem_cache=False):

        if memory_value is None:
            memory_value = memory_key

        self_att_query = tgt[-predict_last_n_only:] if predict_last_n_only else tgt

        tgt2, weights_self = self.self_att(self_att_query, tgt, tgt, attn_mask=tgt_mask,
                                           key_padding_mask=tgt_key_padding_mask, output_weights=True)
        tgt = self_att_query + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        att_query = tgt

        tgt2, weights = self.att(att_query, memory_key, memory_value, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask, output_weights=True,
                                coverage_bias=coverage_bias, use_mem_cache=use_mem_cache)

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights, weights_self


class GlobalAttDecoder(Module):
    """
    Stack of transformer decoder layers
    """

    def __init__(self, params):
        super(GlobalAttDecoder, self).__init__()

        self.num_layers = params["dec_num_layers"]
        self.decoder_layers = ModuleList([GlobalDecoderLayer(params) for _ in range(self.num_layers)])

        # E4 -- shared visual memory K/V projections (opt-in, default off).
        # Cross-attention reads the SAME tensor at every layer: the document's visual memory,
        # which the K/V cache already projects once per document at inference. Giving every layer
        # its own key and value projection of one fixed tensor costs 2 d^2 parameters per layer
        # for a basis each layer could share; the per-layer query and output projections are left
        # alone, so layers can still attend differently. At d = 256 and 8 layers this removes
        # 7 x 2 x (256*256 + 256) = 921,088 parameters, 13.1 % of a 7,033,700-parameter model,
        # with no change to inference FLOPs or latency because those projections are cached.
        # Cross-layer K/V sharing is standard in recent language models; it has no HTR precedent,
        # which is why it is screened rather than assumed (track B gate; B2/B3 report first).
        if params.get("dec_share_memory_kv"):
            shared_k = Linear(params["enc_dim"], params["enc_dim"])
            shared_v = Linear(params["enc_dim"], params["enc_dim"])
            for layer in self.decoder_layers:
                layer.att.in_proj_k = shared_k
                layer.att.in_proj_v = shared_v

        # Adaptive depth: when enabled, training samples a prefix of the decoder stack for
        # each batch, so the same shared weights must produce usable output at several
        # depths. At inference the depth becomes a per-document compute/accuracy dial
        # rather than a fixed architectural constant. Disabled by default, so existing
        # checkpoints and the fixed-depth baseline are unaffected.
        self.elastic_depths = params.get("dec_elastic_depths") or None
        if self.elastic_depths:
            self.elastic_depths = sorted(
                {min(max(int(d), 1), self.num_layers) for d in self.elastic_depths})

    def sample_depth(self):
        """Depth to use for this forward pass (full depth unless training elastically)."""
        if self.training and self.elastic_depths:
            return self.elastic_depths[torch.randint(len(self.elastic_depths), (1,)).item()]
        return self.num_layers

    def reset_mem_kv_cache(self):
        for layer in self.decoder_layers:
            layer.att.reset_mem_kv_cache()

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=False,
                depth=None, coverage_bias=None, use_mem_cache=False):
        output = tgt
        # `depth` lets the caller pin the number of layers (inference-time budget); when
        # it is None we sample during elastic training and use the full stack otherwise.
        n_layers = self.sample_depth() if depth is None \
            else min(max(int(depth), 1), self.num_layers)
        active_layers = self.decoder_layers[:n_layers]
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }

        for i, dec_layer in enumerate(active_layers):
            output, weights, weights_self = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        predict_last_n_only=predict_last_n_only,
                                        coverage_bias=coverage_bias,
                                        use_mem_cache=use_mem_cache)
            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights)
        if use_cache:
            # Adaptive depth with an incremental cache (B5). The self-attention cache is
            # (layers, T, B, C): a step that runs FEWER layers than the cache holds would
            # leave the deeper layers with no history for later steps, and torch.cat would
            # fail on the layer dimension. The standard layer-skipping treatment is used --
            # the skipped layers are the identity, so the last computed layer's state fills
            # their slots. This is an APPROXIMATION and it is only ever reached when a caller
            # varies `depth` between steps; a fixed-depth run never enters it.
            if cache is not None and len(cache_t) < cache.size(0):
                cache_t = cache_t + [cache_t[-1]] * (cache.size(0) - len(cache_t))
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache


class FeaturesUpdater(Module):
    """
    Module that handle 2D positional encoding
    """
    def __init__(self, params):
        super(FeaturesUpdater, self).__init__()
        self.enc_dim = params["enc_dim"]
        self.enc_h_max = params["pe_h_max"]
        self.enc_w_max = params["pe_w_max"]
        self.pe_2d = PositionalEncoding2D(self.enc_dim, self.enc_h_max, self.enc_w_max, params["device"])
        self.use_pe_2d = ("dec_use_pe_2d" not in params) or params["dec_use_pe_2d"]

    def get_pos_features(self, features):
        if self.use_pe_2d:
            return self.pe_2d(features)
        return features
