"""Speculative draft heads for lossless multi-token decoding.

One head per look-ahead offset k = 1 .. m-1. Head k maps the decoder's final hidden state at
input position t -- the same tensor the output layer consumes -- to a distribution over the
token at position t + 1 + k. Head 0 does not exist: the token at t + 1 is what the frozen
model's own output layer already predicts.

The heads are trained on a FROZEN base model, so they cannot change what the base predicts.
At decoding time their proposals are verified against the base model's own distributions and
a mismatch is discarded, which is why the decoded string is identical to the base model's
greedy output by construction rather than by measurement.

Parameters, m = 3, d = 256, vocab-out = 100: 2 x (256*256 + 256 + 256*100 + 100) = 182,984,
i.e. +2.6 % on a 7,033,700-parameter model. `hidden=0` gives a linear head at 25,700 each.
"""
import torch
from torch.nn import GELU, Linear, Module, ModuleList


class SpeculativeHeads(Module):
    def __init__(self, d_model, vocab_out, m, hidden=256):
        super().__init__()
        assert m >= 2, "m = 1 is ordinary single-token decoding; no heads are needed"
        self.m = m
        self.hidden = hidden
        if hidden:
            self.body = ModuleList([Linear(d_model, hidden) for _ in range(m - 1)])
            self.act = GELU()
            self.out = ModuleList([Linear(hidden, vocab_out) for _ in range(m - 1)])
        else:
            self.body = None
            self.out = ModuleList([Linear(d_model, vocab_out) for _ in range(m - 1)])

    def forward(self, hidden_state):
        """hidden_state: (..., d_model) -> list of (m-1) logit tensors (..., vocab_out)."""
        if self.body is None:
            return [o(hidden_state) for o in self.out]
        return [self.out[k](self.act(self.body[k](hidden_state))) for k in range(self.m - 1)]

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def draft_tokens(heads, hidden_state):
    """Greedy draft of the next m-1 tokens from one hidden state (d_model,) or (1, d_model)."""
    with torch.no_grad():
        return [int(l.argmax(-1).reshape(-1)[0]) for l in heads(hidden_state)]
