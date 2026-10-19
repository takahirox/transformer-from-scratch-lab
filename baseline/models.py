"""Explicit attention and sequence classifiers; no nn.Transformer components."""

import math

import torch
from torch import nn


def scaled_attention(q, k, v, allowed=None):
    """Last axes are (query/key position, feature); True means allowed.

    Every query must have at least one allowed key. Fully masked rows raise
    instead of silently producing NaNs. Leading batch/head axes broadcast.
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise ValueError("attention mask must be boolean (True = allowed)")
        if not allowed.any(dim=-1).all():
            raise ValueError("each query needs at least one allowed key")
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by positive heads")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x, allowed=None):
        batch, length, width = x.shape

        def split(projection):
            # (B, L, D) -> (B, H, L, D/H)
            return projection(x).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        mixed, weights = scaled_attention(split(self.q), split(self.k), split(self.v), allowed)
        joined = mixed.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(joined), weights


class PositionalEncoding(nn.Module):
    def __init__(self, width, max_length):
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        table = torch.zeros(max_length, width)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency[:width // 2])
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.table.shape[1]:
            raise ValueError("sequence exceeds positional encoding capacity")
        return x + self.table[:, :x.shape[1]]


class TransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = MultiHeadAttention(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))

    def forward(self, x, allowed=None, return_attention=False):
        # Pre-normalization, with residual connections around both sublayers.
        mixed, weights = self.attention(self.norm1(x), allowed)
        x = x + mixed
        output = x + self.feed_forward(self.norm2(x))
        return (output, weights) if return_attention else output


class SequenceClassifier(nn.Module):
    """Predict the FIRST token from the FINAL position, across random distractors."""

    def __init__(self, kind, classes=8, width=32, heads=4, max_length=128):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(classes, width)
        if kind == "transformer":
            self.position = PositionalEncoding(width, max_length)
            self.sequence = TransformerBlock(width, heads)
            self.norm = nn.LayerNorm(width)
        elif kind in ("rnn", "lstm"):
            cell = nn.RNN if kind == "rnn" else nn.LSTM
            self.sequence = cell(width, width, batch_first=True)
        else:
            raise ValueError(f"unknown model: {kind}")
        self.classifier = nn.Linear(width, classes)

    def forward(self, tokens, return_attention=False):
        x = self.embedding(tokens)
        if self.kind == "transformer":
            x, weights = self.sequence(self.position(x), return_attention=True)
            x = self.norm(x)
        else:
            if return_attention:
                raise ValueError("attention is available only for the Transformer")
            x, _ = self.sequence(x)
        logits = self.classifier(x[:, -1])
        return (logits, weights) if return_attention else logits
