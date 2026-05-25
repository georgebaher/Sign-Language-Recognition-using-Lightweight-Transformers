import math
import torch
import torch.nn as nn


class PositionalEncodingSinCos(nn.Module):
    """Standard "Attention Is All You Need" sin/cos positional encoding for batch_first tensors of shape [B, T, D]."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000, w_pe: bool = True):
        super().__init__()
        self.w_pe = w_pe
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model/2]
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe) # non-trainable buffer, moved to device with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dropout is applied whether PE is added or not, so the regularisation budget
        # stays the same when w_pe is toggled off.
        if self.w_pe:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncodingLearnable(nn.Module):
    """Learnable per-position embedding (BERT-style) for batch_first tensors of shape [B, T, D]."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))  # trainable
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def build_positional_encoding(kind: str, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> nn.Module:
    """Factory returning a PE module by kind: 'sincos', 'learnable', or 'none' (dropout-only passthrough)."""
    if kind == "sincos":
        return PositionalEncodingSinCos(d_model, dropout, max_len, w_pe=True)
    if kind == "learnable":
        return PositionalEncodingLearnable(d_model, dropout, max_len)
    if kind == "none":
        return PositionalEncodingSinCos(d_model, dropout, max_len, w_pe=False)
    raise ValueError(f"Unknown positional encoding kind: {kind!r}. Expected 'sincos', 'learnable', or 'none'.")
