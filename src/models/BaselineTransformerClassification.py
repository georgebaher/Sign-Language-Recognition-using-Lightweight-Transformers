import torch
import torch.nn as nn

from src.models.positional_encoding import build_positional_encoding


class BaselineTransformerClassification(nn.Module):
    """Standard encoder-decoder Transformer (src == tgt) + masked-mean-pool classifier with selectable positional encoding and a NaN-safe path for fully-padded sequences."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, n_heads: int = 8,
                 num_layers: int = 6, dropout: float = 0.0, max_len: int = 500, pe: str = "sincos"):
        """
        pe: positional-encoding kind. One of 'sincos', 'learnable', 'none'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = build_positional_encoding(pe, hidden_dim, dropout, max_len)
        self.transformer = nn.Transformer(hidden_dim, n_heads, num_layers, num_layers,
                                          dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B, T, _ = x.shape

        # Pad mask: True where padding exists (computed on raw input before embedding)
        pad_mask = (x == -2).all(dim=-1)  # [B, T]
        is_fully_padded = pad_mask.all(dim=1)  # [B], True for sequences that are all padding

        # Replace missing values with 0
        x = x.clone()
        x[x == -2] = 0.0

        # Project raw features into the model's hidden space
        x = self.embedding(x)  # [B, T, hidden_dim]

        # Positional encoding (sin-cos / learnable / none — selected by `pe` flag at init)
        x = self.pos_encoding(x)

        # Safe encoding: skip transformer for fully-padded rows (avoid NaN softmax over all-(-inf))
        if is_fully_padded.any():
            memory = x.new_zeros(B, T, self.hidden_dim)
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = self.transformer(
                    x[valid], x[valid],
                    src_key_padding_mask=pad_mask[valid],
                    tgt_key_padding_mask=pad_mask[valid],
                    memory_key_padding_mask=pad_mask[valid],
                )
        else:
            memory = self.transformer(
                x, x,
                src_key_padding_mask=pad_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=pad_mask,
            )

        # Masked average pooling
        valid_mask = (~pad_mask).unsqueeze(-1).float()  # [B, T, 1]
        pooled = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-9)  # [B, hidden_dim]

        return self.classifier(pooled)
