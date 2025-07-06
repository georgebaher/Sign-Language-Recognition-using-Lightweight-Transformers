import torch
import math
import torch.nn as nn


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, w_pe=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.w_pe = w_pe

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] for batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.w_pe:
            return x
        x = x + self.pe[:, :x.size(1)]  # match [B, T, D]
        return self.dropout(x)


class BaselineTransformerClassification(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=150, n_heads=10, max_seq_len=500, num_layers=6, dropout=0.0, w_pe=True):
        print(f"[INFO] Initializing Baseline Transformer with {n_heads} heads and {hidden_dim} hidden_dim.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.output_pos_encoding = PositionalEncodingSinCos(d_model=hidden_dim, dropout=dropout, w_pe=w_pe)
        self.transformer = nn.Transformer(hidden_dim, n_heads, num_layers, num_layers, batch_first=True)
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        print(f"[INFO] Transformer model initialized with {'no ' if not w_pe else ''}positional encoding")

    def forward(self, inputs):
        h = inputs.float()  # [B, T, D]
        # print(h.shape)

        # Create a mask where all features in a timestep are -2 → it's a padding frame
        src_key_padding_mask = (inputs == -2).all(dim=-1)  # shape: [batch_size, seq_len]


        # Replace all -2 values (missing features) with 0
        h[h == -2] = 0.0

        # # Subtract 0.5 from all features
        # h = h - 0.5

        # # 🔍 Check input variation per sample
        # print("Input mean:", h.mean(dim=[1, 2]))
        # print("Input std:", h.std(dim=[1, 2]))

        # Apply positional encoding
        henc = self.output_pos_encoding(h)

        h = self.transformer(henc, henc, src_key_padding_mask=src_key_padding_mask)

        # Temporal average pooling
        pooled = torch.mean(h, dim=1)  # [B, 1, D] and automatically the 1 is squeezed out, so it becomes [B, D]
        # print(f"Pooled representations {pooled.shape}:", pooled)

        res = self.linear_class(pooled)  # [B, n_classes]
        return res
