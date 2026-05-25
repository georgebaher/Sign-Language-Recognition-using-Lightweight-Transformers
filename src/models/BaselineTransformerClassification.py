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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, T, D = inputs.shape

        # Build masks
        src_key_padding_mask = (inputs == -2).all(dim=-1)  # [B, T] bool
        src_key_padding_mask = src_key_padding_mask.to(dtype=torch.bool, device=inputs.device)
        is_fully_padded = src_key_padding_mask.all(dim=1)  # [B]

        # Replace sentinel then add PE
        h = inputs.float().clone()
        h[h == -2] = 0.0
        henc = self.output_pos_encoding(h)  # [B, T, D]

        # Safe encoding: skip transformer for fully-padded rows
        if is_fully_padded.any():
            memory = henc.new_zeros(B, T, D)
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = self.transformer(
                    henc[valid], henc[valid],
                    src_key_padding_mask=src_key_padding_mask[valid],
                    tgt_key_padding_mask=src_key_padding_mask[valid],
                    memory_key_padding_mask=src_key_padding_mask[valid],
                )
        else:
            memory = self.transformer(
                henc, henc,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

        # Masked average pooling
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        pooled = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-9)

        return self.linear_class(pooled)

