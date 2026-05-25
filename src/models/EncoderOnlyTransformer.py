import torch
import torch.nn as nn
import math


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderOnly(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, n_heads=8, num_layers=6, dropout=0.0, max_len=500, w_pe=False):
        super().__init__()
        print(f"[INFO] Initializing EncoderOnly with {n_heads} heads, hidden_dim={hidden_dim}")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.w_pe = w_pe
        self.sin_cos_pos_embedding = PositionalEncodingSinCos(hidden_dim, dropout, max_len, )
        self.learnable_pos_embedding = nn.Parameter(
            torch.randn(1, max_len, hidden_dim))  # learnable positional embedding

        if self.w_pe:
            self.pos_encoder = PositionalEncodingSinCos(hidden_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] with padded timesteps filled by -2 in all D features.
        Returns logits: [B, num_classes]
        """
        B, T, D = x.shape

        # pad_mask: [B, T], True where padded
        pad_mask = (x == -2).all(dim=-1)  # True at timestep if it's all padding
        is_fully_padded = pad_mask.all(dim=1)  # [B], True if *entire* sequence is padding

        # Replace padding sentinel with zeros before encoding
        x = x.clone()
        x[x == -2] = 0.0

        if self.w_pe:
            # Sin-Cos Positional encoding
            x = self.sin_cos_pos_embedding(x)

            # # Learnable Positional encoding
            # x = x + self.learnable_pos_embedding[:, :T]

        # ---- Safe encoding: skip encoder for fully-padded sequences to avoid NaN softmax ----
        if is_fully_padded.any():
            memory = x.new_zeros(B, T, D)  # fill fully-padded sequences with zeros
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = self.encoder(
                    x[valid],
                    src_key_padding_mask=pad_mask[valid]
                )
        else:
            memory = self.encoder(x, src_key_padding_mask=pad_mask)

        # Masked average pooling
        valid_mask = (~pad_mask).unsqueeze(-1).float()  # [B, T, 1]
        pooled = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-9)

        return self.classifier(pooled)

