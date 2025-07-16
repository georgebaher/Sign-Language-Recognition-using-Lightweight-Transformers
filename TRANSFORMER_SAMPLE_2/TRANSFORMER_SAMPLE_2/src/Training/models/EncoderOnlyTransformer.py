import torch
import torch.nn as nn
import math


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


class SPOTEREncoderOnly(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, n_heads=8, num_layers=6, dropout=0.1, max_len=500, w_pe=True):
        super().__init__()
        print(f"[INFO] Initializing SPOTEREncoderOnly with {n_heads} heads, hidden_dim={hidden_dim}")
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

    def forward(self, x):
        B, T, D = x.shape

        # Padding mask: True where padding
        pad_mask = (x == -2).all(dim=-1)  # [B, T]


        # Replace missing values with 0
        x = x.clone()
        x[x == -2] = 0.0

        # Positional encoding
        if self.w_pe:
            # Sin-Cos Positional encoding
            x = self.sin_cos_pos_embedding(x)
            # # Learnable Positional encoding
            # x = x + self.learnable_pos_embedding[:, :T]

        # Encode sequence
        memory = self.encoder(x, src_key_padding_mask=pad_mask)     # [B, T, D]
        # Global average pooling over time
        pooled = torch.mean(memory, dim=1)  # [B, 1, D] and automatically the 1 is squeezed out, so it becomes [B, D]

        # Classify
        return self.classifier(pooled)
