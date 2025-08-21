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

    def forward(self, x):
        B, T, D = x.shape

        # Padding mask: True where padding
        pad_mask = (x == -2).all(dim=-1)  # [B, T]
        is_fully_padded = pad_mask.all(dim=1)  # [B], True for sequences that are all padding

        # Replace missing values with 0
        x = x.clone()
        x[x == -2] = 0.0

        # Positional encoding
        if self.w_pe:
            # Sin-Cos Positional encoding
            x = self.sin_cos_pos_embedding(x)
            # # Learnable Positional encoding
            # x = x + self.learnable_pos_embedding[:, :T]

        # *** FIX NaN issue ***
        #  Encode sequence.
        # Note: 'memory' will contain NaN rows where 'is_fully_padded' is True.
        memory = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, T, D]

        # Fix NaNs and perform correct masked average pooling.

        # First, explicitly replace the NaN rows with zeros. This fixes the primary issue.
        # If a sequence contained no information, its feature representation should be zero.
        memory[is_fully_padded] = 0.0

        # Second, correctly average only over the non-padded time steps.
        # This is more accurate than a simple torch.mean() and improves performance.

        # Create a mask for broadcasting: [B, T] -> [B, T, 1]
        # `~pad_mask` is True for valid tokens.
        output_mask = (~pad_mask).unsqueeze(-1).float()

        # Zero out the memory values at padded positions for all sequences
        masked_memory = memory * output_mask

        # Sum the valid time steps
        summed_memory = torch.sum(masked_memory, dim=1)  # Shape: [B, D]

        # Count the number of valid time steps for each sequence.
        # Use clamp to prevent division by zero for fully padded sequences.
        valid_step_count = output_mask.sum(dim=1)  # Shape: [B, 1]
        valid_step_count = valid_step_count.clamp(min=1e-9)

        # Calculate the masked average
        pooled = summed_memory / valid_step_count  # Shape: [B, D]

        # Classify the pooled representation
        return self.classifier(pooled)
