import torch
import math
import torch.nn as nn



class PositionalEncodingSinCos(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, w_pe=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.w_pe = w_pe

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        posEnc = self.pe[:x.size(0)]
        if not self.w_pe:
            return x
        xAndPosEnc = x + posEnc
        return self.dropout(xAndPosEnc)


class BaselineTransformerClassification(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55, n_heads=9, max_seq_len=50, w_pe=True):
        super().__init__()

        self.output_pos_encoding = PositionalEncodingSinCos(d_model=hidden_dim, dropout=0.0, w_pe=w_pe)

        self.transformer = nn.Transformer(hidden_dim, n_heads, 6, 6, batch_first=True)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        h = inputs.float()  # [B, T, D]

        # Create a mask where all features in a timestep are -2 → it's a padding frame
        src_key_padding_mask = (inputs == -2).all(dim=-1)  # shape: [batch_size, seq_len]

        # Replace all -2 values (missing features) with 0
        h[h == -2] = 0.0

        # Subtract 0.5 from all features
        h = h - 0.5

        # # 🔍 Check input variation per sample
        # print("Input mean:", h.mean(dim=[1, 2]))
        # print("Input std:", h.std(dim=[1, 2]))

        # Apply positional encoding
        henc = self.output_pos_encoding(h)

        h = self.transformer(henc, henc, src_key_padding_mask=src_key_padding_mask)


        # Temporal average pooling
        pooled = torch.mean(h, dim=1)
        # print(f"Pooled representations {pooled.shape}:", pooled)

        res = self.linear_class(pooled)
        return res
