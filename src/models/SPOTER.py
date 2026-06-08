import torch
import torch.nn as nn

from src.models.positional_encoding import build_positional_encoding


class SPOTERTransformer(nn.Module):
    """SPOTER (Sign POse-based TransformER): TransformerEncoder + TransformerDecoder driven by a learned class query, with selectable positional encoding and a NaN-safe path for fully-padded sequences."""

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
        self.class_query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, D]
        self.pos_encoding = build_positional_encoding(pe, hidden_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        # See EncoderOnlyTransformer for why enable_nested_tensor=False.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
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

        # Safe encoding: skip encoder for fully-padded rows (avoid NaN softmax over all-(-inf))
        if is_fully_padded.any():
            memory = x.new_zeros(B, T, self.hidden_dim)
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = self.encoder(
                    x[valid],
                    src_key_padding_mask=pad_mask[valid],
                )
        else:
            memory = self.encoder(x, src_key_padding_mask=pad_mask)

        # Decode the learned class query against the (sanitized) memory
        query = self.class_query.expand(B, -1, -1)  # [B, 1, D]
        if is_fully_padded.any():
            output = query.new_zeros(B, 1, query.size(-1))
            valid = ~is_fully_padded
            if valid.any():
                output[valid] = self.decoder(
                    query[valid],
                    memory[valid],
                    memory_key_padding_mask=pad_mask[valid],
                )
        else:
            output = self.decoder(
                query,
                memory,
                memory_key_padding_mask=pad_mask,
            )

        return self.classifier(output.squeeze(1))
