import torch
import math
import torch.nn as nn

class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SPOTERTransformer(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, n_heads: int = 8, num_layers: int = 6, dropout: float = 0.0, max_len: int = 500, w_pe: bool = True):
        super(SPOTERTransformer, self).__init__()
        print(f"[INFO] Initializing SPOTER model with {n_heads} heads and {hidden_dim} hidden_dim.")
        self.w_pe = w_pe
        self.d_model = hidden_dim
        self.n_heads = n_heads
        self.class_query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, D]
        self.sin_cos_pos_embedding = PositionalEncodingSinCos(hidden_dim, dropout, max_len,)
        self.learnable_pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim))      # learnable positional embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        print(f"[INFO] SPOTER model initialized with {'no ' if not w_pe else ''}positional encoding")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape

        # Pad mask: True where padding exists
        pad_mask = (x == -2).all(dim=-1)  # [B, T]
        is_fully_padded = pad_mask.all(dim=1)  # [B], True for sequences that are all padding

        # Replace missing values with 0
        x = x.clone()
        x[x == -2] = 0.0
        # x = x - 0.5

        if self.w_pe:
            # Sin-Cos Positional encoding
            x = self.sin_cos_pos_embedding(x)

            # # Learnable Positional encoding
            # x = x + self.learnable_pos_embedding[:, :T]

        # Encode sequence
        memory = self.encoder(x, src_key_padding_mask=pad_mask)

        # # --- DEBUG: Check for NaNs in encoder output ---
        # nan_mask = torch.isnan(memory)  # [B, T, D]
        # if nan_mask.any():
        #     print("[DEBUG] NaNs detected in encoder output (memory)!")


        # # Repeat learnable class token across batch
        # query = self.class_query.expand(B, -1, -1)  # [B, 1, D]

        # # Decode using class token
        # output = self.decoder(query, memory, memory_key_padding_mask=pad_mask)  # [B, 1, D]
        #
        # # Classify
        # logits = self.classifier(output.squeeze(1))  # [B, num_classes]
        # return logits

        # *** FIX NaN issue ***
        # Sanitize the encoder's output. Replace NaN rows with zeros.
        # This is the key step to prevent NaN propagation to the decoder.
        memory[is_fully_padded] = 0

        # Proceed with the decoder using the sanitized memory
        query = self.class_query.expand(B, -1, -1)
        output = self.decoder(query, memory, memory_key_padding_mask=pad_mask)

        # Classify
        logits = self.classifier(output.squeeze(1))
        return logits
