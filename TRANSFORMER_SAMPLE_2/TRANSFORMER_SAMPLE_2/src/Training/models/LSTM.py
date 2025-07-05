import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, num_classes, num_layers=1, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        print(f"[INFO] LSTM model initialized with {num_layers} layers and {hidden_dim} hidden units")

    def pad_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the feature dimension to be divisible by number of heads."""
        B, T, D = x.shape
        if D % self.n_heads != 0:
            target_dim = ((D + self.n_heads - 1) // self.n_heads) * self.n_heads
            pad_width = target_dim - D
            pad_tensor = torch.full((B, T, pad_width), fill_value=-2.0, device=x.device)
            x = torch.cat([x, pad_tensor], dim=-1)
        return x

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()

        # Pad feature dim if needed (done here for LSTM, just for consistency with other models)
        x = self.pad_to_heads(x)

        pad_mask = (x == -2).all(dim=-1)
        x = x.clone()
        x[x == -2] = 0.0

        # Forward LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, H]
        lengths = (~pad_mask).sum(dim=1)  # [B]
        final_out = lstm_out[torch.arange(B), lengths - 1]  # [B, H]
        logits = self.classifier(final_out)  # [B, num_classes]
        assert logits.shape == (B, self.classifier.out_features)
        return logits

    # def forward(self, x):
    #     # x: [B, T, D]
    #     pad_mask = (x == -2).all(dim=-1)
    #     x = x.clone()
    #     x[x == -2] = 0.0
    #
    #     # Forward LSTM
    #     lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, H]
    #     final_out = lstm_out[:, -1, :]  # take last time step
    #     logits = self.classifier(final_out)  # [B, num_classes]
    #     return logits