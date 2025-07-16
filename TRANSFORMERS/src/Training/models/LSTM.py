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

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()


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