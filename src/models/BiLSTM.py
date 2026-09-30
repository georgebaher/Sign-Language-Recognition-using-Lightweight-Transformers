import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    A Bidirectional LSTM (Bi-LSTM) classifier for sequence classification.

    This model is designed to handle variable-length sequences within a batch
    by correctly identifying the last valid timestep for the forward pass and
    the first timestep for the backward pass.

    Args:
        input_dim (int): The number of features in the input.
        hidden_dim (int): The number of features in the hidden state for each direction.
        num_classes (int): The number of output classes.
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the
                                   outputs of each LSTM layer except the last layer.
                                   Defaults to 0.0.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.0):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # --- Key Changes for Bidirectionality ---
        # 1. Set `bidirectional=True`. This creates an LSTM that processes the
        #    sequence from left-to-right and right-to-left.
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # The key change
            dropout=dropout if num_layers > 1 else 0  # Dropout is only applied between LSTM layers
        )

        # 2. The output of the Bi-LSTM will be the concatenation of the final forward
        #    and backward hidden states. Therefore, the classifier's input
        #    dimension must be `hidden_dim * 2`.
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        if lengths is None:
            lengths = (~(x == -2).all(dim=-1)).sum(dim=1)
        if (lengths <= 0).any():
            raise ValueError("BiLSTM requires positive sequence lengths")
        x = x.masked_fill(x == -2, 0.)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))
