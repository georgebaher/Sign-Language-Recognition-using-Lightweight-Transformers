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

        print(f"[INFO] BiLSTM initialized | layers={num_layers}, hidden_dim={hidden_dim} (per direction)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bi-LSTM classifier.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SequenceLength, FeatureDim].
                              Assumes padding is represented by rows where all features are -2.0.

        Returns:
            torch.Tensor: Logits tensor of shape [Batch, NumClasses].
        """
        # Get batch size
        B = x.size(0)

        # --- 1. Handle Padding ---
        # Create a mask to identify padded timesteps
        pad_mask = (x == -2).all(dim=-1) # [B, T]
        # Zero out the padded values, so they don't affect the LSTM's calculations
        x = x.clone()
        x[x == -2] = 0.0

        # --- 2. Forward Pass through LSTM ---
        # The output `lstm_out` will have shape [B, T, hidden_dim * 2]
        # The first `hidden_dim` channels are the forward pass outputs.
        # The next `hidden_dim` channels are the backward pass outputs.
        lstm_out, _ = self.lstm(x)

        # --- 3. Extract the Correct Final Hidden States ---
        # Calculate the actual length of each sequence in the batch
        lengths = (~pad_mask).sum(dim=1)

        # For the FORWARD pass, we need the hidden state at the LAST valid timestep.
        # This represents the context from the beginning to the end of the sequence.
        # We select from the first half of the feature dimension.
        forward_out = lstm_out[torch.arange(B), lengths-1, :self.hidden_dim]

        # For the BACKWARD pass, the most comprehensive hidden state (representing
        # context from the end to the beginning) is at the FIRST timestep (index 0).
        # We select from the second half of the feature dimension.
        backward_out = lstm_out[:, 0, self.hidden_dim:]

        # --- 4. Concatenate and Classify ---
        # Concatenate the final forward and backward states to get the full representation
        final_out = torch.cat((forward_out, backward_out), dim=1)

        # Pass the combined representation to the classifier
        logits = self.classifier(final_out)

        # --- 5. Final Sanity Check ---
        assert logits.shape == (B, self.classifier.out_features)

        return logits