import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionEncoder(nn.Module):
    """
    A Late Fusion model using parallel Transformer Encoders for each modality.

    This version removes the initial embedding layers. Instead, it pads each
    modality's feature dimension to be divisible by n_heads and processes them directly.
    """

    def __init__(self, hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, n_heads: int = 8, num_layers: int = 6, hidden_dim: int = 128,
                 dropout: float = 0.0, max_len: int = 500, debug: bool = False):
        super().__init__()

        self.hand_dim = hand_input_dim
        self.pose_dim = pose_input_dim
        self.face_dim = face_input_dim
        self.n_heads = n_heads
        self.debug = debug

        # --- 1. Calculate Padded Dimensions ---
        # Each encoder's d_model must be divisible by n_heads.
        self.hand_padded_dim = ((self.hand_dim + n_heads - 1) // n_heads) * n_heads
        self.pose_padded_dim = ((self.pose_dim + n_heads - 1) // n_heads) * n_heads
        self.face_padded_dim = ((self.face_dim + n_heads - 1) // n_heads) * n_heads

        # --- DEBUG: Print initialization summary ---
        if self.debug:
            print("\n" + "=" * 20 + " Initializing LateFusionEncoder_Padded " + "=" * 20)
            print(f"  - Hand Input Dim: {self.hand_dim} -> Padded to: {self.hand_padded_dim}")
            print(f"  - Pose Input Dim: {self.pose_dim} -> Padded to: {self.pose_padded_dim}")
            print(f"  - Face Input Dim: {self.face_dim} -> Padded to: {self.face_padded_dim}")
            print("=" * 70 + "\n")
        # --- END DEBUG ---

        # --- 2. Modality-Specific Encoders and Classifiers ---
        # NOTE: The nn.Linear embedding layers are GONE.
        # Each encoder is now initialized with its own specific padded dimension.
        hand_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hand_padded_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.hand_encoder = nn.TransformerEncoder(hand_encoder_layer, num_layers=num_layers)
        self.hand_classifier = nn.Linear(self.hand_padded_dim, num_classes)

        pose_encoder_layer = nn.TransformerEncoderLayer(d_model=self.pose_padded_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=num_layers)
        self.pose_classifier = nn.Linear(self.pose_padded_dim, num_classes)

        face_encoder_layer = nn.TransformerEncoderLayer(d_model=self.face_padded_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.face_encoder = nn.TransformerEncoder(face_encoder_layer, num_layers=num_layers)
        self.face_classifier = nn.Linear(self.face_padded_dim, num_classes)

        self.fusion_weights = nn.Parameter(torch.ones(3))

    def _process_stream(self, x: torch.Tensor, padded_dim: int, encoder: nn.TransformerEncoder,
                        classifier: nn.Linear) -> torch.Tensor:
        """Helper function to process one modality stream from input to logits."""
        pad_mask = (x == -2).all(dim=-1)
        is_fully_padded = pad_mask.all(dim=1)
        x = x.clone();
        x[x == -2] = 0.0

        # --- NEW: Pad the feature dimension ---
        # Calculate how much padding is needed.
        num_to_pad = padded_dim - x.shape[-1]
        if num_to_pad > 0:
            # F.pad format is (pad_left, pad_right, pad_top, pad_bottom, etc.)
            # We only want to pad the last dimension on the right.
            x = F.pad(x, (0, num_to_pad), "constant", 0)

        memory = encoder(x, src_key_padding_mask=pad_mask)
        memory[is_fully_padded] = 0.0

        output_mask = (~pad_mask).unsqueeze(-1).float()
        pooled = torch.sum(memory * output_mask, dim=1) / output_mask.sum(dim=1).clamp(min=1e-9)

        return classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the late fusion model.
        """
        # --- 1. Internal Slicing Stage ---
        start_pose = self.hand_dim
        start_face = self.hand_dim + self.pose_dim
        end_face = start_face + self.face_dim

        x_hands = x[:, :, :start_pose]
        x_pose = x[:, :, start_pose:start_face]
        x_face = x[:, :, start_face:end_face]

        # --- 2. Get logits from each parallel "expert" stream ---
        hand_logits = self._process_stream(x_hands, self.hand_padded_dim, self.hand_encoder, self.hand_classifier)
        pose_logits = self._process_stream(x_pose, self.pose_padded_dim, self.pose_encoder, self.pose_classifier)
        face_logits = self._process_stream(x_face, self.face_padded_dim, self.face_encoder, self.face_classifier)

        # --- 3. Fusion Stage ---
        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        stacked_logits = torch.stack([hand_logits, pose_logits, face_logits])
        weighted_logits = stacked_logits * normalized_weights.view(3, 1, 1)
        fused_logits = torch.sum(weighted_logits, dim=0)

        return fused_logits