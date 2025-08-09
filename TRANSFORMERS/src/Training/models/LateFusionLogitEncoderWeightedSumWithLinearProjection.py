import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class LateFusionEncoder(nn.Module):
    """
    A Late Fusion model using parallel Transformer Encoders for each modality.

    This model accepts a single, early-fused input tensor and internally splits
    it into three streams (hands, pose, face) for modality-specific processing.
    Fusion occurs at the logit level via a learned weighted sum.
    """

    def __init__(self, hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, hidden_dim: int = 256, n_heads: int = 8,
                 num_layers: int = 6, dropout: float = 0.0, max_len: int = 500,
                 debug: bool = False):
        super().__init__()

        # --- Store dimensions and debug flag ---
        self.hand_dim = hand_input_dim
        self.pose_dim = pose_input_dim
        self.face_dim = face_input_dim
        self.debug = debug

        # --- DEBUG: Print initialization summary ---
        if self.debug:
            print("\n" + "=" * 20 + " Initializing LateFusionEncoder " + "=" * 20)
            print(f"  - Hand Input Dim: {self.hand_dim}")
            print(f"  - Pose Input Dim: {self.pose_dim}")
            print(f"  - Face Input Dim: {self.face_dim}")
            print(f"  - Hidden Dim (all streams): {hidden_dim}")
            print("=" * 70 + "\n")
        # --- END DEBUG ---

        # --- Modality-Specific Streams ---
        self.hand_embed = nn.Linear(self.hand_dim, hidden_dim)
        hand_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.hand_encoder = nn.TransformerEncoder(hand_encoder_layer, num_layers=num_layers)
        self.hand_classifier = nn.Linear(hidden_dim, num_classes)

        self.pose_embed = nn.Linear(self.pose_dim, hidden_dim)
        pose_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=num_layers)
        self.pose_classifier = nn.Linear(hidden_dim, num_classes)

        self.face_embed = nn.Linear(self.face_dim, hidden_dim)
        face_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout,
                                                        batch_first=True)
        self.face_encoder = nn.TransformerEncoder(face_encoder_layer, num_layers=num_layers)
        self.face_classifier = nn.Linear(hidden_dim, num_classes)


        self.fusion_weights = nn.Parameter(torch.ones(3))

    def _process_stream(self, x: torch.Tensor, embed_layer: nn.Linear, encoder: nn.TransformerEncoder,
                        classifier: nn.Linear) -> torch.Tensor:
        """Helper function to process one modality stream from input to logits."""
        pad_mask = (x == -2).all(dim=-1)
        is_fully_padded = pad_mask.all(dim=1)
        x = x.clone()
        x[x == -2] = 0.0

        x_embedded = embed_layer(x)

        memory = encoder(x_embedded, src_key_padding_mask=pad_mask)
        memory[is_fully_padded] = 0.0

        output_mask = (~pad_mask).unsqueeze(-1).float()
        pooled = torch.sum(memory * output_mask, dim=1) / output_mask.sum(dim=1).clamp(min=1e-9)

        return classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the late fusion model.
        """
        # --- DEBUG: Print input shape ---
        if self.debug:
            print("\n--- LateFusionEncoder Forward Pass ---")
            print(f"[DEBUG] Input shape: {x.shape}")
        # --- END DEBUG ---

        # --- 1. Internal Slicing Stage ---
        start_pose = self.hand_dim
        start_face = self.hand_dim + self.pose_dim
        end_face = start_face + self.face_dim

        x_hands = x[:, :, :start_pose]
        x_pose = x[:, :, start_pose:start_face]
        x_face = x[:, :, start_face:end_face]

        # --- DEBUG: Verify sliced shapes ---
        if self.debug:
            print(f"[DEBUG] Sliced shapes -> Hands: {x_hands.shape}, Pose: {x_pose.shape}, Face: {x_face.shape}")
        # --- END DEBUG ---

        # --- 2. Get logits from each parallel "expert" stream ---
        hand_logits = self._process_stream(x_hands, self.hand_embed, self.hand_encoder, self.hand_classifier)
        pose_logits = self._process_stream(x_pose, self.pose_embed, self.pose_encoder, self.pose_classifier)
        face_logits = self._process_stream(x_face, self.face_embed, self.face_encoder, self.face_classifier)

        # --- 3. Fusion Stage ---
        normalized_weights = F.softmax(self.fusion_weights, dim=0)

        # --- DEBUG: Print learned fusion weights ---
        if self.debug:
            weights_np = normalized_weights.detach().cpu().numpy()
            print(
                f"[DEBUG] Normalized Fusion Weights -> Hand: {weights_np[0]:.4f}, Pose: {weights_np[1]:.4f}, Face: {weights_np[2]:.4f}")
        # --- END DEBUG ---

        stacked_logits = torch.stack([hand_logits, pose_logits, face_logits])
        weighted_logits = stacked_logits * normalized_weights.view(3, 1, 1)
        fused_logits = torch.sum(weighted_logits, dim=0)

        # --- DEBUG: Print final output shape ---
        if self.debug:
            print(f"[DEBUG] Final fused logit shape: {fused_logits.shape}")
            print("--- End Forward Pass ---\n")
            # self.debug = False
        # --- END DEBUG ---

        return fused_logits