import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class LateFusionEncoder(nn.Module):
    """
    A Late Fusion model using parallel Transformer Encoders for each modality.

    This version removes the initial embedding layers. Instead, it pads each
    modality's feature dimension to be divisible by n_heads and processes them directly.
    """

    def __init__(self, hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, n_heads: int = 8, num_layers: int = 6, hidden_dim: int = 256,
                 dropout: float = 0.0, max_len: int = 500, debug: bool = False):
        super().__init__()

        self.hand_dim = hand_input_dim
        self.pose_dim = pose_input_dim
        self.face_dim = face_input_dim
        self.n_heads = n_heads
        self.debug = debug

        # --- 1. Calculate Padded Dimensions ---
        self.hand_padded_dim = ((self.hand_dim + n_heads - 1) // n_heads) * n_heads
        self.pose_padded_dim = ((self.pose_dim + n_heads - 1) // n_heads) * n_heads
        self.face_padded_dim = ((self.face_dim + n_heads - 1) // n_heads) * n_heads

        # --- DEBUG: Print initialization summary ---
        if self.debug:
            print("\n" + "=" * 20 + " Initializing LateFusionEncoder (Padded) " + "=" * 20)
            print(f"  - Hand Stream: Input {self.hand_dim} -> Encoder d_model {self.hand_padded_dim}")
            print(f"  - Pose Stream: Input {self.pose_dim} -> Encoder d_model {self.pose_padded_dim}")
            print(f"  - Face Stream: Input {self.face_dim} -> Encoder d_model {self.face_padded_dim}")
            print("=" * 70 + "\n")
        # --- END DEBUG ---

        # --- 2. Modality-Specific Encoders and Classifiers ---
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

    def _process_stream(self, x: torch.Tensor, stream_name: str, padded_dim: int, encoder: nn.TransformerEncoder,
                        classifier: nn.Linear) -> torch.Tensor:
        """Helper function to pad, process, and classify one modality stream."""
        B, T, D = x.shape
        pad_mask = (x == -2).all(dim=-1)
        is_fully_padded = pad_mask.all(dim=1)
        x = x.clone();
        x[x == -2] = 0.0

        # --- Pad the feature dimension ---
        num_to_pad = padded_dim - x.shape[-1]
        if num_to_pad > 0:
            x = F.pad(x, (0, num_to_pad), "constant", 0)

        # --- DEBUG: Print padded shape ---
        if self.debug:
            print(f"[DEBUG] {stream_name} stream shape after padding: {x.shape}")
        # --- END DEBUG ---

        # ---- Safe encoding: skip encoder for fully-padded sequences to avoid NaN softmax ----
        if is_fully_padded.any():
            memory = x.new_zeros(B, T, D+num_to_pad)  # fill fully-padded sequences with zeros
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = encoder(
                    x[valid],
                    src_key_padding_mask=pad_mask[valid]
                )
        else:
            memory = encoder(x, src_key_padding_mask=pad_mask)

        output_mask = (~pad_mask).unsqueeze(-1).float()
        pooled = torch.sum(memory * output_mask, dim=1) / output_mask.sum(dim=1).clamp(min=1e-9)

        return classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the late fusion model.
        """
        # --- DEBUG: Print input shape ---
        if self.debug:
            print("\n--- LateFusionEncoder (Padded) Forward Pass ---")
            print(f"[DEBUG] Initial input shape: {x.shape}")
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
        hand_logits = self._process_stream(x_hands, "Hand", self.hand_padded_dim, self.hand_encoder,
                                           self.hand_classifier)
        pose_logits = self._process_stream(x_pose, "Pose", self.pose_padded_dim, self.pose_encoder,
                                           self.pose_classifier)
        face_logits = self._process_stream(x_face, "Face", self.face_padded_dim, self.face_encoder,
                                           self.face_classifier)

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
            # Set debug to False after the first pass to avoid cluttering the console
            # self.debug = False
        # --- END DEBUG ---

        return fused_logits
