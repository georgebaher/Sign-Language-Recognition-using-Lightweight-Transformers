import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionPET(nn.Module):
    """
    A late fusion model that combines the outputs of three pre-trained,
    frozen "expert" models via a trainable linear layer.

    This model is fully self-contained:
    1. It accepts a single, early-fused input tensor.
    2. It internally splits the tensor into modality-specific streams.
    3. It pads each stream's feature dimension to match the expert's expected hidden dimension.
    """

    def __init__(self,
                 hand_expert: nn.Module, pose_expert: nn.Module, face_expert: nn.Module,
                 hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, debug: bool = False):
        super().__init__()
        print("[INFO] Initializing LateFusionPET with internal slicing and padding...")

        # --- 1. Store Experts and Dimensions ---
        self.hand_expert = hand_expert
        self.pose_expert = pose_expert
        self.face_expert = face_expert

        # Store the RAW input dimensions for slicing
        self.hand_input_dim = hand_input_dim
        self.pose_input_dim = pose_input_dim
        self.face_input_dim = face_input_dim

        # Infer the TARGET padded dimensions from the experts themselves
        self.hand_target_dim = self.hand_expert.hidden_dim
        self.pose_target_dim = self.pose_expert.hidden_dim
        self.face_target_dim = self.face_expert.hidden_dim

        self.debug = debug

        # --- DEBUG: Print initialization summary ---
        if self.debug:
            print("\n" + "=" * 20 + " Initializing LateFusionPET " + "=" * 20)
            print(f"  - Hand Stream: Input {self.hand_input_dim} -> Expert Target {self.hand_target_dim}")
            print(f"  - Pose Stream: Input {self.pose_input_dim} -> Expert Target {self.pose_target_dim}")
            print(f"  - Face Stream: Input {self.face_input_dim} -> Expert Target {self.face_target_dim}")
            print("=" * 70 + "\n")
        # --- END DEBUG ---

        # --- 2. FREEZE the expert models ---
        print("[INFO] Freezing parameters of the expert models.")
        for expert in [self.hand_expert, self.pose_expert, self.face_expert]:
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()

        # --- 3. Create the trainable fusion layer ---
        self.fusion_classifier = nn.Linear(num_classes * 3, num_classes)
        print(f"[INFO] Trainable fusion layer created: Linear({num_classes * 3}, {num_classes})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the logit fusion model.
        Args:
            x (torch.Tensor): A single, early-fused input tensor.
        """
        # --- DEBUG ---
        if self.debug:
            print("\n--- LateFusionPET Forward Pass ---")
            print(f"[DEBUG] Initial input shape: {x.shape}")
        # --- END DEBUG ---

        # --- 1. Internal Slicing Stage ---
        start_pose = self.hand_input_dim
        start_face = self.hand_input_dim + self.pose_input_dim
        end_face = start_face + self.face_input_dim

        x_hands = x[:, :, :start_pose]
        x_pose = x[:, :, start_pose:start_face]
        x_face = x[:, :, start_face:end_face]

        # --- 2. Get logits from each frozen "expert" stream ---
        with torch.no_grad():
            # --- 2a. Internal Padding Logic ---
            num_pad_hand = self.hand_target_dim - x_hands.shape[-1]
            if num_pad_hand > 0:
                x_hands = F.pad(x_hands, (0, num_pad_hand), "constant", 0)

            num_pad_pose = self.pose_target_dim - x_pose.shape[-1]
            if num_pad_pose > 0:
                x_pose = F.pad(x_pose, (0, num_pad_pose), "constant", 0)

            num_pad_face = self.face_target_dim - x_face.shape[-1]
            if num_pad_face > 0:
                x_face = F.pad(x_face, (0, num_pad_face), "constant", 0)

            # --- DEBUG ---
            if self.debug:
                print(
                    f"[DEBUG] Shapes after padding -> Hands: {x_hands.shape}, Pose: {x_pose.shape}, Face: {x_face.shape}")
            # --- END DEBUG ---

            # --- 2b. Inference with Experts ---
            hand_logits = self.hand_expert(x_hands)
            pose_logits = self.pose_expert(x_pose)
            face_logits = self.face_expert(x_face)

        # --- 3. Fusion Stage ---
        fused_logits = torch.cat((hand_logits, pose_logits, face_logits), dim=1)

        # --- 4. Final Classification ---
        final_logits = self.fusion_classifier(fused_logits)

        # --- DEBUG ---
        if self.debug:
            print(f"[DEBUG] Final logit shape: {final_logits.shape}")
            print("--- End Forward Pass ---\n")
            self.debug = False  # Print only for the first batch
        # --- END DEBUG ---

        return final_logits
