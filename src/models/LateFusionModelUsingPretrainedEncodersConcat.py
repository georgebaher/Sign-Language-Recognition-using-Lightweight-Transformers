import torch
import torch.nn as nn


class LateFusionPET(nn.Module):
    """Late fusion of three pre-trained, frozen expert models.

    Each expert already owns its own input-embedding layer, so this wrapper only
    needs to slice the early-fused input tensor into per-modality streams and
    feed each stream into its expert. Fusion concatenates the three logit
    vectors and passes them through a trainable Linear that mixes them.
    """

    def __init__(self,
                 hand_expert: nn.Module, pose_expert: nn.Module, face_expert: nn.Module,
                 hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, debug: bool = False):
        super().__init__()

        self.hand_expert = hand_expert
        self.pose_expert = pose_expert
        self.face_expert = face_expert

        self.hand_input_dim = hand_input_dim
        self.pose_input_dim = pose_input_dim
        self.face_input_dim = face_input_dim

        self.debug = debug

        for expert in [self.hand_expert, self.pose_expert, self.face_expert]:
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()

        self.fusion_classifier = nn.Linear(num_classes * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start_pose = self.hand_input_dim
        start_face = self.hand_input_dim + self.pose_input_dim
        end_face = start_face + self.face_input_dim

        x_hands = x[:, :, :start_pose]
        x_pose = x[:, :, start_pose:start_face]
        x_face = x[:, :, start_face:end_face]

        with torch.no_grad():
            hand_logits = self.hand_expert(x_hands)
            pose_logits = self.pose_expert(x_pose)
            face_logits = self.face_expert(x_face)

        fused = torch.cat([hand_logits, pose_logits, face_logits], dim=1)
        return self.fusion_classifier(fused)
