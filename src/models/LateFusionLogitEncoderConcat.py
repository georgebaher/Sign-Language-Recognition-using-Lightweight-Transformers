import torch
import torch.nn as nn


class LateFusionEncoder(nn.Module):
    """Late fusion of three modality-specific Transformer encoders.

    Each modality (hand / pose / face) gets its own ``nn.Linear`` embedding into a
    shared ``hidden_dim``, its own TransformerEncoder + classifier, and produces a
    logit vector. The three logit vectors are concatenated and passed through a
    final trainable Linear that mixes them.
    """

    def __init__(self, hand_input_dim: int, pose_input_dim: int, face_input_dim: int,
                 num_classes: int, hidden_dim: int = 256, n_heads: int = 8, num_layers: int = 6,
                 dropout: float = 0.0, max_len: int = 500, debug: bool = False):
        super().__init__()
        self.hand_dim = hand_input_dim
        self.pose_dim = pose_input_dim
        self.face_dim = face_input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.debug = debug

        self.hand_embedding = nn.Linear(hand_input_dim, hidden_dim)
        self.pose_embedding = nn.Linear(pose_input_dim, hidden_dim)
        self.face_embedding = nn.Linear(face_input_dim, hidden_dim)

        def make_encoder():
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
            return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        self.hand_encoder = make_encoder()
        self.pose_encoder = make_encoder()
        self.face_encoder = make_encoder()

        self.hand_classifier = nn.Linear(hidden_dim, num_classes)
        self.pose_classifier = nn.Linear(hidden_dim, num_classes)
        self.face_classifier = nn.Linear(hidden_dim, num_classes)

        self.fusion_classifier = nn.Linear(num_classes * 3, num_classes)

        if self.debug:
            print("\n" + "=" * 20 + " Initializing LateFusionEncoder (Logit Concat) " + "=" * 20)
            print(f"  - Hand Stream: Input {self.hand_dim} -> hidden_dim {hidden_dim}")
            print(f"  - Pose Stream: Input {self.pose_dim} -> hidden_dim {hidden_dim}")
            print(f"  - Face Stream: Input {self.face_dim} -> hidden_dim {hidden_dim}")
            print(f"  - Fusion: Linear({num_classes * 3}, {num_classes})")
            print("=" * 70 + "\n")

    def _process_stream(self, x: torch.Tensor, embedding: nn.Linear, encoder: nn.TransformerEncoder,
                        classifier: nn.Linear) -> torch.Tensor:
        # x: [B, T, modality_input_dim]
        B, T, _ = x.shape

        pad_mask = (x == -2).all(dim=-1)  # [B, T]
        is_fully_padded = pad_mask.all(dim=1)  # [B]

        x = x.clone()
        x[x == -2] = 0.0

        x = embedding(x)  # [B, T, hidden_dim]

        if is_fully_padded.any():
            memory = x.new_zeros(B, T, self.hidden_dim)
            valid = ~is_fully_padded
            if valid.any():
                memory[valid] = encoder(x[valid], src_key_padding_mask=pad_mask[valid])
        else:
            memory = encoder(x, src_key_padding_mask=pad_mask)

        valid_mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-9)
        return classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"\n[DEBUG] LateFusionEncoder (Concat) input shape: {x.shape}")

        start_pose = self.hand_dim
        start_face = self.hand_dim + self.pose_dim
        end_face = start_face + self.face_dim

        x_hands = x[:, :, :start_pose]
        x_pose = x[:, :, start_pose:start_face]
        x_face = x[:, :, start_face:end_face]

        hand_logits = self._process_stream(x_hands, self.hand_embedding, self.hand_encoder, self.hand_classifier)
        pose_logits = self._process_stream(x_pose, self.pose_embedding, self.pose_encoder, self.pose_classifier)
        face_logits = self._process_stream(x_face, self.face_embedding, self.face_encoder, self.face_classifier)

        fused = torch.cat([hand_logits, pose_logits, face_logits], dim=1)  # [B, num_classes * 3]
        return self.fusion_classifier(fused)
