"""Jointly trained modality encoders with learned weighted-logit fusion."""
import torch
from torch import nn
from src.models.EncoderOnlyTransformer import EncoderOnly


class FGLateFusion(nn.Module):
    def __init__(self, dimensions, num_classes, **kwargs):
        super().__init__()
        if len(dimensions) < 2 or any(d <= 0 for d in dimensions):
            raise ValueError('Late fusion needs at least two nonempty modalities')
        self.dimensions = dimensions
        # One independent encoder per modality; dimensions gives each input width.
        self.encoders = nn.ModuleList([EncoderOnly(d, num_classes, **kwargs) for d in dimensions])
        # Learn M fusion parameters; zeros give equal weights after softmax.
        self.fusion_weights = nn.Parameter(torch.zeros(len(dimensions)))

    def forward(self, x, lengths=None):
        # B = batch size, T = padded frame count, M = modalities, C = classes.
        # x: [B, T, sum(dimensions)]; lengths: [B] when provided.
        if x.shape[-1] != sum(self.dimensions):
            raise ValueError('Input width does not match modality dimensions')
        # Split into [B, T, d] per modality; each encoder returns [B, C].
        # Stack the M sets of class scores into logits: [M, B, C].
        logits = torch.stack([model(part, lengths) for model, part in
                              zip(self.encoders, x.split(self.dimensions, dim=-1))])
        # Softmax weights: [M] -> [M, 1, 1], shared across videos and classes.
        # Weight logits [M, B, C], then sum over modalities to return [B, C].
        return (logits * self.fusion_weights.softmax(0)[:, None, None]).sum(0)
