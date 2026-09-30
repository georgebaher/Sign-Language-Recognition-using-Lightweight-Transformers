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
        self.encoders = nn.ModuleList([EncoderOnly(d, num_classes, **kwargs) for d in dimensions])
        self.fusion_weights = nn.Parameter(torch.zeros(len(dimensions)))

    def forward(self, x, lengths=None):
        if x.shape[-1] != sum(self.dimensions):
            raise ValueError('Input width does not match modality dimensions')
        logits = torch.stack([model(part, lengths) for model, part in
                              zip(self.encoders, x.split(self.dimensions, dim=-1))])
        return (logits * self.fusion_weights.softmax(0)[:, None, None]).sum(0)
