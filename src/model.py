"""A randomly initialized two-feature-map SSD-style binary detector."""

import torch
from torch import nn

from src.geometry import make_anchors


def block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.ReLU(),
    )


class VehicleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(block(3, 16), block(16, 32),
                                      block(32, 64), block(64, 96))
        self.extra = block(96, 128)
        self.class_heads = nn.ModuleList([nn.Conv2d(channels, 9 * 2, 3, padding=1)
                                          for channels in (96, 128)])
        self.box_heads = nn.ModuleList([nn.Conv2d(channels, 9 * 4, 3, padding=1)
                                        for channels in (96, 128)])
        self.register_buffer("anchors", make_anchors())

    def forward(self, images):
        if images.shape[1:] != (3, 320, 320):
            raise ValueError("Expected a batch of 3x320x320 CPU images")
        first = self.backbone(images)
        second = self.extra(first)
        classifications, offsets = [], []
        for feature, class_head, box_head in zip((first, second), self.class_heads, self.box_heads):
            batch = feature.shape[0]
            classifications.append(class_head(feature).permute(0, 2, 3, 1).reshape(batch, -1, 2))
            offsets.append(box_head(feature).permute(0, 2, 3, 1).reshape(batch, -1, 4))
        return torch.cat(classifications, dim=1), torch.cat(offsets, dim=1)