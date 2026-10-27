"""Model definition shared by training and ONNX export.

Small on purpose: the target artefact has to download and run in a browser tab.
~1.1M parameters, 64x64x1 input, four stages of residual blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = (24, 48, 96, 192)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short: nn.Module | None = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        res = x if self.short is None else self.short(x)
        return F.relu(out + res, inplace=True)


class SketchNet(nn.Module):
    def __init__(self, num_classes: int, widths: tuple[int, ...] = WIDTHS, dropout: float = 0.1) -> None:
        super().__init__()
        w1, w2, w3, w4 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(1, w1, 3, 1, 1, bias=False), nn.BatchNorm2d(w1), nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(Block(w1, w1), Block(w1, w1))          # 64x64
        self.stage2 = nn.Sequential(Block(w1, w2, 2), Block(w2, w2))        # 32x32
        self.stage3 = nn.Sequential(Block(w2, w3, 2), Block(w3, w3))        # 16x16
        self.stage4 = nn.Sequential(Block(w3, w4, 2), Block(w4, w4))        # 8x8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(w4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(num_classes: int) -> SketchNet:
    return SketchNet(num_classes)
