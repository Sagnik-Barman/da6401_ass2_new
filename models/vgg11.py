"""
models/vgg11.py
───────────────
VGG11 implemented from scratch following the original paper
(Simonyan & Zisserman, 2014 — https://arxiv.org/abs/1409.1556).

Architecture (VGG11 / config A from the paper)
───────────────────────────────────────────────
Block 0: Conv(64)        → MaxPool2d   [224 → 112]
Block 1: Conv(128)       → MaxPool2d   [112 → 56]
Block 2: Conv(256)×2     → MaxPool2d   [56  → 28]
Block 3: Conv(512)×2     → MaxPool2d   [28  → 14]
Block 4: Conv(512)×2     → MaxPool2d   [14  → 7]
Flatten → FC(4096) → FC(4096) → FC(num_classes)

Modifications vs. original paper
──────────────────────────────────
1. BatchNorm2d after every Conv2d (before ReLU).
   Justification: BN normalises pre-activations, stabilises training,
   and allows higher learning rates (Ioffe & Szegedy, 2015).
2. CustomDropout (p=0.5) after each of the first two FC layers.
   Justification: see models/layers.py docstring.
3. AdaptiveAvgPool2d before the classifier ensures the model accepts
   any input resolution ≥ 32×32 while always producing a 7×7 feature
   map for the FC head.

The VGG11 class also exposes `features` (the conv backbone) and
`classifier` (the FC head) as separate attributes so that downstream
tasks (localisation, segmentation, multi-task) can reuse the backbone.

Autograder note
───────────────
  from models.vgg11 import VGG11
"""

import torch
import torch.nn as nn
from models.layers import CustomDropout

IMG_SIZE    = 224
NUM_CLASSES = 37


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d → BatchNorm2d → ReLU (bias=False because BN absorbs bias)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11(nn.Module):
    """
    VGG11 (config A) with BatchNorm and CustomDropout.

    Parameters
    ----------
    num_classes : int   number of output classes (default 37 for Oxford Pet)
    dropout_p   : float dropout probability in the classifier (default 0.5)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = 0.5):
        super().__init__()

        # ── Convolutional feature extractor ─────────────────────────────
        # Keep individual blocks as named attributes so that skip
        # connections can be extracted by LocalizationModel / UNet.
        self.block0 = nn.Sequential(_conv_bn_relu(3,   64))   # → 224
        self.pool0  = nn.MaxPool2d(kernel_size=2, stride=2)    # → 112

        self.block1 = nn.Sequential(_conv_bn_relu(64,  128))  # → 112
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)    # → 56

        self.block2 = nn.Sequential(                           # → 56
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)    # → 28

        self.block3 = nn.Sequential(                           # → 28
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)    # → 14

        self.block4 = nn.Sequential(                           # → 14
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)    # → 7

        # Convenience attribute: the whole conv stack as a sequential
        # (used when the model is treated as a plain encoder)
        self.features = nn.Sequential(
            self.block0, self.pool0,
            self.block1, self.pool1,
            self.block2, self.pool2,
            self.block3, self.pool3,
            self.block4, self.pool4,
        )

        # ── Adaptive pooling ────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ── Classifier head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    # ── weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward pass (full classification pipeline) ───────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encode(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    # ── encoder-only forward (used by downstream tasks) ───────────────────
    def encode(self, x: torch.Tensor):
        """
        Run only the convolutional backbone.

        Returns
        -------
        out   : Tensor [B, 512, 7, 7]   deep feature map (post pool4)
        skips : list of 5 Tensors       pre-pool feature maps for skip connections
                [s0, s1, s2, s3, s4]
                shapes: [B,64,224,224], [B,128,112,112],
                        [B,256,56,56],  [B,512,28,28], [B,512,14,14]
        """
        s0  = self.block0(x)
        p0  = self.pool0(s0)
        s1  = self.block1(p0)
        p1  = self.pool1(s1)
        s2  = self.block2(p1)
        p2  = self.pool2(s2)
        s3  = self.block3(p2)
        p3  = self.pool3(s3)
        s4  = self.block4(p3)
        out = self.pool4(s4)
        return out, [s0, s1, s2, s3, s4]
