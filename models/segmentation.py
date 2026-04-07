"""
models/segmentation.py
──────────────────────
Task 3: U-Net style semantic segmentation using VGG11 as the encoder
(contracting path).

Architecture
────────────
Encoder (VGG11 contracting path)
  block0 → pool0  [B,64,224,224]  → [B,64,112,112]
  block1 → pool1  [B,128,112,112] → [B,128,56,56]
  block2 → pool2  [B,256,56,56]   → [B,256,28,28]
  block3 → pool3  [B,512,28,28]   → [B,512,14,14]
  block4 → pool4  [B,512,14,14]   → [B,512,7,7]   ← bottleneck

Decoder (symmetric expansive path — transposed convolutions)
  dec4: TransposedConv(512→512) + concat(s4[512,14,14])  → [B,512,14,14]
  dec3: TransposedConv(512→256) + concat(s3[512,28,28])  → [B,256,28,28]
  dec2: TransposedConv(256→128) + concat(s2[256,56,56])  → [B,128,56,56]
  dec1: TransposedConv(128→64)  + concat(s1[128,112,112])→ [B,64,112,112]
  dec0: TransposedConv(64→32)   + concat(s0[64,224,224]) → [B,32,224,224]
  seg_head: Conv1×1 → num_seg_classes

Transposed convolutions (not bilinear):
  Bilinear upsampling is a fixed, non-learnable operation.  Transposed
  convolutions allow the network to learn how to reconstruct spatial
  detail — especially important at object boundaries in trimaps.

Loss choice: Focal + Dice composite (see losses/ module).
  The Oxford Pet trimap has heavily imbalanced class frequencies
  (background ≫ foreground, border very sparse).  Plain cross-entropy
  is dominated by the majority class.  Dice Loss directly maximises
  pixel-wise overlap for every class.  Focal Loss down-weights the
  large number of correctly-classified background pixels, focusing
  training on hard, boundary-adjacent pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11, IMG_SIZE


# ── Decoder block ────────────────────────────────────────────────────────
class _DecoderBlock(nn.Module):
    """
    TransposedConv upsample (stride 2) → concat skip → Conv+BN+ReLU ×2.

    Parameters
    ----------
    in_ch   : channels coming from the deeper decoder level
    skip_ch : channels from the skip connection
    out_ch  : output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Learnable upsampling — doubles spatial resolution
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial sizes differ (odd dims after pooling)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── U-Net ────────────────────────────────────────────────────────────────
class SegmentationModel(nn.Module):
    """
    U-Net with VGG11 encoder and symmetric transposed-conv decoder.

    Parameters
    ----------
    encoder         : VGG11 instance (None → fresh VGG11)
    num_seg_classes : int   number of segmentation classes (default 3)
    freeze_encoder  : bool  freeze encoder weights?
    """

    def __init__(
        self,
        encoder: VGG11       = None,
        num_seg_classes: int = 3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else VGG11()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # Decoder — mirrors the VGG11 encoder stages
        self.dec4 = _DecoderBlock(512, 512, 512)
        self.dec3 = _DecoderBlock(512, 512, 256)
        self.dec2 = _DecoderBlock(256, 256, 128)
        self.dec1 = _DecoderBlock(128, 128, 64)
        self.dec0 = _DecoderBlock(64,  64,  32)

        # 1×1 conv to produce per-pixel class logits
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, 224, 224]

        Returns
        -------
        seg_logits : Tensor [B, num_seg_classes, 224, 224]
        """
        bottleneck, skips = self.encoder.encode(x)
        s0, s1, s2, s3, s4 = skips

        d = self.dec4(bottleneck, s4)
        d = self.dec3(d,          s3)
        d = self.dec2(d,          s2)
        d = self.dec1(d,          s1)
        d = self.dec0(d,          s0)

        return self.seg_head(d)
