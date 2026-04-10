"""
models/localization.py
──────────────────────
Improved localization using multi-scale features from the VGG11 encoder.
Uses features from multiple stages instead of just the bottleneck.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11, IMG_SIZE
from models.layers import CustomDropout


class LocalizationModel(nn.Module):
    def __init__(
        self,
        encoder: VGG11      = None,
        freeze_encoder: bool = False,
        dropout_p: float    = 0.3,
        img_size: int       = IMG_SIZE,
    ):
        super().__init__()
        self.img_size = img_size
        self.encoder  = encoder if encoder is not None else VGG11()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # Multi-scale pooling from different encoder stages
        self.pool_s2 = nn.AdaptiveAvgPool2d((4, 4))   # from stage2: 256ch
        self.pool_s3 = nn.AdaptiveAvgPool2d((4, 4))   # from stage3: 512ch
        self.pool_s4 = nn.AdaptiveAvgPool2d((4, 4))   # from stage4: 512ch
        self.pool_bt = nn.AdaptiveAvgPool2d((4, 4))   # bottleneck:  512ch

        # Fuse multi-scale: 512*3 * 4*4 = 24576 features
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear((256 + 512 + 512 + 512) * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.2),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder.encode(x)
        s0, s1, s2, s3, s4 = skips

        f2 = self.pool_s2(s2)
        f3 = self.pool_s3(s3)
        f4 = self.pool_s4(s4)
        fb = self.pool_bt(bottleneck)

        fused = torch.cat([f2, f3, f4, fb], dim=1)
        bbox_norm = self.regression_head(fused)
        bbox = bbox_norm * self.img_size  # now in pixel space

        # Convert [x1, y1, x2, y2] → [cx, cy, w, h] if checkpoint was trained that way
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        return torch.stack([cx, cy, w, h], dim=1)

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder.encode(x)
        s0, s1, s2, s3, s4 = skips

        f2 = self.pool_s2(s2)
        f3 = self.pool_s3(s3)
        f4 = self.pool_s4(s4)
        fb = self.pool_bt(bottleneck)

        fused = torch.cat([f2, f3, f4, fb], dim=1)  # [B, 1792, 4, 4]
        bbox_norm = self.regression_head(fused)
        return bbox_norm * self.img_size