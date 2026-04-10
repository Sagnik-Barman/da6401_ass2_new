"""
models/multitask.py
───────────────────
Task 4: Unified Multi-Task Perception Model.

A single forward pass simultaneously produces:
  1. Breed classification logits   [B, 37]
  2. Bounding-box prediction       [B, 4]  (cx, cy, w, h) in pixel space
  3. Segmentation mask logits      [B, 3, H, W]

The model is initialised from three separately-trained checkpoints:
  checkpoints/classifier.pth
  checkpoints/localizer.pth
  checkpoints/unet.pth

For submission, the __init__ uses gdown to download the .pth files
from Google Drive (IDs filled in by the student — Step 3 of README).

Autograder import:
  from multitask import MultiTaskPerceptionModel

Shared-backbone justification
──────────────────────────────
Low-level convolutional features (edges, textures, blobs) are useful
for all three tasks simultaneously.  Sharing the encoder:
  – Reduces total parameter count.
  – Enables each task's gradient to improve shared representations.
  – Acts as a form of multi-task regularisation.

The three task heads are task-specific and operate in parallel from
the same bottleneck feature map.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11        import VGG11, IMG_SIZE, NUM_CLASSES
from models.layers       import CustomDropout
from models.segmentation import _DecoderBlock


# ── Checkpoint paths (relative, as required by README) ────────────────────
_CLASSIFIER_PATH = os.path.join("checkpoints", "classifier.pth")
_LOCALIZER_PATH  = os.path.join("checkpoints", "localizer.pth")
_UNET_PATH       = os.path.join("checkpoints", "unet.pth")


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared VGG11 backbone with three task heads.

    Parameters
    ----------
    num_classes     : int   classification classes (default 37)
    num_seg_classes : int   segmentation classes   (default 3)
    dropout_p       : float dropout in the classification head (default 0.5)
    img_size        : int   expected spatial input size (default 224)
    """

    def __init__(
        self,
        num_classes:     int   = NUM_CLASSES,
        num_seg_classes: int   = 3,
        dropout_p:       float = 0.5,
        img_size:        int   = IMG_SIZE,
    ):
        super().__init__()
        self.img_size = img_size

        # ── Step 3 from README: download checkpoints via gdown ───────────
        import gdown
        gdown.download(id="1TUHvSfGm1nOs6g-tNwjCUwOaa9rx7XkR", output=_CLASSIFIER_PATH, quiet=False)
        gdown.download(id="1siFAVhefFU90IdnnMFKYDlBCWHaTj8JC",  output=_LOCALIZER_PATH,  quiet=False)
        gdown.download(id="1Ao6RdfllVhrXwwBKUEFrOyysAC9JsENa",       output=_UNET_PATH,       quiet=False)

        # ── Shared VGG11 encoder ─────────────────────────────────────────
        self.encoder = VGG11(num_classes=num_classes, dropout_p=dropout_p)

        # ── Task 1: Classification head ──────────────────────────────────
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
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

        # ── Task 2: Bounding-box regression head ─────────────────────────
        self.pool_s2_loc = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_s3_loc = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_s4_loc = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_bt_loc = nn.AdaptiveAvgPool2d((4, 4))

        self.bbox_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear((256 + 512 + 512 + 512) * 4 * 4, 2048),
        nn.ReLU(inplace=True),
        CustomDropout(p=0.2),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 4),
        nn.Sigmoid(),
)
        #self.bbox_head = nn.Sequential(
#            nn.AdaptiveAvgPool2d((4, 4)),
 #           nn.Flatten(),
  #          nn.Linear(512 * 4 * 4, 1024),
   #         nn.ReLU(inplace=True),
    #        CustomDropout(p=0.3),
     #       nn.Linear(1024, 256),
      #      nn.ReLU(inplace=True),
       #     nn.Linear(256, 4),
        #    nn.ReLU(inplace=True),   # pixel coords ≥ 0
        #)

        # ── Task 3: U-Net segmentation decoder ───────────────────────────
        self.dec4 = _DecoderBlock(512, 512, 512)
        self.dec3 = _DecoderBlock(512, 512, 256)
        self.dec2 = _DecoderBlock(256, 256, 128)
        self.dec1 = _DecoderBlock(128, 128, 64)
        self.dec0 = _DecoderBlock(64,  64,  32)
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)

        # ── Load pretrained weights from the three checkpoints ───────────
        self._load_pretrained_weights()

    # ── Weight loading ────────────────────────────────────────────────────
    def _load_pretrained_weights(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classification checkpoint → encoder + cls_head
        if os.path.exists(_CLASSIFIER_PATH):
            cls_state = torch.load(_CLASSIFIER_PATH, map_location=device)
            # Strip 'vgg.' prefix if saved from ClassificationModel wrapper
            enc_state = {}
            head_state = {}
            for k, v in cls_state.items():
                if k.startswith("vgg.block") or k.startswith("vgg.pool") \
                        or k.startswith("vgg.avgpool"):
                    enc_state[k.replace("vgg.", "")] = v
                elif k.startswith("vgg.classifier"):
                    head_state[k.replace("vgg.classifier.", "")] = v
            self.encoder.load_state_dict(enc_state, strict=False)
            self.cls_head.load_state_dict(head_state, strict=False)
            print(f"Loaded classifier weights from {_CLASSIFIER_PATH}")

        # Localizer checkpoint → bbox_head
        if os.path.exists(_LOCALIZER_PATH):
            loc_state = torch.load(_LOCALIZER_PATH, map_location=device)
            rh_state = {k.replace("regression_head.", ""): v
                        for k, v in loc_state.items()
                        if k.startswith("regression_head.")}
            self.bbox_head.load_state_dict(rh_state, strict=False)
            print(f"Loaded localizer weights from {_LOCALIZER_PATH}")

        # U-Net checkpoint → decoder blocks + seg_head
        if os.path.exists(_UNET_PATH):
            seg_state = torch.load(_UNET_PATH, map_location=device)
            self.dec4.load_state_dict(
                {k.replace("dec4.",""): v for k, v in seg_state.items()
                 if k.startswith("dec4.")}, strict=False)
            self.dec3.load_state_dict(
                {k.replace("dec3.",""): v for k, v in seg_state.items()
                 if k.startswith("dec3.")}, strict=False)
            self.dec2.load_state_dict(
                {k.replace("dec2.",""): v for k, v in seg_state.items()
                 if k.startswith("dec2.")}, strict=False)
            self.dec1.load_state_dict(
                {k.replace("dec1.",""): v for k, v in seg_state.items()
                 if k.startswith("dec1.")}, strict=False)
            self.dec0.load_state_dict(
                {k.replace("dec0.",""): v for k, v in seg_state.items()
                 if k.startswith("dec0.")}, strict=False)
            seg_head_state = {k.replace("seg_head.",""): v
                              for k, v in seg_state.items()
                              if k.startswith("seg_head.")}
            self.seg_head.load_state_dict(seg_head_state, strict=False)
            print(f"Loaded segmentation weights from {_UNET_PATH}")

    # ── Single unified forward pass ───────────────────────────────────────
    def forward(self, x: torch.Tensor):
        bottleneck, skips = self.encoder.encode(x)
        s0, s1, s2, s3, s4 = skips

    # Task 1: classification
        feat_cls   = self.avgpool(bottleneck)
        feat_cls   = torch.flatten(feat_cls, 1)
        cls_logits = self.cls_head(feat_cls)

    # Task 2: bounding-box regression
    #    bbox = self.bbox_head(bottleneck)
    # Task 2: bounding-box regression
    #    raw = self.bbox_head(bottleneck)   # [B, 4]
    #    x1, y1, x2, y2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    #    cx = (x1 + x2) / 2
    #    cy = (y1 + y2) / 2
    #    w  =  x2 - x1
    #    h  =  y2 - y1
    #    bbox = torch.stack([cx, cy, w, h], dim=1)
    # ── Task 2: Bounding-box regression head ─────────────────────────
        # Task 2: bounding-box regression
        f2 = self.pool_s2_loc(s2)
        f3 = self.pool_s3_loc(s3)
        f4 = self.pool_s4_loc(s4)
        fb = self.pool_bt_loc(bottleneck)
        fused_loc = torch.cat([f2, f3, f4, fb], dim=1)
        bbox = self.bbox_head(fused_loc) * self.img_size

    # Task 3: segmentation decoder
        d = self.dec4(bottleneck, s4)
        d = self.dec3(d,          s3)
        d = self.dec2(d,          s2)
        d = self.dec1(d,          s1)
        d = self.dec0(d,          s0)
        seg_logits = self.seg_head(d)

        return {
        'classification': cls_logits,   # [B, 37]
        'localization':   bbox,          # [B, 4]
        'segmentation':   seg_logits,    # [B, 3, 224, 224]
    }
