"""
models/classification.py
────────────────────────
Task 1: thin wrapper around VGG11 for breed classification.
Kept separate so it can be imported / checkpointed independently.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11, NUM_CLASSES


class ClassificationModel(nn.Module):
    """
    37-class pet breed classifier built on VGG11.

    Parameters
    ----------
    num_classes : int   (default 37)
    dropout_p   : float dropout probability in the FC head (default 0.5)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = 0.5):
        super().__init__()
        self.vgg = VGG11(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        logits : [B, num_classes]
        """
        return self.vgg(x)
