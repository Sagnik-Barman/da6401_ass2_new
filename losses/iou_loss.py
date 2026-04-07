"""
losses/iou_loss.py
──────────────────
Custom Intersection-over-Union loss.

Requirements (from README):
  • Inherits from torch.nn.Module.
  • Output range [0, 1].
  • Must implement "mean" and "sum" reduction (default = "mean").
  • No built-in IoU loss functions from external libraries.

Input box format : [x_center, y_center, width, height] in PIXEL space.

Loss = 1 − IoU  ∈ [0, 1]
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Differentiable IoU loss for axis-aligned bounding boxes.

    Parameters
    ----------
    reduction : str  'mean' | 'sum' | 'none'  (default 'mean')
    eps       : float  small constant for numerical stability

    Why IoU instead of MSE?
    ───────────────────────
    MSE treats every coordinate error independently and is not
    scale-invariant.  IoU measures geometric overlap directly:
    the same pixel error matters less for a large box than a small one.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'. Got '{reduction}'."
            )
        self.reduction = reduction
        self.eps       = eps

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor):
        """[cx, cy, w, h] → [x1, y1, x2, y2]"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return x1, y1, x2, y2

    # ── forward ──────────────────────────────────────────────────────────
    def forward(
        self,
        pred:   torch.Tensor,   # [B, 4]  predicted  (cx, cy, w, h)
        target: torch.Tensor,   # [B, 4]  ground-truth (cx, cy, w, h)
    ) -> torch.Tensor:
        """
        Returns
        -------
        Scalar (or per-sample tensor if reduction='none') in [0, 1].
        """
        px1, py1, px2, py2 = self._cxcywh_to_xyxy(pred)
        tx1, ty1, tx2, ty2 = self._cxcywh_to_xyxy(target)

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        inter_w = torch.clamp(ix2 - ix1, min=0.0)
        inter_h = torch.clamp(iy2 - iy1, min=0.0)
        inter   = inter_w * inter_h

        # Areas  (clamp to avoid negative widths/heights from early training)
        pred_area   = torch.clamp(pred[:, 2],   min=0.0) \
                    * torch.clamp(pred[:, 3],   min=0.0)
        target_area = torch.clamp(target[:, 2], min=0.0) \
                    * torch.clamp(target[:, 3], min=0.0)

        union = pred_area + target_area - inter
        iou   = inter / (union + self.eps)   # ∈ [0, 1]
        loss  = 1.0 - iou                    # ∈ [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss   # 'none' → per-sample tensor

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, eps={self.eps}"
