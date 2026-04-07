"""
models/layers.py
────────────────
Custom neural-network layers.

CustomDropout
─────────────
Implements *inverted dropout* (same convention as PyTorch's built-in),
but without using torch.nn.Dropout or torch.nn.functional.dropout.

Inverted-dropout algorithm
──────────────────────────
Training mode:
  1. Sample a Bernoulli mask M ~ Bernoulli(1 − p)  (each element
     independently kept with probability 1−p, zeroed with probability p).
  2. Scale: output = input * M / (1 − p)

  The division by (1−p) ensures that E[output] = input, so no
  rescaling is needed at inference time.

Evaluation mode (self.training == False):
  output = input  (identity — the network is used as-is).

Design & placement justification
──────────────────────────────────
• Dropout is placed *after* BatchNorm + ReLU in the fully-connected
  (FC) classifier layers only.
  – Placing it before BN would corrupt the running mean/variance
    statistics that BN maintains across mini-batches.
  – Placing it after ReLU means we stochastically drop positive
    activations — those that carry information — maximising the
    regularisation benefit.
• We deliberately omit Dropout from the convolutional layers because:
  – Spatial feature maps exhibit strong local correlation, so dropping
    individual activations has little regularisation effect on the
    spatial patterns.
  – BatchNorm already provides implicit regularisation in conv layers
    (smoothed gradient flow, reduced internal covariate shift).
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Inverted dropout layer (no torch.nn.Dropout / F.dropout used).

    Parameters
    ----------
    p : float
        Probability of zeroing an element.  Must be in [0, 1).
        Default: 0.5
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(
                f"Dropout probability p must satisfy 0 ≤ p < 1.  Got p={p}."
            )
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In eval mode (or p == 0) pass through unchanged
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p

        # Bernoulli mask: 1 with probability keep_prob
        mask = torch.bernoulli(
            torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device)
        )

        # Inverted scaling: divide by keep_prob to preserve expected value
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
