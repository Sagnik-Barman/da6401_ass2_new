"""
multitask.py  (top-level, for autograder)
─────────────────────────────────────────
Re-exports MultiTaskPerceptionModel so the autograder can do:
  from multitask import MultiTaskPerceptionModel
"""

from models.multitask import MultiTaskPerceptionModel   # noqa: F401

__all__ = ["MultiTaskPerceptionModel"]
