"""Scalar-only W&B sidecar. Importing this package never imports wandb/torch."""
from .client import init, Tracker, LoggingBlocked

__all__ = ['init', 'Tracker', 'LoggingBlocked']
