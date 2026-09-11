"""Shared constants for the Fellegi-Sunter scoring engine."""

DEFAULT_PRIOR = 0.0001
DEFAULT_THRESHOLD = 0.85

_LOG_CLIP = 690.0  # ln(1e300) clip on the total bayes factor