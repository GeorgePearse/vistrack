"""
Filter module - note that filtering is now handled internally by Similari.

This module is deprecated as Similari's VisualSORT handles all Kalman filtering internally.
It's kept for compatibility with downstream code that might reference it.
"""

import warnings

warnings.warn(
    "The filter module is deprecated. Similari handles Kalman filtering internally.",
    DeprecationWarning,
    stacklevel=2,
)
