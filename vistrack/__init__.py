"""
High-performance multi-object tracking using Similari's Rust backend.

Vistrack provides a clean, easy-to-use interface for real-time multi-object tracking
with bounding boxes. It leverages Similari's VisualSORT algorithm for high performance.

Examples
--------
>>> from vistrack import Detection, Tracker
>>> import numpy as np
>>>
>>> # Create tracker
>>> tracker = Tracker(max_age=30, min_hits=3)
>>>
>>> # In your tracking loop
>>> detections = [
>>>     Detection(
>>>         bbox=np.array([10, 20, 100, 150]),  # [x1, y1, x2, y2]
>>>         confidence=0.9,
>>>         feature=my_reid_feature,  # Optional appearance feature
>>>     )
>>> ]
>>> tracked_objects = tracker.update(detections)
>>> for obj in tracked_objects:
>>>     print(f"Track {obj.id}: {obj.bbox}")
"""

import sys

from .tracker import Detection, Tracker, TrackedObject
from .distances import euclidean_distance, cosine_distance, iou_distance

if sys.version_info >= (3, 8):
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
elif sys.version_info < (3, 8):
    import importlib_metadata

    __version__ = importlib_metadata.version(__name__)

__all__ = [
    "Detection",
    "Tracker",
    "TrackedObject",
    "euclidean_distance",
    "cosine_distance",
    "iou_distance",
]
