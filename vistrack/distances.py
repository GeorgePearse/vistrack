"""
Distance metrics and matching utilities.

Note: Similari handles distance calculations internally (IoU, Mahalanobis, Euclidean, cosine).
This module is kept for compatibility but is largely deprecated.
"""

import warnings
from typing import Callable, Optional

import numpy as np

warnings.warn(
    "The distances module is deprecated. Similari handles distance calculations internally.",
    DeprecationWarning,
    stacklevel=2,
)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance.
    """
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector (should be normalized).
    b : np.ndarray
        Second vector (should be normalized).

    Returns
    -------
    float
        Cosine distance (1 - cosine similarity).
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_norm, b_norm))


def iou_distance(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """
    Compute 1 - IoU distance between two bounding boxes.

    Parameters
    ----------
    bbox_a : np.ndarray
        First bbox as [x1, y1, x2, y2].
    bbox_b : np.ndarray
        Second bbox as [x1, y1, x2, y2].

    Returns
    -------
    float
        1 - IoU distance.
    """
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    # Intersection
    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 1.0

    iou = inter_area / union_area
    return float(1.0 - iou)
