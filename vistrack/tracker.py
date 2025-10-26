"""
Multi-object tracking - rebuilt from scratch for clarity and simplicity.

This module provides a clean interface for real-time multi-object tracking with bounding boxes.
It implements the SORT (Simple Online and Realtime Tracking) algorithm.

Note on Similari Integration:
The Similari library's Python bindings have opaque APIs without clear documentation.
This implementation uses a pure Python SORT-like approach. To integrate actual Similari:
- The VisualSortObservation API requires specific parameter formats that aren't documented
- See comments marked "TODO: Similari" throughout for integration points
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

import numpy as np


@dataclass
class Detection:
    """
    Represents a detection (e.g., from an object detector).

    Attributes
    ----------
    bbox : np.ndarray
        Bounding box as [x1, y1, x2, y2] (top-left and bottom-right corners).
    confidence : float, optional
        Detection confidence score.
    feature : np.ndarray, optional
        Feature vector for appearance-based matching (e.g., from a ReID model).
    label : str, optional
        Class label for the detection.
    metadata : dict, optional
        Additional metadata to attach to the detection.
    """

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float = 1.0
    feature: Optional[np.ndarray] = None
    label: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate detection inputs."""
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox must be shape (4,), got {self.bbox.shape}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.feature is not None and not isinstance(self.feature, np.ndarray):
            raise ValueError(f"feature must be np.ndarray or None, got {type(self.feature)}")

    def to_xyxy(self) -> np.ndarray:
        """Return bbox as [x1, y1, x2, y2]."""
        return self.bbox.astype(np.float32)

    def to_xywh(self) -> np.ndarray:
        """Convert bbox to [x, y, w, h] format."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def to_centroid(self) -> np.ndarray:
        """Return center point of bbox as [cx, cy]."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


@dataclass
class TrackedObject:
    """
    Represents an object being tracked.

    This is a read-only snapshot of the tracker's state for an object at a specific frame.

    Attributes
    ----------
    id : int
        Unique identifier for this track.
    bbox : np.ndarray
        Predicted bounding box as [x1, y1, x2, y2].
    age : int
        Number of frames this track has been active.
    hits : int
        Number of times this track has been matched to a detection.
    confidence : float
        Confidence of the current match.
    feature : Optional[np.ndarray]
        Feature vector from the last matched detection.
    label : Optional[str]
        Class label from the last matched detection.
    metadata : Optional[dict]
        Metadata from the last matched detection.
    """

    id: int
    bbox: np.ndarray
    age: int
    hits: int
    confidence: float = 1.0
    feature: Optional[np.ndarray] = None
    label: Optional[str] = None
    metadata: Optional[dict] = None

    @property
    def estimate(self) -> np.ndarray:
        """Alias for bbox to match common tracking API conventions."""
        return self.bbox

    def to_xyxy(self) -> np.ndarray:
        """Return bbox as [x1, y1, x2, y2]."""
        return self.bbox.astype(np.float32)


class _KalmanFilterSimple:
    """
    Simple 1D Kalman filter for state estimation.

    Tracks position and velocity for each dimension independently.
    """

    def __init__(self, initial_state: float, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """Initialize filter with initial state."""
        self.x = initial_state  # State (position)
        self.v = 0.0  # Velocity
        self.p = 1.0  # Position uncertainty
        self.pv = 0.1  # Velocity uncertainty
        self.q = process_variance  # Process noise
        self.r = measurement_variance  # Measurement noise
        self.dt = 1.0  # Time step

    def predict(self) -> float:
        """Predict next state."""
        self.x = self.x + self.v * self.dt
        self.p = self.p + self.q
        return self.x

    def update(self, measurement: float) -> float:
        """Update state with measurement."""
        # Kalman gain
        k = self.p / (self.p + self.r)
        # Update state
        self.x = self.x + k * (measurement - self.x)
        # Update velocity (simple: difference from previous)
        self.v = (measurement - self.x) / self.dt
        # Update uncertainty
        self.p = (1 - k) * self.p
        return self.x


class _TrackedObjectState:
    """Internal state for a tracked object during tracking."""

    def __init__(self, track_id: int, detection: Detection):
        """Initialize tracking state from a detection."""
        self.track_id = track_id
        self.bbox = detection.bbox.copy()
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.confidence = detection.confidence
        self.feature = detection.feature.copy() if detection.feature is not None else None
        self.label = detection.label
        self.metadata = detection.metadata.copy() if detection.metadata else None

        # Simple Kalman filters for each dimension
        x1, y1, x2, y2 = self.bbox
        self.kf_x1 = _KalmanFilterSimple(x1)
        self.kf_y1 = _KalmanFilterSimple(y1)
        self.kf_x2 = _KalmanFilterSimple(x2)
        self.kf_y2 = _KalmanFilterSimple(y2)

    def predict(self) -> np.ndarray:
        """Predict next bbox."""
        x1 = self.kf_x1.predict()
        y1 = self.kf_y1.predict()
        x2 = self.kf_x2.predict()
        y2 = self.kf_y2.predict()
        self.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.age += 1
        return self.bbox.copy()

    def update(self, detection: Detection) -> None:
        """Update state with a detection."""
        x1, y1, x2, y2 = detection.bbox
        self.kf_x1.update(x1)
        self.kf_y1.update(y1)
        self.kf_x2.update(x2)
        self.kf_y2.update(y2)
        self.bbox = np.array([self.kf_x1.x, self.kf_y1.x, self.kf_x2.x, self.kf_y2.x], dtype=np.float32)
        self.hits += 1
        self.misses = 0
        self.confidence = detection.confidence
        if detection.feature is not None:
            self.feature = detection.feature.copy()
        self.label = detection.label
        if detection.metadata:
            self.metadata = detection.metadata.copy()

    def miss(self) -> None:
        """Record a missed detection."""
        self.misses += 1

    def to_tracked_object(self) -> TrackedObject:
        """Convert to output TrackedObject."""
        return TrackedObject(
            id=self.track_id,
            bbox=self.bbox.copy(),
            age=self.age,
            hits=self.hits,
            confidence=self.confidence,
            feature=self.feature.copy() if self.feature is not None else None,
            label=self.label,
            metadata=self.metadata.copy() if self.metadata else None,
        )


class Tracker:
    """
    Simple, transparent multi-object tracker using SORT algorithm.

    This tracker uses bounding boxes to track objects across frames.
    It implements the SORT (Simple Online and Realtime Tracking) algorithm.

    Parameters
    ----------
    max_age : int, optional
        Maximum number of frames to keep a track alive without detections.
        Default is 30.
    min_hits : int, optional
        Minimum number of detections before a track is considered confirmed.
        Default is 3.
    iou_threshold : float, optional
        IoU threshold for matching detections to tracks.
        Default is 0.3.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """Initialize the tracker."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._track_counter = 0
        self._tracks: Dict[int, _TrackedObjectState] = {}

    def update(
        self,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObject]:
        """
        Update the tracker with new detections.

        Parameters
        ----------
        detections : Optional[List[Detection]]
            List of detections from an object detector. If None or empty,
            the tracker will advance time but return only confirmed tracks.

        Returns
        -------
        List[TrackedObject]
            List of confirmed tracks (age >= min_hits).
        """
        if detections is None:
            detections = []

        # Predict new positions for all tracks
        predicted_bboxes = {}
        for track_id, track in self._tracks.items():
            predicted_bboxes[track_id] = track.predict()

        # Match detections to tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections(
            detections, predicted_bboxes
        )

        # Update matched tracks
        for track_id, det_idx in matched_pairs:
            self._tracks[track_id].update(detections[det_idx])

        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self._tracks[track_id].miss()

        # Create new tracks from unmatched detections
        for det_idx in unmatched_detections:
            self._track_counter += 1
            self._tracks[self._track_counter] = _TrackedObjectState(self._track_counter, detections[det_idx])

        # Remove stale tracks and return confirmed tracks
        result = []
        tracks_to_remove = []
        for track_id, track in self._tracks.items():
            if track.misses > self.max_age:
                tracks_to_remove.append(track_id)
            elif track.hits >= self.min_hits:
                result.append(track.to_tracked_object())

        for track_id in tracks_to_remove:
            del self._tracks[track_id]

        return result

    def _match_detections(
        self,
        detections: List[Detection],
        predicted_bboxes: Dict[int, np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU distance.

        Returns
        -------
        matched_pairs : List[Tuple[int, int]]
            List of (track_id, detection_idx) pairs
        unmatched_detections : List[int]
            Indices of detections that weren't matched
        unmatched_tracks : List[int]
            IDs of tracks that weren't matched
        """
        matched_pairs = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())

        if len(detections) == 0 or len(self._tracks) == 0:
            return matched_pairs, list(unmatched_detections), list(unmatched_tracks)

        # Build distance matrix
        distance_matrix = np.zeros((len(detections), len(self._tracks)))
        track_ids = list(self._tracks.keys())

        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                pred_bbox = predicted_bboxes[track_id]
                iou = self._compute_iou(detection.bbox, pred_bbox)
                # Distance = 1 - IoU
                distance_matrix[det_idx, track_idx] = 1.0 - iou

        # Greedy matching: match minimum distances first
        while True:
            if distance_matrix.size == 0:
                break

            min_idx = np.argmin(distance_matrix)
            det_idx, track_idx = np.unravel_index(min_idx, distance_matrix.shape)
            distance = distance_matrix[det_idx, track_idx]

            if distance >= self.iou_threshold:
                break  # No more good matches

            track_id = track_ids[track_idx]
            matched_pairs.append((track_id, det_idx))
            unmatched_detections.discard(det_idx)
            unmatched_tracks.discard(track_id)

            # Remove matched detection and track from future consideration
            distance_matrix = np.delete(distance_matrix, det_idx, axis=0)
            distance_matrix = np.delete(distance_matrix, track_idx, axis=1)
            track_ids.pop(track_idx)

        return matched_pairs, list(unmatched_detections), list(unmatched_tracks)

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1_a, y1_a, x2_a, y2_a = bbox1
        x1_b, y1_b, x2_b, y2_b = bbox2

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
            return 0.0

        return float(inter_area / union_area)

    @property
    def current_track_count(self) -> int:
        """Number of currently active and confirmed tracks."""
        return len([t for t in self._tracks.values() if t.hits >= self.min_hits])

    def reset(self) -> None:
        """Reset the tracker, clearing all tracks."""
        self._tracks.clear()
        self._track_counter = 0
