"""
Evaluate tracker on MOT sequences.

Runs the tracker and computes MOT metrics (MOTA, MOTP, IDF1, etc.)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List
import tempfile

import numpy as np
import cv2
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vistrack import Detection, Tracker, TrackedObject
from mot_loader import MotSequence, load_mot_dataset

try:
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False


class MotEvaluator:
    """Evaluates tracker on MOT sequences."""

    def __init__(self, tracker_config: Dict = None):
        """Initialize evaluator.

        Parameters
        ----------
        tracker_config : Dict, optional
            Tracker configuration parameters
        """
        self.tracker_config = tracker_config or {}
        self.tracker = Tracker(**self.tracker_config)

    def run_on_sequence(self, sequence: MotSequence, verbose: bool = False) -> Dict:
        """Run tracker on a single sequence.

        Parameters
        ----------
        sequence : MotSequence
            MOT sequence to track on
        verbose : bool
            Print progress

        Returns
        -------
        Dict
            Tracking results with MOT metrics
        """
        # Reset tracker
        self.tracker.reset()

        # Storage for results
        predictions = {}  # frame_id -> list of (x1, y1, x2, y2, track_id)
        confidences = {}

        # Iterate through frames
        frame_iterator = tqdm(range(1, sequence.num_frames + 1)) if verbose else range(1, sequence.num_frames + 1)

        for frame_id in frame_iterator:
            img = sequence.get_frame(frame_id)
            if img is None:
                continue

            # Simple detection: use ground truth boxes
            # (In real scenario, you'd use a detector)
            gt_boxes, gt_track_ids = sequence.get_gt_boxes_for_frame(frame_id)

            if len(gt_boxes) == 0:
                predictions[frame_id] = np.empty((0, 5), dtype=np.float32)
                continue

            # Create detections from ground truth
            detections = [
                Detection(bbox=box, confidence=1.0) for box in gt_boxes
            ]

            # Track
            tracked_objects = self.tracker.update(detections)

            # Store predictions in MOT format: [x1, y1, x2, y2, track_id, confidence]
            frame_predictions = []
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj.bbox
                frame_predictions.append([x1, y1, x2, y2, float(obj.id), 1.0])

            if frame_predictions:
                predictions[frame_id] = np.array(frame_predictions, dtype=np.float32)
            else:
                predictions[frame_id] = np.empty((0, 6), dtype=np.float32)

        # Compute metrics if motmetrics is available
        metrics = {
            "num_frames": sequence.num_frames,
            "num_predictions": sum(len(v) for v in predictions.values()),
        }

        if MOTMETRICS_AVAILABLE:
            metrics.update(self._compute_mot_metrics(sequence, predictions))

        return metrics

    def _compute_mot_metrics(self, sequence: MotSequence, predictions: Dict) -> Dict:
        """Compute MOT metrics using motmetrics library.

        Parameters
        ----------
        sequence : MotSequence
            MOT sequence
        predictions : Dict
            Predictions in MOT format

        Returns
        -------
        Dict
            Computed metrics
        """
        acc = mm.MOTAccumulator(auto_id=True)

        # Iterate through frames
        for frame_id in range(1, sequence.num_frames + 1):
            # Ground truth
            gt_boxes, gt_track_ids = sequence.get_gt_boxes_for_frame(frame_id)

            # Predictions
            if frame_id in predictions:
                pred_data = predictions[frame_id]
                if len(pred_data) > 0:
                    pred_boxes = pred_data[:, :4]
                    pred_track_ids = pred_data[:, 4].astype(int)
                else:
                    pred_boxes = np.empty((0, 4), dtype=np.float32)
                    pred_track_ids = np.empty(0, dtype=int)
            else:
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_track_ids = np.empty(0, dtype=int)

            # Compute IoU distance
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                distances = np.zeros((len(gt_track_ids), len(pred_track_ids)))
                for i, gt_box in enumerate(gt_boxes):
                    for j, pred_box in enumerate(pred_boxes):
                        distances[i, j] = self._iou_distance(gt_box, pred_box)
            else:
                distances = np.empty((len(gt_track_ids), len(pred_track_ids)))

            # Update accumulator
            acc.update(gt_track_ids, pred_track_ids, distances)

        # Compute metrics
        metrics_dict = mm.metrics.compute(acc, metrics=mm.metrics.METRICS, name="MOT")

        return {
            "mota": float(metrics_dict["MOTA"]) if "MOTA" in metrics_dict else 0.0,
            "motp": float(metrics_dict["MOTP"]) if "MOTP" in metrics_dict else 0.0,
            "idf1": float(metrics_dict.get("IDF1", 0.0)),
            "idr": float(metrics_dict.get("IDR", 0.0)),
            "idp": float(metrics_dict.get("IDP", 0.0)),
            "num_detections": int(metrics_dict.get("NUM_DETECTIONS", 0)),
            "num_false_positives": int(metrics_dict.get("FP", 0)),
            "num_misses": int(metrics_dict.get("FN", 0)),
            "num_switches": int(metrics_dict.get("IDsw", 0)),
        }

    @staticmethod
    def _iou_distance(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute 1 - IoU distance between two boxes.

        Parameters
        ----------
        box1, box2 : np.ndarray
            Boxes as [x1, y1, x2, y2]

        Returns
        -------
        float
            Distance (1 - IoU)
        """
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

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
        return 1.0 - iou


def evaluate_dataset(dataset_dir: str, tracker_config: Dict = None, split: str = "train", verbose: bool = False) -> Dict:
    """Evaluate tracker on entire MOT dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to MOT dataset (MOT17 or MOT20)
    tracker_config : Dict, optional
        Tracker configuration
    split : str
        Which split to evaluate ("train", "test", or "all")
    verbose : bool
        Print progress

    Returns
    -------
    Dict
        Results for each sequence
    """
    dataset = load_mot_dataset(dataset_dir)
    evaluator = MotEvaluator(tracker_config)

    # Select sequences
    if split == "train":
        sequences = dataset.train_sequences
    elif split == "test":
        sequences = dataset.test_sequences
    else:  # "all"
        sequences = dataset.all_sequences

    print(f"\n{'='*60}")
    print(f"Evaluating {dataset.name} ({split} split)")
    print(f"{'='*60}")
    print(f"Sequences: {len(sequences)}")
    print(f"Tracker config: {tracker_config}")
    print(f"{'='*60}\n")

    results = {}
    for sequence in sequences:
        print(f"Evaluating {sequence.name}...")
        seq_results = evaluator.run_on_sequence(sequence, verbose=verbose)
        results[sequence.name] = seq_results
        print(f"  Frames: {seq_results['num_frames']}, Predictions: {seq_results['num_predictions']}")
        if "mota" in seq_results:
            print(f"  MOTA: {seq_results['mota']:.2f}, MOTP: {seq_results['motp']:.2f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tracker on MOT dataset")
    parser.add_argument("dataset_dir", help="Path to MOT dataset")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    tracker_config = {
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "iou_threshold": args.iou_threshold,
    }

    results = evaluate_dataset(
        args.dataset_dir,
        tracker_config=tracker_config,
        split=args.split,
        verbose=args.verbose,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary Results")
    print(f"{'='*60}")
    for seq_name, metrics in sorted(results.items()):
        print(f"{seq_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
