"""
Data loader for MOT (Multiple Object Tracking) challenge datasets.

Supports MOT17 and MOT20 formats.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2
from tqdm import tqdm


@dataclass
class MotAnnotation:
    """Single annotation from MOT format."""

    frame_id: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    conf: float
    class_id: int
    visibility: float

    def to_xyxy(self) -> np.ndarray:
        """Convert to [x1, y1, x2, y2] format."""
        return np.array([
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        ], dtype=np.float32)

    def to_xywh(self) -> np.ndarray:
        """Convert to [x, y, w, h] format."""
        return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)


class MotSequence:
    """Single MOT sequence with video and annotations."""

    def __init__(self, sequence_dir: Path, is_training: bool = True):
        """Initialize MOT sequence.

        Parameters
        ----------
        sequence_dir : Path
            Path to sequence directory (e.g., MOT17-02)
        is_training : bool
            Whether this is training or test split
        """
        self.sequence_dir = Path(sequence_dir)
        self.name = self.sequence_dir.name
        self.is_training = is_training

        # Load seqinfo
        self.seqinfo = self._load_seqinfo()
        self.num_frames = self.seqinfo["imgsNum"]
        self.width = self.seqinfo["imWidth"]
        self.height = self.seqinfo["imHeight"]
        self.fps = self.seqinfo.get("frameRate", 30)

        # Load ground truth annotations if available
        self.gt_file = self.sequence_dir / "gt" / "gt.txt"
        self.annotations = self._load_annotations() if self.gt_file.exists() else {}

    def _load_seqinfo(self) -> Dict:
        """Load sequence information from seqinfo.ini."""
        seqinfo_file = self.sequence_dir / "seqinfo.ini"

        if not seqinfo_file.exists():
            raise FileNotFoundError(f"seqinfo.ini not found in {self.sequence_dir}")

        seqinfo = {}
        with open(seqinfo_file) as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    # Try to convert to int, otherwise keep as string
                    try:
                        seqinfo[key] = int(value)
                    except ValueError:
                        seqinfo[key] = value

        return seqinfo

    def _load_annotations(self) -> Dict[int, List[MotAnnotation]]:
        """Load ground truth annotations.

        Returns
        -------
        Dict[int, List[MotAnnotation]]
            Annotations indexed by frame_id
        """
        annotations = {}

        with open(self.gt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 9:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                conf = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])

                # Skip invalid annotations
                if conf < 0 or visibility < 0.1:
                    continue

                # Skip crowd annotations (class_id == 0)
                if class_id == 0:
                    continue

                annotation = MotAnnotation(
                    frame_id=frame_id,
                    track_id=track_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    conf=conf,
                    class_id=class_id,
                    visibility=visibility,
                )

                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append(annotation)

        return annotations

    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get frame image.

        Parameters
        ----------
        frame_id : int
            Frame number (1-indexed)

        Returns
        -------
        Optional[np.ndarray]
            BGR image, or None if not found
        """
        img_dir = self.sequence_dir / "img1"
        # MOT uses 1-indexed frame IDs with 6-digit padding
        img_file = img_dir / f"{frame_id:06d}.jpg"

        if not img_file.exists():
            return None

        return cv2.imread(str(img_file))

    def get_gt_annotations(self, frame_id: int) -> List[MotAnnotation]:
        """Get ground truth annotations for a frame.

        Parameters
        ----------
        frame_id : int
            Frame number (1-indexed)

        Returns
        -------
        List[MotAnnotation]
            Annotations for this frame
        """
        return self.annotations.get(frame_id, [])

    def get_gt_boxes_for_frame(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get ground truth boxes and track IDs for a frame.

        Parameters
        ----------
        frame_id : int
            Frame number (1-indexed)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            boxes: (n, 4) array of [x1, y1, x2, y2]
            track_ids: (n,) array of track IDs
        """
        annotations = self.get_gt_annotations(frame_id)

        if not annotations:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int32)

        boxes = np.array([ann.to_xyxy() for ann in annotations], dtype=np.float32)
        track_ids = np.array([ann.track_id for ann in annotations], dtype=np.int32)

        return boxes, track_ids

    def __len__(self) -> int:
        """Total number of frames in sequence."""
        return self.num_frames

    def __repr__(self) -> str:
        return (
            f"MotSequence({self.name}, frames={self.num_frames}, "
            f"size={self.width}x{self.height}, fps={self.fps})"
        )


class MotDataset:
    """MOT dataset with train/test splits."""

    def __init__(self, dataset_dir: Path):
        """Initialize MOT dataset.

        Parameters
        ----------
        dataset_dir : Path
            Path to MOT dataset root (MOT17 or MOT20)
        """
        self.dataset_dir = Path(dataset_dir)
        self.name = self.dataset_dir.name

        # Load sequences
        self.train_sequences = self._load_sequences("train")
        self.test_sequences = self._load_sequences("test")

    def _load_sequences(self, split: str) -> List[MotSequence]:
        """Load all sequences in a split."""
        split_dir = self.dataset_dir / split

        if not split_dir.exists():
            return []

        sequences = []
        for seq_dir in sorted(split_dir.glob("*")):
            if not seq_dir.is_dir():
                continue

            try:
                seq = MotSequence(seq_dir, is_training=(split == "train"))
                sequences.append(seq)
            except Exception as e:
                print(f"Warning: Failed to load {seq_dir}: {e}")

        return sequences

    @property
    def all_sequences(self) -> List[MotSequence]:
        """All sequences (train + test)."""
        return self.train_sequences + self.test_sequences

    def __len__(self) -> int:
        """Total number of sequences."""
        return len(self.all_sequences)

    def __repr__(self) -> str:
        return (
            f"MotDataset({self.name}, "
            f"train={len(self.train_sequences)}, "
            f"test={len(self.test_sequences)})"
        )


def load_mot_dataset(dataset_dir: str) -> MotDataset:
    """Load MOT dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to MOT dataset root (MOT17 or MOT20)

    Returns
    -------
    MotDataset
        Loaded dataset
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset = MotDataset(dataset_dir)

    if len(dataset) == 0:
        raise ValueError(f"No sequences found in {dataset_dir}")

    return dataset
