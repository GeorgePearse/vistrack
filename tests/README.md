# Vistrack Benchmark Suite

Comprehensive testing and benchmarking pipeline for Vistrack multi-object tracking on MOT17 and MOT20 datasets.

## Overview

This testing suite provides:
- **MOT Dataset Support**: Loaders for MOT17 and MOT20 formats
- **Automated Evaluation**: Run tracker on sequences and compute MOT metrics
- **Benchmark Reports**: Compare tracker configurations with detailed statistics
- **Extensible Design**: Easy to add new tracker configs or metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r tests/requirements.txt
```

### 2. Download Datasets

Visit [MOTChallenge.net](https://motchallenge.net/) and download:
- **MOT17**: https://motchallenge.net/data/MOT17/
- **MOT20**: https://motchallenge.net/data/MOT20/

Organize as follows:
```
tests/data/
├── MOT17/
│   ├── train/
│   │   ├── MOT17-02/
│   │   ├── MOT17-04/
│   │   └── ...
│   └── test/
│       ├── MOT17-01/
│       ├── MOT17-03/
│       └── ...
└── MOT20/
    ├── train/
    │   ├── MOT20-01/
    │   ├── MOT20-02/
    │   └── ...
    └── test/
        ├── MOT20-03/
        ├── MOT20-05/
        └── ...
```

### 3. Run Benchmarks

**Full benchmark suite (MOT17 + MOT20, multiple configs)**:
```bash
python tests/benchmark.py
```

**MOT17 only, single config**:
```bash
python tests/evaluate.py tests/data/MOT17 --split train
```

**MOT20 with custom config**:
```bash
python tests/evaluate.py tests/data/MOT20 --split train \
    --max-age 40 --min-hits 1 --iou-threshold 0.2
```

## Modules

### `mot_loader.py`

Data loader for MOT format datasets.

**Key Classes:**
- `MotSequence`: Single MOT video sequence with annotations
- `MotDataset`: Full MOT dataset (train/test splits)

**Usage:**
```python
from mot_loader import load_mot_dataset

dataset = load_mot_dataset("tests/data/MOT17")
for sequence in dataset.train_sequences:
    print(f"{sequence.name}: {sequence.num_frames} frames")

    for frame_id in range(1, sequence.num_frames + 1):
        frame = sequence.get_frame(frame_id)
        boxes, track_ids = sequence.get_gt_boxes_for_frame(frame_id)
```

### `evaluate.py`

Tracker evaluation on MOT sequences.

**Key Classes:**
- `MotEvaluator`: Runs tracker and computes metrics

**Key Functions:**
- `evaluate_dataset()`: Full dataset evaluation

**Computed Metrics:**
- MOTA: Multiple Object Tracking Accuracy
- MOTP: Multiple Object Tracking Precision
- IDF1: ID F1 score (identity consistency)
- IDR/IDP: ID Recall/Precision
- FP/FN: False Positives/Negatives
- IDsw: ID Switches

**Usage:**
```python
from evaluate import evaluate_dataset

results = evaluate_dataset(
    "tests/data/MOT17",
    tracker_config={
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
    },
    split="train",
    verbose=True,
)

for seq_name, metrics in results.items():
    print(f"{seq_name}: MOTA={metrics['mota']:.2f}")
```

### `benchmark.py`

Full benchmark suite with multiple configurations.

**Tested Configurations:**
1. **Default**: max_age=30, min_hits=3, iou_threshold=0.3
2. **Conservative**: max_age=20, min_hits=5, iou_threshold=0.4 (fewer false positives)
3. **Aggressive**: max_age=40, min_hits=1, iou_threshold=0.2 (more detections)

**Output:**
- JSON results: `tests/results/benchmark_YYYYMMDD_HHMMSS.json`
- Human-readable report: `tests/results/benchmark_YYYYMMDD_HHMMSS_report.txt`

## Key Metrics Explained

| Metric | Meaning | Ideal |
|--------|---------|-------|
| **MOTA** | Multiple Object Tracking Accuracy | High (>70%) |
| **MOTP** | Multiple Object Tracking Precision | High (>80%) |
| **IDF1** | Identity F1 score | High (>70%) |
| **IDsw** | ID Switches per sequence | Low (<10) |
| **FP** | False Positive detections | Low |
| **FN** | False Negative detections | Low |

## Tracker Configuration Parameters

```python
Tracker(
    max_age=30,          # Frames to keep unmatched track alive
    min_hits=3,          # Frames before track is confirmed
    iou_threshold=0.3,   # IoU distance threshold for matching
)
```

**Tuning Guide:**
- **Increase `max_age`**: Tolerates longer occlusions, more ID switches
- **Decrease `min_hits`**: Earlier track confirmation, more false tracks
- **Decrease `iou_threshold`**: More permissive matching, more ID switches
- **Increase `iou_threshold`**: Stricter matching, more missed detections

## Dataset Statistics

### MOT17
- **Training**: 7 sequences, ~14K frames
- **Testing**: 7 sequences, ~20K frames
- **Focus**: Standard tracking scenarios
- **Size**: 5.5 GB

### MOT20
- **Training**: 4 sequences, ~9K frames
- **Testing**: 4 sequences, ~4.5K frames
- **Focus**: Crowded scenes (train stations, streets)
- **Size**: 5.0 GB

## Example Workflow

```bash
# 1. Install dependencies
pip install -r tests/requirements.txt

# 2. Download MOT17 and MOT20 from motchallenge.net

# 3. Run quick evaluation on MOT17
python tests/evaluate.py tests/data/MOT17 --split train

# 4. Run full benchmark with all configs
python tests/benchmark.py

# 5. Review results
cat tests/results/benchmark_*_report.txt
```

## Expected Performance

With the current SORT implementation:

**MOT17 (standard scenes):**
- MOTA: 40-50%
- MOTP: 75-80%
- IDF1: 45-55%

**MOT20 (crowded scenes):**
- MOTA: 30-40% (harder than MOT17)
- MOTP: 70-75%
- IDF1: 35-45%

(These are baseline SORT performance levels; fine-tuning configs can improve results)

## Troubleshooting

### ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

### ImportError: No module named 'motmetrics'
```bash
pip install motmetrics
```

Evaluation will run but skip MOT metric computation.

### FileNotFoundError: Dataset directory
Make sure datasets are extracted to `tests/data/MOT17` and `tests/data/MOT20`.

### Memory issues
- Evaluate on smaller splits first (`--split train`)
- Process one sequence at a time
- Reduce image resolution if needed

## Extending the Pipeline

### Add Custom Tracker Configuration

Edit `benchmark.py`:
```python
configs = [
    {
        "name": "my_config",
        "params": {"max_age": 25, "min_hits": 2, "iou_threshold": 0.35},
    },
    # ... other configs
]
```

### Add Custom Metrics

Extend `MotEvaluator._compute_mot_metrics()` in `evaluate.py` to add new metrics.

### Use Different Detectors

Modify `MotEvaluator.run_on_sequence()` to use real detections instead of ground truth:
```python
# Instead of:
detections = [Detection(bbox=box, confidence=1.0) for box in gt_boxes]

# Use your detector:
detections = my_detector.detect(img)
```

## References

- **MOTChallenge**: https://motchallenge.net/
- **SORT Paper**: https://arxiv.org/abs/1602.00763
- **motmetrics**: https://github.com/cheind/py-motmetrics
- **MOT Metrics**: https://arxiv.org/abs/1604.01802

## License

Same as Vistrack (Apache-2.0)
