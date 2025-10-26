# Testing & Benchmarking Guide

## Quick Start

### 1. Install Test Dependencies

```bash
pip install -r tests/requirements.txt
```

Or individually:
```bash
pip install opencv-python tqdm motmetrics numpy
```

### 2. Download MOT Datasets

**MOT17**: https://motchallenge.net/data/MOT17/
**MOT20**: https://motchallenge.net/data/MOT20/

Extract to:
```
tests/data/MOT17/train/  (MOT17-02, MOT17-04, ...)
tests/data/MOT17/test/   (MOT17-01, MOT17-03, ...)
tests/data/MOT20/train/  (MOT20-01, MOT20-02, ...)
tests/data/MOT20/test/   (MOT20-03, MOT20-05, ...)
```

### 3. Run Benchmarks

**Full suite** (MOT17 + MOT20, multiple configs):
```bash
python tests/benchmark.py
```

**Single dataset, single config**:
```bash
python tests/evaluate.py tests/data/MOT17 --split train
python tests/evaluate.py tests/data/MOT20 --split train
```

**Custom tracker config**:
```bash
python tests/evaluate.py tests/data/MOT17 --split train \
    --max-age 40 --min-hits 1 --iou-threshold 0.2
```

## Results

Results are saved to `tests/results/`:
- `benchmark_YYYYMMDD_HHMMSS.json` - Machine-readable results
- `benchmark_YYYYMMDD_HHMMSS_report.txt` - Human-readable report

### Example Report

```
DATASET: MOT17

Configuration: default
Summary Statistics:
  mota                 :   45.32 (±12.45) [20.15, 65.80]
  motp                 :   77.89 (±3.21) [72.10, 83.50]
  ...

Per-Sequence Results:
Sequence             MOTA     MOTP     IDF1      Dets        FP        FN        Sw
MOT17-02           45.12    78.34    48.23     7284       892       1245        18
MOT17-04           52.34    77.12    55.67     9143       673       945         12
...
```

## Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| MOTA | Multiple Object Tracking Accuracy | >70% |
| MOTP | Multiple Object Tracking Precision | >80% |
| IDF1 | Identity F1 (ID consistency) | >70% |
| FP | False Positives | Low |
| FN | False Negatives | Low |
| IDsw | ID Switches | <10 per seq |

## Tracker Parameters

```python
Tracker(
    max_age=30,          # Keep track alive for 30 frames without match
    min_hits=3,          # Confirm track after 3 matches
    iou_threshold=0.3,   # IoU distance threshold for matching
)
```

**Quick Tuning:**
- **More FP** → increase `iou_threshold` or `min_hits`
- **More FN** → decrease `iou_threshold` or `max_age`
- **More ID switches** → decrease `iou_threshold`
- **Fewer confirmed tracks** → increase `min_hits`

## Troubleshooting

### ModuleNotFoundError for test modules

Add `tests/` to Python path:
```bash
PYTHONPATH=tests python tests/benchmark.py
```

Or edit scripts:
```python
import sys
sys.path.insert(0, 'tests')
```

### motmetrics not available

Tests will still run but skip MOT metric computation. Install:
```bash
pip install motmetrics
```

### Out of memory

Evaluate smaller splits:
```bash
python tests/evaluate.py tests/data/MOT17 --split train
```

Process sequences individually.

## Full Testing Workflow

```bash
# 1. Setup
pip install -r tests/requirements.txt
# Download MOT17/MOT20 from motchallenge.net

# 2. Test imports
python -c "from tests import *; print('OK')"

# 3. Quick evaluation on MOT17
python tests/evaluate.py tests/data/MOT17 --split train --verbose

# 4. Run full benchmark
python tests/benchmark.py

# 5. Review results
cat tests/results/benchmark_*_report.txt

# 6. Custom configuration test
python tests/evaluate.py tests/data/MOT17 \
    --max-age 25 --min-hits 2 --iou-threshold 0.35 --split train
```

## Development

### Adding New Tracker Config

Edit `tests/benchmark.py`:
```python
configs = [
    {
        "name": "my_config",
        "params": {"max_age": 35, "min_hits": 2, "iou_threshold": 0.32},
    },
]
```

### Using Real Detections

Modify `tests/evaluate.py` in `MotEvaluator.run_on_sequence()`:
```python
# Replace ground truth detection creation with:
detections = my_detector.detect(img)  # Your detector
detections = [Detection(bbox=d['bbox'], confidence=d['conf'])
              for d in detections]
```

### Custom Metrics

Extend `MotEvaluator._compute_mot_metrics()` to add metrics beyond MOT standard set.

## References

- **MOTChallenge**: https://motchallenge.net/
- **SORT Paper**: https://arxiv.org/abs/1602.00763
- **MOT Metrics**: https://arxiv.org/abs/1604.01802
- **py-motmetrics**: https://github.com/cheind/py-motmetrics

## Expected Results

With the default SORT implementation:

**MOT17** (7 sequences, standard scenes):
- MOTA: 40-50%
- MOTP: 75-80%
- IDF1: 45-55%

**MOT20** (4 sequences, crowded scenes):
- MOTA: 30-40% (harder)
- MOTP: 70-75%
- IDF1: 35-45%

Results vary based on tracker configuration and detection quality.

---

For detailed module documentation, see `tests/README.md`.
