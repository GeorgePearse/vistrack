"""
Comprehensive benchmark runner for MOT17 and MOT20 datasets.

Runs tracker with different configurations and produces comparison reports.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import evaluate_dataset


def run_benchmark_suite(
    mot17_dir: str = None,
    mot20_dir: str = None,
    output_dir: str = "tests/results",
) -> Dict:
    """Run full benchmark suite on MOT17 and MOT20.

    Parameters
    ----------
    mot17_dir : str, optional
        Path to MOT17 dataset
    mot20_dir : str, optional
        Path to MOT20 dataset
    output_dir : str
        Where to save results

    Returns
    -------
    Dict
        All benchmark results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "datasets": {},
    }

    # Tracker configurations to test
    configs = [
        {
            "name": "default",
            "params": {"max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
        },
        {
            "name": "conservative",
            "params": {"max_age": 20, "min_hits": 5, "iou_threshold": 0.4},
        },
        {
            "name": "aggressive",
            "params": {"max_age": 40, "min_hits": 1, "iou_threshold": 0.2},
        },
    ]

    # Evaluate on MOT17
    if mot17_dir and Path(mot17_dir).exists():
        print("\n" + "=" * 70)
        print("MOT17 BENCHMARK")
        print("=" * 70)
        results["datasets"]["MOT17"] = {}

        for config in configs:
            print(f"\nConfiguration: {config['name']}")
            print(f"Parameters: {config['params']}")
            config_results = evaluate_dataset(
                mot17_dir,
                tracker_config=config["params"],
                split="train",
                verbose=False,
            )
            results["datasets"]["MOT17"][config["name"]] = config_results

    # Evaluate on MOT20
    if mot20_dir and Path(mot20_dir).exists():
        print("\n" + "=" * 70)
        print("MOT20 BENCHMARK")
        print("=" * 70)
        results["datasets"]["MOT20"] = {}

        for config in configs:
            print(f"\nConfiguration: {config['name']}")
            print(f"Parameters: {config['params']}")
            config_results = evaluate_dataset(
                mot20_dir,
                tracker_config=config["params"],
                split="train",
                verbose=False,
            )
            results["datasets"]["MOT20"][config["name"]] = config_results

    # Save results
    results_file = output_dir / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_file}")

    # Generate report
    report_file = output_dir / f"benchmark_{timestamp}_report.txt"
    _generate_report(results, report_file)
    print(f"✓ Report saved to {report_file}")

    return results


def _generate_report(results: Dict, output_file: Path) -> None:
    """Generate human-readable benchmark report.

    Parameters
    ----------
    results : Dict
        Benchmark results
    output_file : Path
        Where to save report
    """
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("VISTRACK BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {results['timestamp']}\n\n")

        for dataset_name, dataset_results in results.get("datasets", {}).items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DATASET: {dataset_name}\n")
            f.write("=" * 80 + "\n\n")

            # Aggregate metrics across sequences for each config
            for config_name, config_results in dataset_results.items():
                f.write(f"\nConfiguration: {config_name}\n")
                f.write("-" * 80 + "\n")

                # Aggregate statistics
                metrics_keys = set()
                for seq_metrics in config_results.values():
                    metrics_keys.update(seq_metrics.keys())

                # Calculate aggregates
                aggregates = {}
                for key in metrics_keys:
                    values = [seq_metrics.get(key) for seq_metrics in config_results.values()]
                    values = [v for v in values if v is not None and isinstance(v, (int, float))]

                    if values:
                        aggregates[key] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                        }

                # Print summary
                f.write("Summary Statistics:\n")
                for key, stats in sorted(aggregates.items()):
                    if "mean" in stats:
                        f.write(f"  {key:20s}: {stats['mean']:8.2f} (±{stats['std']:.2f}) "
                                f"[{stats['min']:.2f}, {stats['max']:.2f}]\n")

                # Per-sequence results
                f.write("\nPer-Sequence Results:\n")
                f.write(f"{'Sequence':<20} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8} "
                        f"{'Dets':>8} {'FP':>8} {'FN':>8} {'Sw':>8}\n")
                f.write("-" * 80 + "\n")

                for seq_name, seq_metrics in sorted(config_results.items()):
                    mota = seq_metrics.get("mota", 0.0)
                    motp = seq_metrics.get("motp", 0.0)
                    idf1 = seq_metrics.get("idf1", 0.0)
                    dets = seq_metrics.get("num_detections", 0)
                    fp = seq_metrics.get("num_false_positives", 0)
                    fn = seq_metrics.get("num_misses", 0)
                    sw = seq_metrics.get("num_switches", 0)

                    f.write(f"{seq_name:<20} {mota:8.2f} {motp:8.2f} {idf1:8.2f} "
                            f"{dets:8d} {fp:8d} {fn:8d} {sw:8d}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive MOT benchmark")
    parser.add_argument("--mot17-dir", help="Path to MOT17 dataset")
    parser.add_argument("--mot20-dir", help="Path to MOT20 dataset")
    parser.add_argument("--data-dir", default="tests/data", help="Base data directory")
    parser.add_argument("--output-dir", default="tests/results", help="Output directory")

    args = parser.parse_args()

    # Auto-detect datasets if using base data dir
    mot17_dir = args.mot17_dir or f"{args.data_dir}/MOT17"
    mot20_dir = args.mot20_dir or f"{args.data_dir}/MOT20"

    results = run_benchmark_suite(
        mot17_dir=mot17_dir if Path(mot17_dir).exists() else None,
        mot20_dir=mot20_dir if Path(mot20_dir).exists() else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
