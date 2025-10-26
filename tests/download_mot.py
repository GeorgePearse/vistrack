"""
Download MOT17 and MOT20 datasets for benchmarking.

Usage:
    python tests/download_mot.py --dataset mot17 --data-dir tests/data
    python tests/download_mot.py --dataset mot20 --data-dir tests/data
"""

import os
import sys
import argparse
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar based on bytes downloaded."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str) -> None:
    """Download a file from URL with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url}...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    print(f"✓ Downloaded to {output_path}")


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract zip file."""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def extract_tar(tar_path: str, extract_to: str) -> None:
    """Extract tar file."""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def download_mot17(data_dir: str = "tests/data") -> None:
    """Download MOT17 dataset."""
    data_dir = Path(data_dir)
    mot17_dir = data_dir / "MOT17"

    if (mot17_dir / "train").exists() and (mot17_dir / "test").exists():
        print("✓ MOT17 already downloaded")
        return

    print("\n" + "=" * 60)
    print("MOT17 Dataset Download")
    print("=" * 60)
    print("Note: MOT17 is hosted on motchallenge.net")
    print("Please download manually from:")
    print("  https://motchallenge.net/data/MOT17/")
    print("\nOrganize the downloaded files as:")
    print(f"  {mot17_dir}/")
    print("  ├── train/")
    print("  │   ├── MOT17-02/")
    print("  │   ├── MOT17-04/")
    print("  │   └── ...")
    print("  └── test/")
    print("      ├── MOT17-01/")
    print("      ├── MOT17-03/")
    print("      └── ...")
    print("=" * 60)


def download_mot20(data_dir: str = "tests/data") -> None:
    """Download MOT20 dataset."""
    data_dir = Path(data_dir)
    mot20_dir = data_dir / "MOT20"

    if (mot20_dir / "train").exists() and (mot20_dir / "test").exists():
        print("✓ MOT20 already downloaded")
        return

    print("\n" + "=" * 60)
    print("MOT20 Dataset Download")
    print("=" * 60)
    print("Note: MOT20 is hosted on motchallenge.net")
    print("Please download manually from:")
    print("  https://motchallenge.net/data/MOT20/")
    print("\nOrganize the downloaded files as:")
    print(f"  {mot20_dir}/")
    print("  ├── train/")
    print("  │   ├── MOT20-01/")
    print("  │   ├── MOT20-02/")
    print("  │   └── ...")
    print("  └── test/")
    print("      ├── MOT20-03/")
    print("      ├── MOT20-05/")
    print("      └── ...")
    print("=" * 60)


def verify_dataset(dataset_dir: str) -> bool:
    """Verify that dataset structure is correct."""
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        return False

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        return False

    # Check for at least one sequence
    sequences = list(train_dir.glob("*/"))
    if not sequences:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download MOT datasets")
    parser.add_argument(
        "--dataset",
        choices=["mot17", "mot20", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--data-dir",
        default="tests/data",
        help="Directory to store datasets",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("mot17", "all"):
        download_mot17(args.data_dir)

    if args.dataset in ("mot20", "all"):
        download_mot20(args.data_dir)

    print("\n" + "=" * 60)
    print("Dataset Download Instructions")
    print("=" * 60)
    print("\n1. Visit https://motchallenge.net/")
    print("2. Download MOT17 and/or MOT20 datasets")
    print("3. Extract to the paths shown above")
    print(f"\nDefault data directory: {data_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
