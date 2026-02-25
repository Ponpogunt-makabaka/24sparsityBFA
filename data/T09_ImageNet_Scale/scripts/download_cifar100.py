#!/usr/bin/env python3
"""
Download and verify CIFAR-100 into data/cifar100.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torchvision.datasets as datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-100 to data/cifar100")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data" / "cifar100"
    data_root.mkdir(parents=True, exist_ok=True)

    train_set = datasets.CIFAR100(root=str(data_root), train=True, download=True)
    test_set = datasets.CIFAR100(root=str(data_root), train=False, download=True)

    extracted = data_root / "cifar-100-python"
    train_file = extracted / "train"
    test_file = extracted / "test"
    meta_file = extracted / "meta"

    if not extracted.is_dir() or not train_file.is_file() or not test_file.is_file() or not meta_file.is_file():
        raise RuntimeError(f"CIFAR-100 extraction incomplete under {extracted}")

    summary = {
        "data_root": str(data_root),
        "extracted_dir": str(extracted),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "num_classes": 100,
        "files": {
            "train": str(train_file),
            "test": str(test_file),
            "meta": str(meta_file),
        },
    }

    log_path = repo_root / "data" / "T09_ImageNet_Scale" / "logs" / "cifar100_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[verify] Wrote summary: {log_path}")


if __name__ == "__main__":
    main()
