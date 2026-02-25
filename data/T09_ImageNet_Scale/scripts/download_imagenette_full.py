#!/usr/bin/env python3
"""
Download and verify official Imagenette (full resolution, imagenette2).

Output layout:
  data/imagenette_full/
    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG
    imagenette2.tgz
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _progress(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        pct = min(100.0, (downloaded / total_size) * 100.0)
        print(f"\r[download] {pct:6.2f}% ({downloaded}/{total_size} bytes)", end="", flush=True)

    print(f"[download] URL: {url}")
    urllib.request.urlretrieve(url, str(tmp), reporthook=_progress)
    print()
    tmp.replace(dest)
    print(f"[download] Saved archive: {dest}")


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _extract_archive(archive: Path, extract_to: Path) -> Path:
    _safe_rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[extract] Extracting {archive} -> {extract_to}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_to)

    subdirs = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected 1 extracted root dir, got {len(subdirs)}: {subdirs}")
    return subdirs[0]


def _count_images(class_dir: Path) -> int:
    exts = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
    n = 0
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def _summarize_split(split_dir: Path) -> dict:
    if not split_dir.is_dir():
        raise RuntimeError(f"Missing split dir: {split_dir}")
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    counts = {p.name: _count_images(p) for p in class_dirs}
    if not counts:
        raise RuntimeError(f"No class folders found in {split_dir}")
    min_c = min(counts.values())
    max_c = max(counts.values())
    ratio = float(max_c) / float(max(min_c, 1))
    return {
        "num_classes": len(class_dirs),
        "counts": counts,
        "total_images": int(sum(counts.values())),
        "min_class_count": int(min_c),
        "max_class_count": int(max_c),
        "max_to_min_ratio": ratio,
    }


def _verify(train_summary: dict, val_summary: dict) -> None:
    train_classes = set(train_summary["counts"].keys())
    val_classes = set(val_summary["counts"].keys())
    if train_classes != val_classes:
        missing_train = sorted(list(val_classes - train_classes))
        missing_val = sorted(list(train_classes - val_classes))
        raise RuntimeError(
            f"Train/val class mismatch. Missing in train={missing_train}, missing in val={missing_val}"
        )

    if train_summary["num_classes"] != 10 or val_summary["num_classes"] != 10:
        raise RuntimeError(
            f"Expected 10 classes. train={train_summary['num_classes']} val={val_summary['num_classes']}"
        )

    # Official Imagenette (imagenette2) totals.
    # This confirms we are not using a proxy/subset.
    if train_summary["total_images"] != 9469 or val_summary["total_images"] != 3925:
        raise RuntimeError(
            "Unexpected dataset totals for official imagenette2. "
            f"train_total={train_summary['total_images']} val_total={val_summary['total_images']}"
        )

    if train_summary["min_class_count"] < 850 or val_summary["min_class_count"] < 350:
        raise RuntimeError(
            "Imagenette class counts too low. "
            f"train_min={train_summary['min_class_count']} val_min={val_summary['min_class_count']}"
        )

    # "Balanced" criterion for this step: class-count spread per split stays within 20%.
    if train_summary["max_to_min_ratio"] > 1.20 or val_summary["max_to_min_ratio"] > 1.20:
        raise RuntimeError(
            "Class distribution is too imbalanced for expected official Imagenette. "
            f"train_ratio={train_summary['max_to_min_ratio']:.4f} "
            f"val_ratio={val_summary['max_to_min_ratio']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + verify official Imagenette full dataset")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_root = repo_root / "data" / "imagenette_full"
    archive = out_root / "imagenette2.tgz"
    extract_tmp = out_root / "_extract_tmp"

    out_root.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        _download(IMAGENETTE_URL, archive)
    else:
        print(f"[download] Reusing existing archive: {archive}")

    extracted_root = _extract_archive(archive, extract_tmp)

    train_src = extracted_root / "train"
    val_src = extracted_root / "val"
    if not train_src.is_dir() or not val_src.is_dir():
        raise RuntimeError(f"Extracted dataset missing train/val under {extracted_root}")

    train_dst = out_root / "train"
    val_dst = out_root / "val"

    _safe_rmtree(train_dst)
    _safe_rmtree(val_dst)
    shutil.move(str(train_src), str(train_dst))
    shutil.move(str(val_src), str(val_dst))
    _safe_rmtree(extract_tmp)

    train_summary = _summarize_split(train_dst)
    val_summary = _summarize_split(val_dst)
    _verify(train_summary, val_summary)

    summary = {
        "dataset": "imagenette2_full",
        "source_url": IMAGENETTE_URL,
        "output_root": str(out_root),
        "train": train_summary,
        "val": val_summary,
    }
    summary_path = repo_root / "data" / "T09_ImageNet_Scale" / "logs" / "imagenette_full_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[verify] Imagenette verification passed.")
    print(json.dumps(summary, indent=2))
    print(f"[verify] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
