#!/usr/bin/env python3
"""
Download official torchvision FP32 pretrained weights and save to T09 workspace.

Outputs:
  data/T09_ImageNet_Scale/weights/resnet18-f37072fd.pth
  data/T09_ImageNet_Scale/weights/mobilenet_v2-b0353104.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _save_and_verify(state_dict: dict, out_path: Path, required_keys: dict[str, tuple]) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_path)

    loaded = torch.load(out_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Saved object is not a state_dict: {out_path}")

    for k, shape in required_keys.items():
        if k not in loaded:
            raise RuntimeError(f"Missing key {k} in {out_path}")
        if tuple(loaded[k].shape) != tuple(shape):
            raise RuntimeError(
                f"Shape mismatch for {k} in {out_path}: got {tuple(loaded[k].shape)} expected {tuple(shape)}"
            )

    info = {
        "path": str(out_path),
        "num_tensors": len(loaded),
        "sha256": _sha256(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + verify official torchvision FP32 weights")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = repo_root / "data" / "T09_ImageNet_Scale" / "weights"
    log_path = repo_root / "data" / "T09_ImageNet_Scale" / "logs" / "weights_summary.json"
    cache_dir = repo_root / "data" / "T09_ImageNet_Scale" / "torch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    # Force official torchvision pretrained weights download.
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    resnet_out = out_dir / "resnet18-f37072fd.pth"
    mobilenet_out = out_dir / "mobilenet_v2-b0353104.pth"

    resnet_info = _save_and_verify(
        resnet.state_dict(),
        resnet_out,
        required_keys={
            "conv1.weight": (64, 3, 7, 7),
            "fc.weight": (1000, 512),
        },
    )
    mobilenet_info = _save_and_verify(
        mobilenet.state_dict(),
        mobilenet_out,
        required_keys={
            "features.0.0.weight": (32, 3, 3, 3),
            "classifier.1.weight": (1000, 1280),
        },
    )

    summary = {
        "resnet18_url": ResNet18_Weights.IMAGENET1K_V1.url,
        "mobilenet_v2_url": MobileNet_V2_Weights.IMAGENET1K_V1.url,
        "resnet18": resnet_info,
        "mobilenet_v2": mobilenet_info,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[verify] Wrote summary: {log_path}")


if __name__ == "__main__":
    main()
