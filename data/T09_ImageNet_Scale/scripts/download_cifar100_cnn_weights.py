#!/usr/bin/env python3
"""
Download public CIFAR-100 dense CNN weights used by T09 and verify compatibility.

Outputs:
  data/T09_ImageNet_Scale/weights/resnet18_cifar100_hf.bin
  data/T09_ImageNet_Scale/weights/mobilenet_v2_cifar100_chenyaofo.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import mobilenet_v2, resnet18


RESNET18_HF_REPO = "edadaltocg/resnet18_cifar100"
RESNET18_HF_FILENAME = "pytorch_model.bin"

MOBILENET_HUB_REPO = "chenyaofo/pytorch-cifar-models"
MOBILENET_HUB_ENTRY = "cifar100_mobilenetv2_x1_0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_resnet18_state_dict(state_dict: dict) -> None:
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"ResNet-18 CIFAR-100 state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )


def verify_mobilenet_state_dict(state_dict: dict) -> None:
    model = mobilenet_v2(weights=None, num_classes=100)
    model.features[0][0].stride = (1, 1)
    model.features[2].conv[1][0].stride = (1, 1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"MobileNetV2 CIFAR-100 state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public CIFAR-100 dense CNN weights")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = repo_root / "data" / "T09_ImageNet_Scale" / "weights"
    log_path = repo_root / "data" / "T09_ImageNet_Scale" / "logs" / "cifar100_cnn_weights_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resnet_out = out_dir / "resnet18_cifar100_hf.bin"
    mobilenet_out = out_dir / "mobilenet_v2_cifar100_chenyaofo.pth"

    resnet_cached = Path(
        hf_hub_download(
            repo_id=RESNET18_HF_REPO,
            filename=RESNET18_HF_FILENAME,
            cache_dir=str(repo_root / "data" / "T09_ImageNet_Scale" / "hf_cache"),
        )
    )
    shutil.copyfile(resnet_cached, resnet_out)
    resnet_state_dict = torch.load(resnet_out, map_location="cpu")
    verify_resnet18_state_dict(resnet_state_dict)

    mobilenet_model = torch.hub.load(
        MOBILENET_HUB_REPO,
        MOBILENET_HUB_ENTRY,
        pretrained=True,
        trust_repo=True,
    )
    mobilenet_state_dict = mobilenet_model.state_dict()
    verify_mobilenet_state_dict(mobilenet_state_dict)
    torch.save(mobilenet_state_dict, mobilenet_out)

    summary = {
        "resnet18": {
            "source": {
                "type": "huggingface_hub",
                "repo_id": RESNET18_HF_REPO,
                "filename": RESNET18_HF_FILENAME,
            },
            "output_path": str(resnet_out),
            "sha256": sha256(resnet_out),
            "size_bytes": resnet_out.stat().st_size,
            "num_tensors": len(resnet_state_dict),
            "required_shapes": {
                "conv1.weight": list(resnet_state_dict["conv1.weight"].shape),
                "fc.weight": list(resnet_state_dict["fc.weight"].shape),
            },
        },
        "mobilenet_v2": {
            "source": {
                "type": "torch_hub",
                "repo": MOBILENET_HUB_REPO,
                "entry": MOBILENET_HUB_ENTRY,
            },
            "output_path": str(mobilenet_out),
            "sha256": sha256(mobilenet_out),
            "size_bytes": mobilenet_out.stat().st_size,
            "num_tensors": len(mobilenet_state_dict),
            "required_shapes": {
                "features.0.0.weight": list(mobilenet_state_dict["features.0.0.weight"].shape),
                "classifier.1.weight": list(mobilenet_state_dict["classifier.1.weight"].shape),
            },
        },
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[verify] Wrote summary: {log_path}")


if __name__ == "__main__":
    main()
