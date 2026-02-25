#!/usr/bin/env python3
"""
Download official DeiT-Tiny pretrained weights into T09 workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.hub import download_url_to_file


DEIT_TINY_URL = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
DEIT_TINY_FILENAME = "deit_tiny_patch16_224-a1311bcf.pth"
HASH_PREFIX = "a1311bcf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official DeiT-Tiny weights")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = repo_root / "data" / "T09_ImageNet_Scale" / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / DEIT_TINY_FILENAME
    tmp_path = out_path.with_suffix(".pth.part")

    print(f"[download] {DEIT_TINY_URL}")
    download_url_to_file(
        DEIT_TINY_URL,
        str(tmp_path),
        hash_prefix=HASH_PREFIX,
        progress=True,
    )
    tmp_path.replace(out_path)

    ckpt = torch.load(out_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Downloaded DeiT checkpoint is not a dict.")

    state_dict = ckpt.get("model", ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError("No valid state_dict found in DeiT checkpoint.")

    required = ["cls_token", "pos_embed", "patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "head.weight"]
    missing = [k for k in required if k not in state_dict]
    if missing:
        raise RuntimeError(f"Downloaded checkpoint missing required keys: {missing}")

    summary = {
        "url": DEIT_TINY_URL,
        "output_path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256(out_path),
        "num_tensors": len(state_dict),
        "required_keys_present": required,
    }

    log_path = repo_root / "data" / "T09_ImageNet_Scale" / "logs" / "deit_tiny_weights_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[verify] Wrote summary: {log_path}")


if __name__ == "__main__":
    main()
