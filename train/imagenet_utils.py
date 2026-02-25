#!/usr/bin/env python3
"""
ImageNet data loading helpers.

Supports custom root paths and a subset fallback if full ImageNet is not found.
"""
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _build_imagenette_target_transform(classes, wnids=None):
    if wnids:
        if set(wnids).issubset(IMAGENETTE_WNID_TO_IMAGENET_INDEX.keys()):
            mapping = [IMAGENETTE_WNID_TO_IMAGENET_INDEX[w] for w in wnids]
        else:
            return None
    else:
        if not classes:
            return None
        if not set(classes).issubset(IMAGENETTE_WNID_TO_IMAGENET_INDEX.keys()):
            return None
        mapping = [IMAGENETTE_WNID_TO_IMAGENET_INDEX[c] for c in classes]

    def _map_target(y):
        return mapping[y]

    return _map_target


def _resolve_imagenet_root(root: str, split: str, subset_root: Optional[str] = None) -> Tuple[str, bool]:
    """
    Resolve the ImageNet root path.

    Returns:
        (resolved_path, is_subset)
    """
    if root is not None:
        split_path = os.path.join(root, split)
        if os.path.isdir(split_path):
            return split_path, False
        if os.path.isdir(root) and _has_class_subdirs(root):
            return root, False

    if subset_root is not None:
        split_path = os.path.join(subset_root, split)
        if os.path.isdir(split_path):
            return split_path, True
        if os.path.isdir(subset_root) and _has_class_subdirs(subset_root):
            return subset_root, True

    raise FileNotFoundError(
        f"ImageNet not found. Checked root={root} (split={split}) and subset_root={subset_root}."
    )


def _has_class_subdirs(path: str) -> bool:
    try:
        return any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))
    except FileNotFoundError:
        return False


def get_imagenet_loader(
    root: str,
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 4,
    subset_root: Optional[str] = None,
    shuffle: bool = False,
    imagenette_root: Optional[str] = None
) -> DataLoader:
    """
    Build an ImageNet DataLoader with a root path and optional subset fallback.
    """
    use_imagenette = False
    try:
        resolved_root, is_subset = _resolve_imagenet_root(root, split, subset_root)
    except FileNotFoundError:
        resolved_root, is_subset = None, True
        use_imagenette = True

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    if use_imagenette:
        imagenette_root = imagenette_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "imagenette"
        )
        try:
            dataset = datasets.Imagenette(
                root=imagenette_root,
                split=split,
                size="full",
                download=False,
                transform=transform
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Imagenette dataset at {imagenette_root}. "
                "Download is disabled in this environment."
            ) from exc
        wnids = getattr(dataset, "wnids", None)
        target_transform = _build_imagenette_target_transform(dataset.classes, wnids=wnids)
        if target_transform is not None:
            dataset.target_transform = target_transform
            print("[ImageNet] Applied Imagenette -> ImageNet-1k label mapping.")
    else:
        dataset = datasets.ImageFolder(resolved_root, transform=transform)
        target_transform = _build_imagenette_target_transform(dataset.classes)
        if target_transform is not None:
            dataset.target_transform = target_transform
            print("[ImageNet] Applied Imagenette -> ImageNet-1k label mapping (ImageFolder).")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if use_imagenette:
        print(f"[ImageNet] Using torchvision Imagenette dataset at: {imagenette_root}")
    elif is_subset:
        print(f"[ImageNet] Using subset root: {resolved_root}")
    else:
        print(f"[ImageNet] Using full root: {resolved_root}")

    return loader
