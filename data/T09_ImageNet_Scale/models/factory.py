"""
Model Factory for creating Dense and Sparse models.

Supports:
- ResNet-20 (CIFAR-10)
- ResNet-18 / MobileNet-V2 / DeiT-Tiny (ImageNet)
with optional 2:4 sparsity + PTQ Int8 conversion.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet20 import resnet20
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedLinear, Int8QuantizedMultiheadAttention


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_local_weights(model: nn.Module, weights_path: str, strict: bool = True) -> None:
    if weights_path is None:
        return
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights_path not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[Factory] Warning: non-strict load. Missing={len(missing)} Unexpected={len(unexpected)}")


def _looks_like_timm_deit_state_dict(state_dict: dict) -> bool:
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    keys = list(state_dict.keys())
    # timm DeiT checkpoints typically have these keys
    return (
        any(k == "cls_token" for k in keys)
        and any(k == "pos_embed" for k in keys)
        and any(k.startswith("patch_embed.proj.") for k in keys)
        and any(k.startswith("blocks.0.") for k in keys)
    )


def _remap_timm_deit_to_torchvision_vit(state_dict: dict) -> dict:
    """
    Remap Facebook/DeiT (timm) checkpoint keys to torchvision VisionTransformer keys.

    Example mappings:
    - cls_token -> class_token
    - pos_embed -> encoder.pos_embedding
    - patch_embed.proj.* -> conv_proj.*
    - blocks.{i}.attn.qkv.* -> encoder.layers.encoder_layer_{i}.self_attention.in_proj_*
    - blocks.{i}.mlp.fc{1,2}.* -> encoder.layers.encoder_layer_{i}.mlp.{0,3}.*
    - norm.* -> encoder.ln.*
    - head.* -> heads.head.*
    """
    remapped = {}
    for k, v in state_dict.items():
        if k == "cls_token":
            remapped["class_token"] = v
            continue
        if k == "pos_embed":
            remapped["encoder.pos_embedding"] = v
            continue
        if k.startswith("patch_embed.proj."):
            remapped["conv_proj." + k[len("patch_embed.proj."):]] = v
            continue
        if k.startswith("blocks."):
            parts = k.split(".")
            # blocks.{i}.X...
            if len(parts) < 3:
                continue
            blk = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"encoder.layers.encoder_layer_{blk}."
            if rest.startswith("norm1."):
                remapped[prefix + "ln_1." + rest[len("norm1."):]] = v
                continue
            if rest.startswith("norm2."):
                remapped[prefix + "ln_2." + rest[len("norm2."):]] = v
                continue
            if rest.startswith("attn.qkv.weight"):
                remapped[prefix + "self_attention.in_proj_weight"] = v
                continue
            if rest.startswith("attn.qkv.bias"):
                remapped[prefix + "self_attention.in_proj_bias"] = v
                continue
            if rest.startswith("attn.proj.weight"):
                remapped[prefix + "self_attention.out_proj.weight"] = v
                continue
            if rest.startswith("attn.proj.bias"):
                remapped[prefix + "self_attention.out_proj.bias"] = v
                continue
            if rest.startswith("mlp.fc1."):
                remapped[prefix + "mlp.0." + rest[len("mlp.fc1."):]] = v
                continue
            if rest.startswith("mlp.fc2."):
                remapped[prefix + "mlp.3." + rest[len("mlp.fc2."):]] = v
                continue
            continue
        if k.startswith("norm."):
            remapped["encoder.ln." + k[len("norm."):]] = v
            continue
        if k.startswith("head."):
            remapped["heads.head." + k[len("head."):]] = v
            continue
        # Ignore any extra keys (e.g., distillation heads) that torchvision ViT does not use.
    return remapped


def _compute_2_4_mask_conv(weight: torch.Tensor) -> torch.Tensor | None:
    """Compute 2:4 mask along input-channel dimension (dim=1)."""
    w = weight.detach()
    out_ch, in_ch, kh, kw = w.shape
    if in_ch % 4 != 0:
        return None
    w_temp = w.abs().view(out_ch, in_ch // 4, 4, kh, kw)
    idx = torch.argsort(w_temp, dim=2)[:, :, :2, :, :]
    mask = torch.ones_like(w_temp)
    mask = mask.scatter_(dim=2, index=idx, value=0)
    return mask.view_as(w)


def _compute_2_4_mask_linear(weight: torch.Tensor) -> torch.Tensor:
    """Compute 2:4 mask along in_features (columns)."""
    w = weight.detach()
    out_features, in_features = w.shape
    if in_features % 4 != 0:
        raise ValueError("in_features must be divisible by 4 for 2:4 sparsity.")
    w_temp = w.abs().view(out_features, in_features // 4, 4)
    idx = torch.argsort(w_temp, dim=2)[:, :, :2]
    mask = torch.ones_like(w_temp)
    mask = mask.scatter_(dim=2, index=idx, value=0)
    return mask.view_as(w)


def _is_depthwise_conv(module: nn.Conv2d) -> bool:
    return module.groups == module.in_channels and module.in_channels == module.out_channels


def _is_pointwise_conv(module: nn.Conv2d) -> bool:
    return module.kernel_size == (1, 1) and module.groups == 1


def _replace_with_int8_sparse(
    module: nn.Module,
    conv_filter,
    linear_filter,
    apply_conv_mask: bool,
    apply_linear_mask: bool,
    prefix: str = ""
):
    """Recursively replace Conv2d/Linear with Int8 modules (optional masks)."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            mask = None
            if apply_conv_mask and conv_filter(child, full_name):
                mask = _compute_2_4_mask_conv(child.weight)
            new_conv = Int8QuantizedConv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                sparse_mask=mask
            )
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_conv.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
        elif isinstance(child, nn.Linear):
            mask = None
            if apply_linear_mask and linear_filter(child, full_name):
                mask = _compute_2_4_mask_linear(child.weight)
            new_fc = Int8QuantizedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                sparse_mask=mask
            )
            new_fc.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_fc.bias is not None:
                new_fc.bias.data.copy_(child.bias.data)
            setattr(module, name, new_fc)
        else:
            _replace_with_int8_sparse(
                child, conv_filter, linear_filter, apply_conv_mask, apply_linear_mask, prefix=full_name
            )


def _replace_mha_with_int8_qkv(
    module: nn.Module,
    attn_filter,
    compute_qkv_mask,
    prefix: str = ""
):
    """
    Replace `nn.MultiheadAttention` with `Int8QuantizedMultiheadAttention` so QKV (in_proj_weight)
    can carry a 2:4 sparse mask and int8 storage for CSR-index attacks.
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention):
            mask = None
            if attn_filter(child, full_name):
                mask = compute_qkv_mask(child.in_proj_weight)
            new_attn = Int8QuantizedMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=child.in_proj_bias is not None,
                add_bias_kv=False,
                add_zero_attn=child.add_zero_attn,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=child.batch_first,
                sparse_mask=mask
            )
            new_attn.in_proj_weight.data.copy_(child.in_proj_weight.data)
            if child.in_proj_bias is not None and new_attn.in_proj_bias is not None:
                new_attn.in_proj_bias.data.copy_(child.in_proj_bias.data)
            new_attn.out_proj.weight.data.copy_(child.out_proj.weight.data)
            if child.out_proj.bias is not None and new_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.data.copy_(child.out_proj.bias.data)
            setattr(module, name, new_attn)
        else:
            _replace_mha_with_int8_qkv(child, attn_filter, compute_qkv_mask, prefix=full_name)


def _calibrate_int8_modules(model: nn.Module):
    for mod in model.modules():
        if isinstance(mod, (Int8QuantizedConv2d, Int8QuantizedLinear, Int8QuantizedMultiheadAttention)):
            mod.calibrate_quantization()


def create_imagenet_resnet18_int8_sparse(device: str = "cpu", weights_path: str | None = None):
    try:
        from torchvision.models import resnet18
    except Exception as exc:
        raise ImportError("torchvision is required for resnet18") from exc
    model = resnet18(weights=None).to(device)
    if weights_path:
        _load_local_weights(model, weights_path, strict=True)
        print(f"[Factory] Loaded local ResNet-18 weights: {weights_path}")
    else:
        print("[Factory] ResNet-18 using random init (no weights_path provided).")
    model.eval()

    # Apply 2:4 sparsity to internal conv/linear only (skip conv1, fc, downsample)
    def _resnet18_conv_filter(mod: nn.Conv2d, name: str) -> bool:
        if name == "conv1" or ".downsample." in name:
            return False
        if mod.in_channels % 4 != 0:
            return False
        return True

    def _resnet18_linear_filter(mod: nn.Linear, name: str) -> bool:
        return name != "fc"

    _replace_with_int8_sparse(
        model,
        conv_filter=_resnet18_conv_filter,
        linear_filter=_resnet18_linear_filter,
        apply_conv_mask=True,
        apply_linear_mask=True
    )
    _calibrate_int8_modules(model)
    model.eval()
    return model


def create_imagenet_mobilenet_v2_int8_sparse(device: str = "cpu", weights_path: str | None = None):
    try:
        from torchvision.models import mobilenet_v2
    except Exception as exc:
        raise ImportError("torchvision is required for mobilenet_v2") from exc
    model = mobilenet_v2(weights=None).to(device)
    if weights_path:
        _load_local_weights(model, weights_path, strict=True)
        print(f"[Factory] Loaded local MobileNet-V2 weights: {weights_path}")
    else:
        print("[Factory] MobileNet-V2 using random init (no weights_path provided).")
    model.eval()

    # Only apply 2:4 sparsity to pointwise conv (1x1, groups=1) and Linear
    _replace_with_int8_sparse(
        model,
        conv_filter=lambda m, n: _is_pointwise_conv(m),
        linear_filter=lambda m, n: True,
        apply_conv_mask=True,
        apply_linear_mask=True
    )
    _calibrate_int8_modules(model)
    model.eval()
    return model


def create_imagenet_deit_tiny_int8_sparse(device: str = "cpu", weights_path: str | None = None):
    try:
        from torchvision.models.vision_transformer import VisionTransformer
    except Exception as vit_exc:
        raise ImportError("torchvision VisionTransformer is required for DeiT-Tiny") from vit_exc

    # DeiT-Tiny config
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=1000,
    ).to(device)

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights_path not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        if _looks_like_timm_deit_state_dict(state_dict):
            state_dict = _remap_timm_deit_to_torchvision_vit(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Factory] Warning: non-strict load. Missing={len(missing)} Unexpected={len(unexpected)}")
        print(f"[Factory] Loaded local DeiT-Tiny weights: {weights_path}")
    else:
        print("[Factory] DeiT-Tiny/ViT using random init (no weights_path provided).")
    model.eval()

    # Replace MHA so QKV (in_proj_weight) can carry sparse mask + int8 storage.
    _replace_mha_with_int8_qkv(
        model,
        attn_filter=lambda m, n: n.startswith("encoder.layers.encoder_layer_") and n.endswith(".self_attention"),
        compute_qkv_mask=_compute_2_4_mask_linear
    )

    # Apply 2:4 sparsity to internal MLP Linear layers only (skip conv_proj/head).
    _replace_with_int8_sparse(
        model,
        conv_filter=lambda m, n: False,
        linear_filter=lambda m, n: (".mlp.0" in n) or (".mlp.3" in n),
        apply_conv_mask=False,
        apply_linear_mask=True
    )
    _calibrate_int8_modules(model)
    model.eval()
    return model


def create_imagenet_deit_tiny_fp32(device: str = "cpu", weights_path: str | None = None) -> nn.Module:
    """
    Create a FP32 torchvision VisionTransformer (DeiT-Tiny config) and load official DeiT weights.
    Useful for ladder tests before sparsity/PTQ.
    """
    try:
        from torchvision.models.vision_transformer import VisionTransformer
    except Exception as vit_exc:
        raise ImportError("torchvision VisionTransformer is required for DeiT-Tiny") from vit_exc

    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=1000,
    ).to(device)

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights_path not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        if _looks_like_timm_deit_state_dict(state_dict):
            state_dict = _remap_timm_deit_to_torchvision_vit(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Factory] Warning: non-strict load. Missing={len(missing)} Unexpected={len(unexpected)}")
        print(f"[Factory] Loaded local DeiT-Tiny weights: {weights_path}")

    model.eval()
    return model


def create_resnet20(sparsity_type=None, pretrained_path=None, **kwargs):
    """
    Create a ResNet-20 model with specified sparsity type.

    Args:
        sparsity_type (str): Sparsity configuration
            - None: Dense model (standard FP32)
            - "2:4": 2:4 structured sparse model (N=2, M=4)
        pretrained_path (str): Path to pretrained model checkpoint
        **kwargs: Additional arguments passed to the model

    Returns:
        torch.nn.Module: ResNet-20 model

    Examples:
        >>> # Create dense model
        >>> model = create_resnet20(sparsity_type=None)
        >>>
        >>> # Create 2:4 sparse model
        >>> model = create_resnet20(sparsity_type="2:4")
        >>>
        >>> # Load from checkpoint
        >>> model = create_resnet20(sparsity_type="2:4",
        >>>                         pretrained_path="models/sparse_model.pth")
    """
    model = resnet20(pretrained=False, sparsity_type=sparsity_type, **kwargs)

    # Load pretrained weights if provided
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {pretrained_path}")

    return model


def create_model_from_config(config):
    """
    Create a model based on configuration dictionary.

    Args:
        config (dict): Configuration dictionary with keys:
            - 'model_type': 'dense' or 'sparse'
            - 'sparsity': dict with 'N' and 'M' values (for sparse models)
            - 'pretrained_path': optional path to checkpoint

    Returns:
        torch.nn.Module: Configured model
    """
    model_type = config.get('model_type', 'dense')

    if model_type == 'sparse':
        sparsity_config = config.get('sparsity', {})
        n, m = sparsity_config.get('N', 2), sparsity_config.get('M', 4)
        sparsity_type = f"{n}:{m}"
    else:
        sparsity_type = None

    pretrained_path = config.get('pretrained_path', None)

    return create_resnet20(
        sparsity_type=sparsity_type,
        pretrained_path=pretrained_path,
        num_classes=config.get('num_classes', 10),
        in_channels=config.get('input_channels', 3)
    )


if __name__ == '__main__':
    # Test model creation
    print("Testing model factory...")

    # Test dense model
    print("\n1. Creating Dense ResNet-20...")
    dense_model = create_resnet20(sparsity_type=None)
    dense_params = sum(p.numel() for p in dense_model.parameters())
    print(f"   Dense model parameters: {dense_params:,}")

    # Test sparse model
    print("\n2. Creating 2:4 Sparse ResNet-20...")
    sparse_model = create_resnet20(sparsity_type="2:4")
    sparse_params = sum(p.numel() for p in sparse_model.parameters())
    print(f"   Sparse model parameters: {sparse_params:,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    x = torch.randn(2, 3, 32, 32)

    dense_model.eval()
    with torch.no_grad():
        y_dense = dense_model(x)
    print(f"   Dense output shape: {y_dense.shape}")

    sparse_model.eval()
    with torch.no_grad():
        y_sparse = sparse_model(x)
    print(f"   Sparse output shape: {y_sparse.shape}")

    # Test mask freezing
    print("\n4. Testing mask freezing...")
    sparse_model.freeze_sparse_masks()
    print("   Masks frozen")

    sparse_model.unfreeze_sparse_masks()
    print("   Masks unfrozen")

    print("\nAll tests passed!")
