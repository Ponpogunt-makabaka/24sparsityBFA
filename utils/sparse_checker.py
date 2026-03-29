"""
2:4 Sparsity Compliance Checker — GEMM K-dimension semantics.

Validates that 2:4 structured sparsity patterns match NVIDIA Ampere
sparse tensor core requirements:
  - Linear: W[out, in], K=in, groups of 4 along in_features
  - Conv2d: W[out, in, kH, kW] → GEMM view [out_g, in_g*kH*kW],
            K=in_g*kH*kW, groups of 4 along K
            (NHWC memory order: in varies fastest)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerCheckResult:
    """Per-layer 2:4 compliance check result."""
    layer_name: str
    param_shape: list
    param_type: str          # "linear", "conv2d", "other"
    groups: int              # number of groups in Conv2d (1 for standard)
    K: int                   # reduction dimension length
    K_div_4: bool            # K % 4 == 0
    eligible: bool           # eligible for 2:4 sparsity
    skip_reason: str         # reason for skipping if not eligible
    applied: bool            # was 2:4 actually applied
    total_groups: int        # number of 4-element groups
    compliant_groups: int    # groups with exactly 2 non-zero
    violation_rate: float    # fraction of non-compliant groups


def check_2by4_semantics_linear(weight: torch.Tensor) -> LayerCheckResult:
    """Check 2:4 compliance for a Linear weight [out, in]."""
    assert weight.dim() == 2, f"Expected 2D, got {weight.dim()}D"
    out_features, in_features = weight.shape
    K = in_features

    flat = weight.detach().view(out_features, -1)
    if K % 4 != 0:
        return LayerCheckResult(
            layer_name="", param_shape=list(weight.shape),
            param_type="linear", groups=1, K=K, K_div_4=False,
            eligible=False, skip_reason=f"in_features={K} not divisible by 4",
            applied=False, total_groups=0, compliant_groups=0, violation_rate=0.0,
        )

    grouped = flat.reshape(-1, 4)
    nonzero_per_group = (grouped != 0).sum(dim=1)
    compliant = int((nonzero_per_group == 2).sum().item())
    total = int(grouped.shape[0])
    violation = 1.0 - compliant / max(total, 1)

    return LayerCheckResult(
        layer_name="", param_shape=list(weight.shape),
        param_type="linear", groups=1, K=K, K_div_4=True,
        eligible=True, skip_reason="",
        applied=(violation < 1.0), total_groups=total,
        compliant_groups=compliant, violation_rate=violation,
    )


def check_2by4_semantics_conv2d(
    weight: torch.Tensor,
    conv_groups: int = 1,
) -> LayerCheckResult:
    """
    Check 2:4 compliance for Conv2d weight [out, in, kH, kW].

    Uses GEMM view per group:
      For each group g:
        Wg = weight[out_g_start:out_g_end, :, :, :]  (shape [out_g, in_g, kH, kW])
        Flat: [out_g, in_g * kH * kW]
        K = in_g * kH * kW
    Then checks groups of 4 along K in NHWC order (in varies fastest):
        permute(0, 2, 3, 1) → [out_g, kH, kW, in_g] → view(-1, 4)
    """
    assert weight.dim() == 4, f"Expected 4D, got {weight.dim()}D"
    out_ch, in_ch, kH, kW = weight.shape
    in_g = in_ch  # per-group input channels
    out_g = out_ch // conv_groups

    K = in_g * kH * kW

    if K % 4 != 0:
        return LayerCheckResult(
            layer_name="", param_shape=list(weight.shape),
            param_type="conv2d", groups=conv_groups, K=K, K_div_4=False,
            eligible=False,
            skip_reason=f"K={in_g}*{kH}*{kW}={K} not divisible by 4",
            applied=False, total_groups=0, compliant_groups=0,
            violation_rate=0.0,
        )

    # Use NHWC order: permute to [out, kH, kW, in] then flatten
    w = weight.detach()
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()
    grouped = w_nhwc.reshape(-1, 4)
    nonzero_per_group = (grouped != 0).sum(dim=1)
    compliant = int((nonzero_per_group == 2).sum().item())
    total = int(grouped.shape[0])
    violation = 1.0 - compliant / max(total, 1)

    return LayerCheckResult(
        layer_name="", param_shape=list(weight.shape),
        param_type="conv2d", groups=conv_groups, K=K, K_div_4=True,
        eligible=True, skip_reason="",
        applied=(violation < 1.0), total_groups=total,
        compliant_groups=compliant, violation_rate=violation,
    )


def check_model_sparsity(
    model: nn.Module,
    eligible_layers: Optional[Dict[str, str]] = None,
) -> List[LayerCheckResult]:
    """
    Check all parameters in model for 2:4 compliance.

    Args:
        model: The model to check.
        eligible_layers: Dict mapping param_name → reason if eligible,
                         or None to check all 2D/4D params.

    Returns:
        List of LayerCheckResult for each checked parameter.
    """
    results = []
    for name, param in model.named_parameters():
        if param.dim() == 2:
            result = check_2by4_semantics_linear(param.data)
            result.layer_name = name
        elif param.dim() == 4:
            # Find the module to get groups
            conv_groups = 1
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                try:
                    parent = model.get_submodule(parent_name.replace(".weight", ""))
                except (AttributeError, ValueError):
                    parent = None
                if parent is None:
                    # Try getting the conv module directly
                    param_name_no_weight = name.replace(".weight", "")
                    try:
                        mod = model.get_submodule(param_name_no_weight)
                        if isinstance(mod, nn.Conv2d):
                            conv_groups = mod.groups
                    except (AttributeError, ValueError):
                        pass
                elif isinstance(parent, nn.Conv2d):
                    conv_groups = parent.groups
            # Also try direct module lookup
            if conv_groups == 1 and name.endswith(".weight"):
                try:
                    mod = model.get_submodule(name[:-len(".weight")])
                    if isinstance(mod, nn.Conv2d):
                        conv_groups = mod.groups
                except (AttributeError, ValueError):
                    pass

            result = check_2by4_semantics_conv2d(param.data, conv_groups)
            result.layer_name = name
        else:
            continue

        if eligible_layers is not None:
            if name not in eligible_layers:
                result.eligible = False
                result.skip_reason = "not in eligible_layers list"

        results.append(result)

    return results


def print_check_summary(results: List[LayerCheckResult]) -> None:
    """Print a summary table of check results."""
    print(f"\n{'='*100}")
    print(f"{'Layer':<55} {'Shape':<20} {'K':>6} {'Elig':>5} {'Applied':>7} {'Viol%':>7} {'Skip Reason'}")
    print(f"{'-'*100}")
    for r in results:
        shape_str = str(r.param_shape)
        elig_str = "Y" if r.eligible else "N"
        applied_str = "Y" if r.applied else "N"
        viol_str = f"{r.violation_rate*100:.1f}%" if r.applied else "-"
        skip = r.skip_reason[:30] if r.skip_reason else ""
        print(f"{r.layer_name:<55} {shape_str:<20} {r.K:>6} {elig_str:>5} {applied_str:>7} {viol_str:>7} {skip}")
    print(f"{'='*100}")

    # Summary stats
    eligible = [r for r in results if r.eligible]
    applied = [r for r in results if r.applied]
    violated = [r for r in results if r.applied and r.violation_rate > 0]
    print(f"\nTotal params checked: {len(results)}")
    print(f"Eligible for 2:4: {len(eligible)}")
    print(f"Actually applied: {len(applied)}")
    print(f"With violations: {len(violated)}")
    if violated:
        for v in violated:
            print(f"  VIOLATION: {v.layer_name} rate={v.violation_rate*100:.2f}%")


def save_check_report(
    results: List[LayerCheckResult],
    output_path: Path,
) -> None:
    """Save check results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total_params": len(results),
        "eligible": sum(1 for r in results if r.eligible),
        "applied": sum(1 for r in results if r.applied),
        "violated": sum(1 for r in results if r.applied and r.violation_rate > 0),
        "layers": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Saved compliance report: {output_path}")
