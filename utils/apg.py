import torch
from typing import List, Optional

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def normalized_guidance(
    pred_cond: torch.Tensor,   # [B, C, H, W]
    pred_uncond: torch.Tensor, # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    # pred_guided = pred_cond + (guidance_scale - 1) * normalized_update # For positive guidance
    pred_guided = pred_cond - (guidance_scale - 1) * normalized_update # For negative guidance
    return pred_guided


def normalized_compositional_guidance(
    pred_conds: List[torch.Tensor],      # List of [B, C, H, W] conditional predictions
    pred_uncond: torch.Tensor,           # [B, C, H, W] unconditional prediction
    guidance_scales: List[float],        # List of guidance scales for each condition (can be + or -)
    momentum_buffers: Optional[List[MomentumBuffer]] = None, # List of MomentumBuffers for each condition
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    positive_update = torch.zeros_like(pred_uncond)
    negative_update = torch.zeros_like(pred_uncond)
    pos_count, neg_count = 0, 0

    # Separate processing for positive and negative guidance scales
    for i, (pred_cond, guidance_scale) in enumerate(zip(pred_conds, guidance_scales)):
        # Calculate difference for each condition
        diff = pred_cond - pred_uncond
        if momentum_buffers and momentum_buffers[i] is not None:
            momentum_buffers[i].update(diff)
            diff = momentum_buffers[i].running_average

        # Apply normalization threshold if specified
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor

        # Compute parallel and orthogonal components
        diff_parallel, diff_orthogonal = project(diff, pred_cond)
        normalized_update = diff_orthogonal + eta * diff_parallel

        # Accumulate in either positive or negative update
        if guidance_scale > 0:
            positive_update += (guidance_scale - 1) * normalized_update
            pos_count += 1
        else:
            negative_update += (guidance_scale - 1) * normalized_update
            neg_count += 1

    # Normalize each set if counts are non-zero
    if pos_count > 0:
        positive_update /= pos_count
    if neg_count > 0:
        negative_update /= neg_count

    # Combine both updates equally
    compositional_update = positive_update + negative_update

    # Apply compositional update to the unconditional prediction for final guided result
    pred_guided = pred_uncond + compositional_update
    del pos_count, neg_count, diff, diff_parallel, diff_orthogonal, normalized_update, compositional_update
    del pred_conds, pred_uncond, guidance_scales, momentum_buffers, eta, norm_threshold
    return pred_guided

