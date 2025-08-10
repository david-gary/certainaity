"""Reusable loss functions for ForenScope model training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Sørensen–Dice loss for binary segmentation masks.

    Args:
        pred:   (B, 1, H, W) or (B, H, W) float32 sigmoid probabilities in [0, 1]
        target: same shape as ``pred``, binary ground-truth mask
        eps:    smoothing term to avoid division by zero

    Returns:
        Scalar loss tensor.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def bce_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """Combined binary cross-entropy + Dice loss for PatchForensic training.

    Args:
        pred:   (B, 1, H, W) float32 sigmoid probabilities
        target: (B, H, W) or (B, 1, H, W) binary ground-truth mask
        alpha:  weight for the BCE component
        beta:   weight for the Dice component

    Returns:
        Scalar loss tensor.
    """
    bce = F.binary_cross_entropy(pred.squeeze(1), target.float(), reduction="mean")
    dice = dice_loss(pred.squeeze(1), target)
    return alpha * bce + beta * dice


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Focal loss for InpaintingDetector training (handles class imbalance).

    Args:
        pred:   (B, H, W) or (B, 1, H, W) float32 sigmoid probabilities
        target: same shape, binary ground-truth mask
        gamma:  focusing parameter (higher = more focus on hard examples)
        alpha:  balancing factor for the positive class

    Returns:
        Scalar loss tensor.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def contrastive_loss(
    emb_a: torch.Tensor,
    emb_p: torch.Tensor,
    emb_n: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet contrastive loss for SPSL Siamese training.

    Minimises distance between anchor ``emb_a`` and positive ``emb_p`` while
    pushing the anchor away from negative ``emb_n`` by at least ``margin``.

    Args:
        emb_a: (B, D) anchor embeddings (L2-normalised)
        emb_p: (B, D) positive embeddings (same class as anchor)
        emb_n: (B, D) negative embeddings (different class)
        margin: minimum desired separation between positive and negative pairs

    Returns:
        Scalar loss tensor.
    """
    dist_pos = F.pairwise_distance(emb_a, emb_p)
    dist_neg = F.pairwise_distance(emb_a, emb_n)
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()
