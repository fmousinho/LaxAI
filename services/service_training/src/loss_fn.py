import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn import TripletMarginWithDistanceLoss


def semi_hard_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Compute the Semi-Hard Triplet Loss.
    Semi-hard triplets satisfy: d(a, p) < d(a, n) < d(a, p) + margin
    
    This focuses learning on triplets that are neither too easy (d(a,n) >> d(a,p))
    nor too hard (d(a,n) < d(a,p)), which can be more effective than all triplets.

    Args:
        anchor: Tensor of shape (N, D) representing anchor embeddings
        positive: Tensor of shape (N, D) representing positive embeddings
        negative: Tensor of shape (N, D) representing negative embeddings
        margin: Margin for triplet loss
    Returns:
        Scalar tensor representing the triplet loss.
    """
    # Compute distances efficiently (reuse instead of recomputing)
    pos_dist = F.pairwise_distance(anchor, positive, p=2)  # (N,)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)  # (N,)
    
    # Semi-hard mask: d(a,p) < d(a,n) < d(a,p) + margin
    mask = (pos_dist < neg_dist) & (neg_dist < pos_dist + margin)
    
    # Handle case where no semi-hard triplets exist
    if mask.sum() == 0:
        # Fall back to all triplets when no semi-hard triplets exist
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return triplet_loss.mean()
    
    # Compute loss only for semi-hard triplets
    semi_hard_pos = pos_dist[mask]
    semi_hard_neg = neg_dist[mask]
    triplet_loss = semi_hard_pos - semi_hard_neg + margin
   
    return triplet_loss.mean()


# Standardize loss function name for imports
loss_fn = semi_hard_triplet_loss

logger.info("Loss function: Semi-hard Triplet Loss")


