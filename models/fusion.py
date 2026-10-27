"""Reliability-aware fusion of completed modality representations."""

import torch


def reliability_fusion(view_embeddings, view_logits, reliability, return_weights=False):
    """Weight views by learned global importance and fixed completion reliability."""
    global_weights = torch.softmax(view_logits, dim=0)
    if reliability is None:
        weights = global_weights.expand(view_embeddings[0].shape[0], -1)
    else:
        weights = global_weights.unsqueeze(0) * reliability.to(view_embeddings[0])
        weights = weights / weights.sum(dim=1, keepdim=True)
    fused = sum(z * weights[:, i:i + 1] for i, z in enumerate(view_embeddings))
    return (fused, weights) if return_weights else fused


def participant_reliability(brain_observed, endocrine_observed, rho_brain, rho_endocrine):
    """Observed views receive reliability 1; completed views use held-out R2."""
    return torch.stack([
        torch.where(brain_observed.bool(), 1.0, float(rho_brain)),
        torch.where(endocrine_observed.bool(), 1.0, float(rho_endocrine)),
    ], dim=1).float()
