"""Contrastive objectives for participant and cluster representations."""

import torch
import torch.nn.functional as F


def feature_contrastive_loss(view_z, fused_z, graph, same_cluster, temperature=0.5):
    cross_similarity = F.cosine_similarity(
        view_z[:, None, :], fused_z[None, :, :], dim=-1
    ) / temperature
    within_similarity = F.cosine_similarity(
        view_z[:, None, :], view_z[None, :, :], dim=-1
    ) / temperature

    positive_mask = graph.bool() | torch.eye(
        len(view_z), dtype=torch.bool, device=view_z.device
    )
    positive_mask &= same_cluster.bool()
    negative_mask = ~graph.bool() & ~same_cluster.bool()

    positive_term = -(cross_similarity * positive_mask).sum()
    floor = torch.log(torch.tensor(1e-45, device=view_z.device, dtype=view_z.dtype))
    negative_cross = cross_similarity.masked_fill(~negative_mask, 0)
    negative_within = within_similarity.masked_fill(~negative_mask, 0)
    negative_cross = negative_cross.masked_fill(negative_cross == 0, floor)
    negative_within = negative_within.masked_fill(negative_within == 0, floor)
    candidates = torch.cat(
        [negative_within, negative_cross],
        dim=1,
    )
    # Retained to match the objective used in the reported experiments.
    negative_term = torch.logsumexp(candidates / temperature, dim=1).sum()
    return (positive_term + negative_term) / len(view_z)


def cluster_contrastive_loss(fused_centers, view_centers, labels, n_clusters, temperature=0.5):
    fused_similarity = fused_centers @ fused_centers.T / temperature
    paired_similarity = (fused_centers * view_centers).sum(dim=1) / temperature
    scores = fused_similarity.float()
    diagonal = torch.arange(n_clusters, device=scores.device)
    scores[diagonal, diagonal] = paired_similarity

    active = torch.bincount(labels, minlength=n_clusters) > 0
    scores[:, ~active] = -10
    per_cluster = -scores.diagonal() + torch.logsumexp(scores, dim=1)
    return per_cluster[active].mean()
