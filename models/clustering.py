"""Training and cosine assignment for MARC's two-view clustering stage."""

import torch
import torch.nn.functional as F
from sklearn.cluster import kmeans_plusplus
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fusion import reliability_fusion
from .losses import cluster_contrastive_loss, feature_contrastive_loss
from .network import ViewAutoencoder


def cosine_kmeans(features, n_clusters, seed, n_init=10, max_iter=300, tol=1e-4):
    """Return labels and centroids from cosine k-means++ runs."""
    samples = F.normalize(features.to(torch.float64), dim=1)
    variance_tolerance = samples.var(dim=0).mean() * tol
    best_inertia = float("inf")
    best_labels = None
    best_centroids = None

    for run in range(n_init):
        initial_centers, _ = kmeans_plusplus(
            samples.detach().cpu().numpy(),
            n_clusters=n_clusters,
            random_state=seed + run,
        )
        centroids = torch.as_tensor(
            initial_centers, dtype=samples.dtype, device=samples.device
        )

        labels = (samples @ centroids.T).argmax(dim=1)
        previous_labels = labels
        for _ in range(max_iter):
            previous_centroids = centroids
            totals = torch.zeros_like(centroids)
            totals.index_add_(0, labels, samples)
            counts = torch.bincount(labels, minlength=n_clusters).to(samples.dtype)
            centroids = F.normalize(totals / counts.clamp_min(1)[:, None], dim=1)
            similarity = samples @ centroids.T
            labels = similarity.argmax(dim=1)
            inertia = (1 - similarity.max(dim=1).values).sum().item()

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.clone()
                best_centroids = centroids.clone()

            shift = (1 - (previous_centroids * centroids).sum(dim=1)).sum()
            if torch.equal(labels, previous_labels) or shift <= variance_tolerance:
                break
            previous_labels = labels.clone()

    return best_labels, best_centroids


def _neighbor_graph(view, k):
    graph = torch.zeros((len(view), len(view)), dtype=torch.int32, device=view.device)
    for row in range(len(view)):
        distances = (view - view[row]).square().sum(dim=1)
        neighbors = torch.topk(distances, k, largest=False).indices[1:]
        graph[row, neighbors] = 1
    return torch.maximum(graph, graph.T)


class MARCClustering(nn.Module):
    def __init__(self, input_dims, embedding_dim, n_clusters, n_samples, seed, device):
        super().__init__()
        self.view_models = nn.ModuleList(
            [ViewAutoencoder(dim, embedding_dim) for dim in input_dims]
        )
        self.view_logits = nn.Parameter(torch.full((len(input_dims),), 1 / len(input_dims)))
        self.n_clusters = n_clusters
        self.seed = seed
        self.labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    def forward(self, views):
        outputs = [model(view) for model, view in zip(self.view_models, views)]
        return [item[0] for item in outputs], [item[1] for item in outputs]

    def fuse(self, embeddings, reliability=None, return_weights=False):
        return reliability_fusion(embeddings, self.view_logits, reliability, return_weights)

    def centroids(self, embeddings, labels):
        assignment = F.one_hot(labels, num_classes=self.n_clusters).to(embeddings.dtype).T
        assignment = F.normalize(assignment, p=1, dim=1)
        return F.normalize(assignment @ embeddings, dim=1)

    def assign(self, fused):
        labels, _ = cosine_kmeans(fused, self.n_clusters, self.seed)
        return labels


@torch.no_grad()
def _refresh_pseudo_labels(model, views, reliability, batch_size):
    rows = torch.arange(len(views[0]))
    loader = DataLoader(TensorDataset(rows), batch_size=batch_size, shuffle=False)
    encoded_views = [[] for _ in views]
    model.eval()
    for (indices,) in loader:
        indices = indices.to(views[0].device)
        batch = [view[indices] for view in views]
        _, embeddings = model(batch)
        for target, embedding in zip(encoded_views, embeddings):
            target.append(embedding)
    embeddings = [torch.cat(parts) for parts in encoded_views]
    model.labels = model.assign(model.fuse(embeddings, reliability))


def fit_clustering(model, views, reliability, batch_size=256, pretrain_epochs=200,
                   epochs=100, knn_k=10, learning_rate=1e-4, weight_decay=0.0,
                   temperature=0.5, alpha=0.01, beta=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    reconstruction_error = nn.MSELoss()
    rows = torch.arange(len(views[0]))
    batches = TensorDataset(rows)

    for _ in range(pretrain_epochs):
        model.train()
        for (indices,) in DataLoader(batches, batch_size=batch_size, shuffle=True):
            indices = indices.to(views[0].device)
            observed = [view[indices] for view in views]
            reconstructions, _ = model(observed)
            loss = sum(
                reconstruction_error(view, reconstruction)
                for view, reconstruction in zip(observed, reconstructions)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for _ in range(epochs):
        _refresh_pseudo_labels(model, views, reliability, batch_size)
        model.train()
        for (indices,) in DataLoader(batches, batch_size=batch_size, shuffle=True):
            indices = indices.to(views[0].device)
            observed = [view[indices] for view in views]
            labels = model.labels[indices]
            same_cluster = labels[:, None].eq(labels[None, :])
            reconstructions, embeddings = model(observed)
            batch_reliability = None if reliability is None else reliability[indices]
            fused = model.fuse(embeddings, batch_reliability)
            fused_centroids = model.centroids(fused, labels)

            reconstruction_terms = []
            cluster_terms = []
            feature_terms = []
            for view, reconstruction, embedding in zip(observed, reconstructions, embeddings):
                graph = _neighbor_graph(view, knn_k)
                view_centroids = model.centroids(embedding, labels)
                reconstruction_terms.append(reconstruction_error(view, reconstruction))
                cluster_terms.append(cluster_contrastive_loss(
                    fused_centroids, view_centroids, labels, model.n_clusters, temperature
                ))
                feature_terms.append(feature_contrastive_loss(
                    embedding, fused, graph, same_cluster, temperature
                ))

            loss = (
                sum(reconstruction_terms)
                + alpha * sum(cluster_terms)
                + beta * sum(feature_terms)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    _refresh_pseudo_labels(model, views, reliability, batch_size)
    return model
