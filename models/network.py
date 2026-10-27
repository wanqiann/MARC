"""Per-view autoencoders used by the clustering stage."""

from torch import nn


class ViewAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent
