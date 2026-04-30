import torch
import torch.nn as nn


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        ...

    def forward(self, x, edge_index, batch):
        ...


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super().__init__()
        ...

    def forward(self, z):
        ...


class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        ...

    def encode(self, data):
        ...

    def decode(self, z) -> torch.Tensor:
        ...

    def reparameterise(self, mu, logvar) -> torch.Tensor:
        ...

    def forward(self, data):
        ...

    def sample(self, n: int = 1) -> list[torch.Tensor]:
        ...
