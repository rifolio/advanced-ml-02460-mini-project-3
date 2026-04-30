import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h_graph = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.mu_head(h_graph), self.logvar_head(h_graph)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        output_size = max_nodes * (max_nodes - 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        )
        self.register_buffer("triu_idx", torch.triu_indices(max_nodes, max_nodes, offset=1))

    def forward(self, z):
        probs = self.mlp(z)                                    # (B, output_size)
        B = z.size(0)
        adj = torch.zeros(B, self.max_nodes, self.max_nodes, device=z.device)
        adj[:, self.triu_idx[0], self.triu_idx[1]] = probs
        adj = adj + adj.transpose(1, 2)                        # symmetrize
        return adj                                             # (B, max_nodes, max_nodes)


class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.encoder = GNNEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, max_nodes)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)

    def reparameterise(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n: int = 1) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        adj_probs = self.decode(z)                             # (n, max_nodes, max_nodes)
        adj = ((adj_probs + adj_probs.transpose(1, 2)) >= 1.0).float()
        idx = torch.arange(self.max_nodes, device=device)
        adj[:, idx, idx] = 0.0
        return [adj[i] for i in range(n)]
