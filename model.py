import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class NodeLevelEncoder(nn.Module):
    """GCN encoder that returns per-node mu and logvar (no global pooling)."""

    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.mu_head(h), self.logvar_head(h)  # (total_nodes, latent_dim) each


class InnerProductDecoder(nn.Module):
    """Scaled dot-product decoder with a learnable sparsity bias.

    Scaling by 1/sqrt(D) stabilises variance; bias_init should be set to
    logit(training_density) so the decoder starts at the right sparsity level
    rather than 50%.  For MUTAG (density ~0.13) logit ≈ -1.9, default -2.0.
    """

    def __init__(self, latent_dim: int, bias_init: float = -2.0):
        super().__init__()
        self.scale = latent_dim ** -0.5
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, z):
        # z: (B, N, D) → adj_probs: (B, N, N)
        return torch.sigmoid(self.scale * torch.bmm(z, z.transpose(1, 2)) + self.bias)


class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.encoder = NodeLevelEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder(latent_dim)
        self._node_counts: list[int] | None = None

    def fit_node_counts(self, train_dataset):
        self._node_counts = [data.num_nodes for data in train_dataset]

    def encode(self, data):
        return self.encoder(data.x, data.edge_index)

    def reparameterise(self, mu, logvar) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, data):
        mu, logvar = self.encode(data)           # (total_nodes, latent_dim)
        z = self.reparameterise(mu, logvar)
        z_dense, mask = to_dense_batch(z, data.batch, max_num_nodes=self.max_nodes)
        adj_recon = self.decode(z_dense)         # (B, max_nodes, max_nodes)
        return adj_recon, mu, logvar, mask

    def sample(self, n: int = 1, stochastic: bool = True) -> list[torch.Tensor]:
        if self._node_counts is None:
            raise RuntimeError("Call fit_node_counts(train_dataset) before sampling")
        device = next(self.parameters()).device

        idx = torch.multinomial(
            torch.ones(len(self._node_counts)), num_samples=n, replacement=True
        )
        node_counts = [self._node_counts[i] for i in idx.tolist()]

        samples = []
        for n_nodes in node_counts:
            z = torch.randn(n_nodes, self.latent_dim, device=device)
            adj_probs = torch.sigmoid(self.decoder.scale * z @ z.T + self.decoder.bias)
            adj_probs.fill_diagonal_(0.0)

            triu = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
            upper_probs = adj_probs[triu[0], triu[1]]

            adj = torch.zeros(n_nodes, n_nodes, device=device)
            if stochastic:
                edges = (torch.rand_like(upper_probs) < upper_probs).float()
            else:
                edges = (upper_probs >= 0.5).float()
            adj[triu[0], triu[1]] = edges
            adj = adj + adj.T
            samples.append(adj)

        return samples
