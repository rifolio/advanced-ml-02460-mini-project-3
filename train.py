import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from model import GraphVAE


def vae_loss(adj_recon, adj_target, mu, logvar, beta: float = 1.0, pos_weight: float = 1.0):
    max_nodes = adj_recon.size(1)
    idx = torch.triu_indices(max_nodes, max_nodes, offset=1, device=adj_recon.device)

    recon_upper = adj_recon[:, idx[0], idx[1]]
    target_upper = adj_target[:, idx[0], idx[1]]

    pw = torch.tensor(pos_weight, device=adj_recon.device)
    recon_loss = F.binary_cross_entropy_with_logits(
        torch.logit(recon_upper.clamp(1e-6, 1 - 1e-6)),
        target_upper,
        pos_weight=pw,
        reduction="mean",
    )

    # KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def train(
    model: GraphVAE,
    train_dataset,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    beta: float = 1.0,
    beta_warmup_epochs: int = 100,
    pos_weight: float = 1.0,
) -> GraphVAE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        # Linear KL warmup: ramp beta from 0 → target over first beta_warmup_epochs epochs.
        # Prevents posterior collapse before the encoder has learned useful representations.
        current_beta = beta * min(1.0, epoch / max(1, beta_warmup_epochs))

        model.train()
        total_loss = total_recon = total_kl = 0.0

        for batch in loader:
            batch = batch.to(device)
            adj_target = to_dense_adj(
                batch.edge_index, batch.batch, max_num_nodes=model.max_nodes
            )

            adj_recon, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(adj_recon, adj_target, mu, logvar, current_beta, pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon
            total_kl += kl

        n_batches = len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={total_loss/n_batches:.4f}  "
                f"recon={total_recon/n_batches:.4f}  "
                f"kl={total_kl/n_batches:.4f}  "
                f"beta={current_beta:.4f}"
            )

    return model
