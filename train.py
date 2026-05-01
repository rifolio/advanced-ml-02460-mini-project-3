import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from model import GraphVAE


def vae_loss(adj_recon, adj_target, mu, logvar, mask, beta: float = 1.0):
    max_nodes = adj_recon.size(1)

    # Only compute reconstruction loss for pairs of real nodes (upper triangle)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)          # (B, N, N)
    triu = torch.triu(
        torch.ones(max_nodes, max_nodes, dtype=torch.bool, device=adj_recon.device),
        diagonal=1,
    )
    full_mask = pair_mask & triu.unsqueeze(0)

    recon_loss = F.binary_cross_entropy(
        adj_recon[full_mask], adj_target[full_mask], reduction="mean"
    )

    # KL averaged over all real node latents in the batch
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def _beta_for_epoch(epoch: int, beta: float, kl_warmup_epochs: int) -> float:
    if kl_warmup_epochs <= 0:
        return beta
    if epoch >= kl_warmup_epochs:
        return beta
    return beta * (epoch / kl_warmup_epochs)


def train(
    model: GraphVAE,
    train_dataset,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    beta: float = 1.0,
    kl_warmup_epochs: int = 100,
) -> GraphVAE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        beta_t = _beta_for_epoch(epoch, beta, kl_warmup_epochs)
        model.train()
        total_loss = total_recon = total_kl = 0.0

        for batch in loader:
            batch = batch.to(device)
            adj_target = to_dense_adj(
                batch.edge_index, batch.batch, max_num_nodes=model.max_nodes
            )

            adj_recon, mu, logvar, mask = model(batch)
            loss, recon, kl = vae_loss(adj_recon, adj_target, mu, logvar, mask, beta_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon
            total_kl += kl

        n_batches = len(loader)
        if epoch % 10 == 0 or epoch == 1:
            beta_str = f"  β={beta_t:.3f}" if kl_warmup_epochs > 0 else ""
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={total_loss/n_batches:.4f}  "
                f"recon={total_recon/n_batches:.4f}  "
                f"kl={total_kl/n_batches:.4f}{beta_str}"
            )

    return model
