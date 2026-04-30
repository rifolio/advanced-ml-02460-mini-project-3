from model import GraphVAE


def vae_loss(adj_recon, adj_target, mu, logvar) -> float:
    ...


def train(model: GraphVAE, train_dataset, epochs: int = 100, lr: float = 1e-3):
    ...
