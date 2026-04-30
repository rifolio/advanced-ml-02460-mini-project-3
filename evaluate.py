import torch


def compute_metrics(samples: list[torch.Tensor], train_graphs: list[torch.Tensor]) -> dict:
    ...


def plot_statistics(
    baseline_samples: list[torch.Tensor],
    model_samples: list[torch.Tensor],
    train_graphs: list[torch.Tensor],
):
    ...
