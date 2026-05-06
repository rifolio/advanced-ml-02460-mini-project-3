import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def _adj_to_nx(adj: torch.Tensor) -> nx.Graph:
    binary = (adj >= 0.5).numpy()
    active_indices = np.where(np.any(binary, axis=1))[0]

    G = nx.Graph()
    G.add_nodes_from(range(len(active_indices)))
    for new_i, orig_i in enumerate(active_indices):
        for new_j, orig_j in enumerate(active_indices):
            if new_i < new_j and binary[orig_i, orig_j]:
                G.add_edge(new_i, new_j)
    return G


def compute_metrics(samples: list[torch.Tensor], train_graphs: list[torch.Tensor]) -> dict:
    train_hashes: set[str] = {
        nx.weisfeiler_lehman_graph_hash(_adj_to_nx(adj)) for adj in train_graphs
    }

    sample_hashes = [nx.weisfeiler_lehman_graph_hash(_adj_to_nx(adj)) for adj in samples]
    n = len(sample_hashes)
    if n == 0:
        return {"novel": 0.0, "unique": 0.0, "novel_and_unique": 0.0}

    novel_mask = [h not in train_hashes for h in sample_hashes]

    seen: set[str] = set()
    unique_mask = []
    for h in sample_hashes:
        unique_mask.append(h not in seen)
        seen.add(h)

    return {
        "novel": 100.0 * sum(novel_mask) / n,
        "unique": 100.0 * sum(unique_mask) / n,
        "novel_and_unique": 100.0 * sum(a and b for a, b in zip(novel_mask, unique_mask)) / n,
    }


def _collect_stat(graphs: list[torch.Tensor], stat: str) -> list[float]:
    values: list[float] = []
    for adj in graphs:
        G = _adj_to_nx(adj)
        if G.number_of_nodes() == 0:
            continue
        if stat == "degree":
            vals = list(dict(G.degree()).values())
        elif stat == "clustering":
            vals = list(nx.clustering(G).values())
        elif stat == "eigenvector":
            try:
                vals = list(nx.eigenvector_centrality(G, max_iter=1000).values())
            except Exception:
                try:
                    vals = list(nx.eigenvector_centrality_numpy(G).values())
                except Exception:
                    n_nodes = G.number_of_nodes()
                    vals = [1.0 / n_nodes] * n_nodes
        else:
            raise ValueError(f"Unknown stat: {stat}")
        values.extend(vals)
    return values


def plot_statistics(
    baseline_samples: list[torch.Tensor],
    model_samples: list[torch.Tensor],
    train_graphs: list[torch.Tensor],
    save_path: str = "results/statistics.png",
):
    stats = ["degree", "clustering", "eigenvector"]
    stat_labels = {
        "degree": "Node Degree",
        "clustering": "Clustering Coefficient",
        "eigenvector": "Eigenvector Centrality",
    }
    col_labels = ["Baseline", "Deep model", "Training data"]
    datasets = [baseline_samples, model_samples, train_graphs]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for row_idx, stat in enumerate(stats):
        all_vals = [_collect_stat(ds, stat) for ds in datasets]
        all_combined = [v for col_vals in all_vals for v in col_vals]

        if len(all_combined) == 0:
            g_min, g_max = 0.0, 1.0
        else:
            g_min, g_max = float(min(all_combined)), float(max(all_combined))
        if g_min == g_max:
            g_max = g_min + 1.0

        bins = np.linspace(g_min, g_max, 31)

        for col_idx, (col_vals, col_label) in enumerate(zip(all_vals, col_labels)):
            ax = axes[row_idx, col_idx]
            if col_vals:
                ax.hist(col_vals, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
            if row_idx == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(stat_labels[stat], fontsize=10)
            ax.set_xlabel("Value", fontsize=9)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved statistics plot to {save_path}")


def visualize_graphs(
    baseline_samples: list[torch.Tensor],
    model_samples: list[torch.Tensor],
    train_graphs: list[torch.Tensor],
    n: int = 6,
    save_path: str = "results/graph_samples.png",
):
    """Draw n sampled graphs per source as node-link diagrams."""
    rows = [
        ("Training data", train_graphs),
        ("GraphVAE", model_samples),
        ("Baseline (ER)", baseline_samples),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.5 * n, 2.5 * len(rows)))
    fig.subplots_adjust(hspace=0.4, wspace=0.1)

    rng = np.random.default_rng(0)
    for row_idx, (label, graphs) in enumerate(rows):
        indices = rng.choice(len(graphs), size=min(n, len(graphs)), replace=False)
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(indices):
                continue
            G = _adj_to_nx(graphs[indices[col_idx]])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=90, labelpad=4)
            title = f"{G.number_of_nodes()}n / {G.number_of_edges()}e"
            ax.set_title(title, fontsize=7, pad=2)
            if G.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
                continue
            pos = nx.spring_layout(G, seed=col_idx)
            nx.draw_networkx(
                G, pos=pos, ax=ax,
                node_size=120, node_color="steelblue",
                edge_color="gray", width=1.2,
                with_labels=False, arrows=False,
            )

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved graph visualizations to {save_path}")


def visualize_training_data(
    train_graphs: list[torch.Tensor],
    n: int = 18,
    save_path: str = "results/training_data.png",
):
    """Draw a grid of real training graphs so we know what the target looks like."""
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    fig.suptitle("Training data (MUTAG)", fontsize=13, fontweight="bold", y=1.01)
    axes = axes.flatten()

    rng = np.random.default_rng(1)
    indices = rng.choice(len(train_graphs), size=min(n, len(train_graphs)), replace=False)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(indices):
            continue
        G = _adj_to_nx(train_graphs[indices[i]])
        ax.set_title(f"{G.number_of_nodes()}n / {G.number_of_edges()}e", fontsize=7, pad=2)
        if G.number_of_nodes() == 0:
            continue
        pos = nx.spring_layout(G, seed=i)
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_size=130, node_color="coral",
            edge_color="gray", width=1.2,
            with_labels=False, arrows=False,
        )

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved training data visualization to {save_path}")
