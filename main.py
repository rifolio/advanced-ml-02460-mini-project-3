import argparse

from data import load_mutag, to_adjacency_list
from baseline import ErdosRenyiBaseline
from model import GraphVAE
from train import train
from evaluate import compute_metrics, plot_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    IN_CHANNELS = 7
    MAX_NODES = 28
    N_SAMPLES = args.samples

    train_ds, _ = load_mutag()
    train_adjs = to_adjacency_list(train_ds)

    print("=== Fitting Erdős-Rényi baseline ===")
    baseline = ErdosRenyiBaseline(train_ds)

    print("\n=== Training GraphVAE ===")
    model = GraphVAE(
        in_channels=IN_CHANNELS,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_nodes=MAX_NODES,
    )
    model = train(model, train_ds, epochs=args.epochs, lr=args.lr, beta=args.beta)

    print(f"\n=== Sampling {N_SAMPLES} graphs from each model ===")
    baseline_samples = [baseline.sample() for _ in range(N_SAMPLES)]
    model_samples = model.sample(n=N_SAMPLES)

    print("\n=== Metrics ===")
    baseline_metrics = compute_metrics(baseline_samples, train_adjs)
    model_metrics = compute_metrics(model_samples, train_adjs)

    header = f"{'':20s} {'Novel':>8s} {'Unique':>8s} {'Novel+Unique':>13s}"
    print(header)
    print("-" * len(header))
    for name, m in [("Baseline", baseline_metrics), ("GraphVAE", model_metrics)]:
        print(
            f"{name:20s} {m['novel']:7.1f}%  {m['unique']:7.1f}%  {m['novel_and_unique']:12.1f}%"
        )

    print("\n=== Plotting graph statistics ===")
    plot_statistics(baseline_samples, model_samples, train_adjs)


if __name__ == "__main__":
    main()
