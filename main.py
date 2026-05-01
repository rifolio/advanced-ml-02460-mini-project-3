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
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=None,
        help="Linearly ramp KL weight from 0 to --beta; default: max(1, epochs//2). "
        "Use 0 to disable (constant beta).",
    )
    parser.add_argument(
        "--deterministic-sample",
        action="store_true",
        help="Threshold decoder probs at eval (old behavior); default is Bernoulli sampling.",
    )
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
    kl_warmup = (
        args.kl_warmup_epochs
        if args.kl_warmup_epochs is not None
        else max(1, args.epochs // 2)
    )
    model = train(
        model,
        train_ds,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        kl_warmup_epochs=kl_warmup,
    )
    model.fit_node_counts(train_ds)

    print(f"\n=== Sampling {N_SAMPLES} graphs from each model ===")
    baseline_samples = [baseline.sample() for _ in range(N_SAMPLES)]
    model_samples = model.sample(
        n=N_SAMPLES, stochastic=not args.deterministic_sample
    )

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
