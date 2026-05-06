import argparse
import os

import torch

from data import load_mutag, to_adjacency_list
from baseline import ErdosRenyiBaseline
from model import GraphVAE
from train import train
from evaluate import compute_metrics, plot_statistics, visualize_graphs, visualize_training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-warmup", type=int, default=100, help="ramp beta from 0 to --beta over this many epochs")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="BCE pos_weight to counter edge sparsity imbalance")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--save-model", default="results/model.pt", help="save trained model here")
    parser.add_argument("--load-model", default=None, help="skip training, load model from this path")
    parser.add_argument("--visualize-graphs", action="store_true", help="draw sampled graph diagrams")
    parser.add_argument("--visualize-training", action="store_true", help="draw training data graphs (no model needed)")
    parser.add_argument("--img-dir", default="results", help="directory to save all output figures")
    args = parser.parse_args()

    IN_CHANNELS = 7
    MAX_NODES = 28
    N_SAMPLES = args.samples

    train_ds, _ = load_mutag()
    train_adjs = to_adjacency_list(train_ds)

    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)

    if args.visualize_training:
        print("=== Visualizing training data ===")
        visualize_training_data(train_adjs, save_path=f"{img_dir}/training_data.png")

    print("=== Fitting Erdős-Rényi baseline ===")
    baseline = ErdosRenyiBaseline(train_ds)

    model = GraphVAE(
        in_channels=IN_CHANNELS,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_nodes=MAX_NODES,
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"=== Loading model from {args.load_model} ===")
        model.load_state_dict(torch.load(args.load_model, map_location="cpu"))
    else:
        print("\n=== Training GraphVAE ===")
        model = train(
            model, train_ds,
            epochs=args.epochs, lr=args.lr,
            beta=args.beta, beta_warmup_epochs=args.beta_warmup,
            pos_weight=args.pos_weight,
        )
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")

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
    plot_statistics(baseline_samples, model_samples, train_adjs,
                    save_path=f"{img_dir}/statistics.png")

    if args.visualize_graphs:
        print("\n=== Visualizing sample graphs ===")
        visualize_graphs(baseline_samples, model_samples, train_adjs,
                         save_path=f"{img_dir}/graph_samples.png")


if __name__ == "__main__":
    main()
