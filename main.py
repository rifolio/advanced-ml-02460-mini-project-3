from data import load_mutag, to_adjacency_list
from baseline import ErdosRenyiBaseline
from model import GraphVAE
from train import train
from evaluate import compute_metrics, plot_statistics


def main():
    train_ds, test_ds = load_mutag()
    train_adjs = to_adjacency_list(train_ds)

    baseline = ErdosRenyiBaseline(train_ds)
    baseline.fit()

    model = GraphVAE(...)
    model = train(model, train_ds)

    baseline_samples = [baseline.sample() for _ in range(1000)]
    model_samples = model.sample(n=1000)

    print(compute_metrics(baseline_samples, train_adjs))
    print(compute_metrics(model_samples, train_adjs))
    plot_statistics(baseline_samples, model_samples, train_adjs)


if __name__ == "__main__":
    main()
