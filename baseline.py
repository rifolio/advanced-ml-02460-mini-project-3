import torch


class ErdosRenyiBaseline:
    def __init__(self, train_dataset):
        self.node_counts = []  # one N per training graph (for empirical sampling)
        self.link_probs = {}   # N -> link probability r[N]
        self.fit(train_dataset)

    def fit(self, train_dataset):
        edges_per_n = {}  # N -> total actual edges across graphs with N nodes
        count_per_n = {}  # N -> number of graphs with N nodes

        for data in train_dataset:
            n = data.num_nodes
            actual_edges = data.edge_index.shape[1] // 2  # each undirected edge stored twice
            count_per_n[n] = count_per_n.get(n, 0) + 1
            edges_per_n[n] = edges_per_n.get(n, 0) + actual_edges

        for n, count in count_per_n.items():
            max_edges = n * (n - 1) // 2
            total_edges = edges_per_n[n]
            self.link_probs[n] = total_edges / (count * max_edges) if max_edges > 0 else 0.0
            self.node_counts.extend([n] * count)

    def sample(self) -> torch.Tensor:
        idx = torch.multinomial(torch.ones(len(self.node_counts)), num_samples=1).item()
        n = self.node_counts[idx]
        r = self.link_probs[n]

        upper = torch.triu(torch.bernoulli(torch.full((n, n), r)), diagonal=1)
        adj = upper + upper.T
        return adj.float()
