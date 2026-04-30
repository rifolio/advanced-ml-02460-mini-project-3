import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj


def load_mutag(root: str = "data/", test_split: float = 0.2, seed: int = 42):
    dataset = TUDataset(root=root, name="MUTAG")
    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_test
    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test], generator=generator
    )
    return list(train_ds), list(test_ds)


def to_adjacency(data) -> torch.Tensor:
    return to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]


def to_adjacency_list(dataset) -> list[torch.Tensor]:
    return [to_adjacency(d) for d in dataset]
