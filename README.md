# Mini-project 3 — Graph Generative Models (AML 02460)

Graph generation on the MUTAG dataset. Two models: Erdős-Rényi baseline and a VAE with a GNN encoder.

## Structure

```
data.py        # MUTAG loader, train/test split, adjacency matrix conversion
baseline.py    # Erdős-Rényi generative model
model.py       # GraphVAE: GNN encoder + MLP decoder, graph-level latent
train.py       # VAE training loop and ELBO loss
evaluate.py    # Novel/unique metrics and graph statistics histograms
main.py        # Entry point — runs the full pipeline
data/          # Auto-downloaded by PyTorch Geometric (gitignored)
```

## Setup

```bash
uv sync
```

## Development Order

Build and validate each component before moving to the next. This order matters — the baseline and evaluation pipeline can be tested together before the VAE exists.

### 1. Baseline (`baseline.py`)

Implement `ErdosRenyiBaseline`:
- On `fit()`, compute the empirical distribution of node counts from the training graphs
- For each node count N, compute the link probability r = edges / (N*(N-1)/2) across all training graphs with N nodes
- On `sample()`, draw N from the empirical distribution, then sample a symmetric adjacency matrix where each edge is Bernoulli(r)

Test: sample a few graphs and inspect their sizes and densities manually.

### 2. Evaluation pipeline (`evaluate.py`)

Implement `compute_metrics` and `plot_statistics` before the VAE, so you can validate both against the baseline first.

**Metrics:**
- Convert adjacency tensors to `networkx` graphs
- Use `networkx.weisfeiler_lehman_graph_hash` to fingerprint each graph
- Novel: hash not in training set fingerprints
- Unique: no duplicate hashes among the 1000 samples
- Novel+Unique: both conditions

**Histograms:**
- Compute per-graph distributions of node degree, clustering coefficient, and eigenvector centrality using networkx
- Plot a 3x3 grid: rows = metric, columns = (baseline, model, training data)
- Use shared bin edges per row so the distributions are visually comparable

Test: run baseline -> `compute_metrics` -> `plot_statistics` end-to-end before touching the VAE.

### 3. VAE model (`model.py`)

**GNNEncoder:**
- Use 2-3 `GCNConv` (or `GATConv`) layers with ReLU activations
- Global mean pool over node embeddings to get a single graph-level vector
- Two linear heads: one for `mu`, one for `logvar`

**MLPDecoder:**
- Input: latent vector z of size `latent_dim`
- Output: flattened upper triangle of the adjacency matrix, length N*(N-1)/2
- Apply sigmoid so outputs are in [0,1] (treated as edge probabilities)
- Reconstruct the full symmetric matrix from the upper triangle

**GraphVAE:**
- `reparameterise`: z = mu + eps * exp(0.5 * logvar), eps ~ N(0,I)
- `sample(n)`: draw z ~ N(0,I), decode, threshold at 0.5

Note: all graphs must be zero-padded to `max_nodes` before encoding/decoding, since the decoder output size is fixed.

### 4. Training loop (`train.py`)

**ELBO loss:**
- Reconstruction: BCE between predicted edge probabilities and true upper-triangle entries
- KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
- Total: reconstruction + beta * KL (start with beta=1, tune if posterior collapse occurs)

**Loop:**
- Use `DataLoader` from PyTorch Geometric
- Adam optimiser, lr=1e-3, ~100-200 epochs
- Log loss each epoch; if KL term collapses to zero early, increase beta

### 5. Tie together (`main.py`)

Fill in the `GraphVAE(...)` constructor arguments once model hyperparameters are decided. Sample 1000 graphs from each model, print the metrics table, and display the histograms.

## Evaluation Targets (from project spec)

| | Novel | Unique | Novel+Unique |
|---|---|---|---|
| Baseline | ? | ? | ? |
| GraphVAE | ? | ? | ? |

## Dependencies

Managed with `uv`. See `pyproject.toml`.
