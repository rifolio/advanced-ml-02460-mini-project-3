Advanced machine learning

*Graphs and node embeddings*

Mikkel N. Schmidt

Technical University of Denmark,
DTU Compute, Department of Applied Mathematics and Computer Science.

2/47

# Overview

1. Module overview
2. Background
- Motivation
- Graphs
- Machine learning tasks on graphs
3. Message passing on graphs
4. Graph statistics
- Eigenvector centrality
- Clustering coefficient
- Weisfieler-Lehman test
5. Node embeddings
- Encoder-decoder perspective
- Encoder: Shallow embedding
6. Exercises

3/47

Module overview

4/47

Module 3: Graph models

1. Graphs &amp; node embeddings
Reading: Hamilton ch. 1–4
Exercise 1

2. Graph neural networks
Reading: Hamilton ch. 5–6
Exercise 2

3. Theory &amp; generative models
Reading: Hamilton ch. 7–9
Exercise 3

4. Mini project 3

5/47

# Background

6/47

# Background: Motivation

# Motivation

![img-0.jpeg](img-0.jpeg)

Humanities and social science

Friendship. Collaboration. Disease spread.

![img-1.jpeg](img-1.jpeg)

Biology

Neural cells. Protein interactions. Metabolic networks.

![img-2.jpeg](img-2.jpeg)

Chemistry

Molecular structure. Reaction networks. Material properties.

Motivation

![img-3.jpeg](img-3.jpeg)

Technical science

Communication networks. Road networks. Flight planning.

![img-4.jpeg](img-4.jpeg)

Political science

Case law. International relations. Influence analysis.

![img-5.jpeg](img-5.jpeg)

Computer science

Call graph analysis. Control flow.

8/47

9/47

# Background: Graphs

# Simple graph

Definition: A simple graph is a tuple

$$
\mathcal {G} = (\mathcal {V}, \mathcal {E})
$$

where

$\mathcal{V}$: A set of nodes, representing entities in the graph.
$\mathcal{E}$: A set of edges, representing relationships between nodes.

# Example

$$
\mathcal {V} = \{1, 2, \dots , 6 \}
$$

$$
\mathcal {E} = \{(1, 2), (1, 3), (1, 4), \dots , (7, 6) \}
$$

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c c c} 0 &amp; 1 &amp; 1 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 1 &amp; 0 &amp; 1 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 1 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 1 &amp; 1 &amp; 1 &amp; 0 &amp; 1 &amp; 1 &amp; 1 \\ 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 1 \\ 0 &amp; 0 &amp; 0 &amp; 1 &amp; 1 &amp; 0 &amp; 1 \\ 0 &amp; 0 &amp; 0 &amp; 1 &amp; 1 &amp; 1 &amp; 0 \end{array} \right]
$$

$$
\boldsymbol {D} = \operatorname {d i a g} (3, 3, 3, 6, 3, 3, 3)
$$

![img-6.jpeg](img-6.jpeg)

# Heterogenous graph

Definition: A heterogeneous graph is a tuple

$$
\mathcal {G} = (\mathcal {V}, \mathcal {E}, \mathcal {T} _ {\mathrm {V}}, \mathcal {T} _ {\mathrm {E}}, f _ {\mathrm {V}}, f _ {\mathrm {E}})
$$

where

$\mathcal{V}$: A set of nodes, representing entities in the graph.

$\mathcal{E}$: A set of edges, representing relationships between nodes.

$\mathcal{T}_{\mathrm{V}}$: A set of node types; each node $v \in \mathcal{V}$ has a specific type $t_v \in \mathcal{T}_{\mathrm{V}}$.

$\mathcal{T}_{\mathrm{E}}$: A set of edge types; each edge $e \in \mathcal{E}$ has a specific type $t_e \in \mathcal{T}_{\mathrm{E}}$.

$f_{\mathrm{V}}$: A node type mapping function, $f_{\mathrm{V}}: \mathcal{V} \to \mathcal{T}_{\mathrm{V}}$
that assigns each node $v$ to its corresponding type, $t_v = f_{\mathrm{V}}(v)$.

$f_{\mathrm{E}}$: An edge type mapping function, $f_{\mathrm{E}}: \mathcal{E} \to \mathcal{T}_{\mathrm{E}}$
that assigns each edge $e$ to its corresponding type, $t_e = f_{\mathrm{E}}(e)$.

11/47

Special graphs

|  Directed | Edges have a direction.  |
| --- | --- |
|  Undirected | Edges have no direction (sometimes represented with duplicate opposed directed edges).  |
|  Simple | No self edges and no multiple edges.  |
|  Bipartite | Nodes can be divided into to disjoint sets such that all edges are between sets.  |
|  Complete | Any two nodes are connected by an edge.  |
|  Connected | There exists a path between any two distinct nodes.  |
|  Tree | Acyclic graph where each pair of nodes are connected by exactly one path.  |

13/47

# Graph Laplacians

## Simple undirected graphs

1. Unnormalized Laplacian

$$
L = D - A
$$

2. Symmetric normalized Laplacian

$$
L_{\mathrm{sym}} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
$$

3. Random walk Laplacian

$$
L_{\mathrm{rw}} = D^{-1} L = I - D^{-1} A
$$

All non-negative eigenvalues.

Graph Laplacians

Laplacians offer a way to understand graphs through a matrix representation.

- Links a discrete graph with a continuous mathematical structure.
- Eigenvalues/vectors relate to structural properties of the graph.

Used for task such as

- Estimate node connectedness
- Graph partitioning/clustering
- Graph signal processing (signals that propagate on a graph)

15/47

# Background: Machine learning tasks on graphs

16/47

# Machine learning tasks

## Unsupervised
- Community detection: Identify groups of connected nodes.
- Link prediction: Predict formation of new edges.
- Node embedding: Learn low-dimensional representations of nodes.

## Supervised
- Node/edge classification: Predict node/edge labels
- Graph property prediction: Predict properties of the entire graph.

## Generative
- Graph generation: Generate new graphs that resemble real-world data.
- Graph completion: Predict missing nodes/edges in incomplete graph.

Task levels

Examples in the context of molecular graphs

Node level Atomic forces and charge.

Edge level Bond formation. Binding sites.

Path level Electron flow.

Subgraph level Functional groups. Reaction prediction.

Graph level Molecule toxicity, energy, stability, solubility.

![img-7.jpeg](img-7.jpeg)

18/47

Message passing on graphs

Why message passing

- There is no intrinsic notion of node order. The adjacency matrices

$$
A \quad \text{and} \quad A^{*} = P A P^{T}
$$

describe the same graph for any permutation matrix $P$.

- The way we represent and process a graph must not depend on any particular permutation.

$$
f(A) = f(P A P^{\top})
$$

19/47

# Message passing

1. Initialization: Each node has an initial state representation.
2. Messages: Each node sends a messages to its neighbors based on its current state.
3. Aggregate: Each node aggregates its received messages in combination with its previous state.
4. Update: Nodes update their state based on aggregated messages.
5. Iteration: Steps 2-4 are repeated to let information propagate through the graph.
6. Readout: The final node states are used to make node level predictions, or aggregated to make graph level predictions.

![img-8.jpeg](img-8.jpeg)

Message passing

Generalized message passing

$$
\boldsymbol {h} _ {u} ^ {(k + 1)} = \operatorname {U P D A T E} ^ {(k)} \left(\boldsymbol {h} _ {u} ^ {(k)}, \text {A G G R E G A T E} ^ {(k)} \left(\left\{\boldsymbol {h} _ {v} ^ {(k)}: v \in \mathcal {N} (u) \right\}\right)\right)
$$

Graph level readout

$$
\boldsymbol {y} = \text {R E A D O U T} \left(\left\{\boldsymbol {h} _ {1}, \dots , \boldsymbol {h} _ {N} \right\}\right)
$$

21/47

22/47

# Graph statistics

Graph statistics: Eigenvector centrality

24/47

# Eigenvector centrality

- Measures "influence" in networks: Scores nodes based on connections to other influential nodes.
- Connections matter, but not equally: Links from high-scoring nodes count more than those from low-scoring ones.
- High score = connected to many other high-scorers.
- Similar to PageRank (a website importance score based on quality backlinks)

# Eigenvector centrality definition

- Recursive definition: Centrality is a scaled sum of neighbors' centrality.

A constant that defines the scale

$$
e _ {u} = \overbrace {\frac {1}{\lambda}} ^ {\sim} \sum_ {v \in \mathcal {N} (u)} e _ {v}
$$

Sum over all neighbors of  $u$

This can be seen as "message passing" on the graph.

- In matrix notation, this is an eigenvalue problem

$$
\lambda \boldsymbol {e} = \boldsymbol {A} \boldsymbol {e}
$$

and the solution¹ is the eigenvector corresponding to the largest eigenvalue.

¹The Perron-Frobenius theorem guarantees that the largest eigenvalue is unique with an eigenvector than can be chosen to have non-negative entries: So that solution is the most interesting.

Graph statistics: Clustering coefficient

Clustering coefficient

- Measures "clumpiness" of connections around a single node.
- High score = neighbors are connected to each other.
- Calculated as ratio of triangles to possible triangles around the node.
- Values between 0 (no triangles) and 1 (all possible triangles formed).
- Density of local neighborhood subgraph.

Clustering coefficient definition

- Ratio of triangles to possible triangles around the node.

Number of edges between neighbors of $u$

$$
c_u = \frac{ \overline{ \left| (v_1, v_2) \in \mathcal{E} : v_1, v_2 \in \mathcal{N}(u) \right| } }{ \left( \begin{array} { c } d_u \\ 2 \end{array} \right) }
$$

Number of possible edges between neighbors of $u$

Not directly computable by message passing.

- In matrix notation

$$
\boldsymbol{c} = \left( \boldsymbol{D}(\boldsymbol{D} - \boldsymbol{I}) \right)^{-1} \operatorname{diag}(\boldsymbol{A}^3)
$$

28/47

# Clustering coefficient example

Clustering coefficient of node 1:

$$
c _ {1} = \frac {0}{\binom {1} {2}} = \frac {0}{0}
$$

Clustering coefficient of node 2:

$$
c _ {2} = \frac {3}{\binom {4} {2}} = \frac {3}{6} = 0.5
$$

![img-9.jpeg](img-9.jpeg)

$$
\boldsymbol {A} = \left[ \begin{array}{l l l l l} 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 1 &amp; 0 &amp; 1 &amp; 1 &amp; 1 \\ 0 &amp; 1 &amp; 0 &amp; 1 &amp; 1 \\ 0 &amp; 1 &amp; 1 &amp; 0 &amp; 1 \\ 0 &amp; 1 &amp; 1 &amp; 1 &amp; 0 \end{array} \right]
$$

$$
\boldsymbol {A} ^ {2} = \left[ \begin{array}{l l l l l} 1 &amp; 0 &amp; 1 &amp; 1 &amp; 1 \\ 0 &amp; 4 &amp; 2 &amp; 2 &amp; 2 \\ 1 &amp; 2 &amp; 3 &amp; 2 &amp; 2 \\ 1 &amp; 2 &amp; 2 &amp; 3 &amp; 2 \\ 1 &amp; 2 &amp; 2 &amp; 2 &amp; 3 \end{array} \right]
$$

$$
\boldsymbol {A} ^ {3} = \left[ \begin{array}{l l l l l} 0 &amp; 4 &amp; 2 &amp; 2 &amp; 2 \\ 4 &amp; 6 &amp; 8 &amp; 8 &amp; 8 \\ 2 &amp; 8 &amp; 6 &amp; 7 &amp; 7 \\ 2 &amp; 8 &amp; 7 &amp; 6 &amp; 7 \\ 2 &amp; 8 &amp; 7 &amp; 7 &amp; 6 \end{array} \right]
$$

$$
\boldsymbol {D} = \left[ \begin{array}{l l l l l} 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 4 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 3 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 3 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 0 &amp; 3 \end{array} \right]
$$

30/47

Graph statistics: Weisfieler-Lehman test

Weisfieler-Lehman

- Determine if two graphs are isomorphic, meaning they have the same underlying structure even if labeled differently.
- Effective but not guaranteed to always determine if two graphs are truly isomorphic.

Weisfieler-Lehman algorithm

1. Assign initial labels

$$
l _ {v} ^ {(0)} = d _ {v}
$$

2. Iteratively assign new labels by hasing the multi-set within neighborhood

$$
l _ {v} ^ {(i)} = \mathrm {H A S H} (\{\{l _ {u} ^ {(i - 1)} \forall u \in \mathcal {N} (v) \} \})
$$

3. Summarize the labels

$$
\mathrm {H A S H} \left(\left\{\left\{l _ {u} ^ {(i - 1)} \forall u \in \mathcal {V} \right\} \right\}\right)
$$

4. If summary for two graphs do not agree, they cannot be isomorphic.

32/47

Weisfieler-Lehman example

1. Label nodes by their degree

![img-10.jpeg](img-10.jpeg)

Weisfieler-Lehman example

1. Label nodes by their degree

![img-11.jpeg](img-11.jpeg)

Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels

![img-12.jpeg](img-12.jpeg)

Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels

![img-13.jpeg](img-13.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels

![img-14.jpeg](img-14.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels

![img-15.jpeg](img-15.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels

![img-16.jpeg](img-16.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels

![img-17.jpeg](img-17.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels

![img-18.jpeg](img-18.jpeg)

# Weisfieler-Lehman example

1. Label nodes by their degree
2. Update labels by forming the multi-set of neighbors' labels
3. Hash the labels
4. Summarize the labels

![img-19.jpeg](img-19.jpeg)

Weisfieler-Lehman failure case

- Cannot always detect that two graphs are isomorphic. For example, Weisfieler-Lehman cannot distinguish these two graphs.

![img-20.jpeg](img-20.jpeg)

![img-21.jpeg](img-21.jpeg)

- More advanced isomorphism test exist, but it is a hard problem.
- A general problem with message passing: Difficult to detect global structure.

35/47

# Node embeddings

36/47

# Node embeddings: Encoder-decoder perspective

# Encoder-decoder perspective

Encoder: Maps a vectex into a latent representation.

![img-22.jpeg](img-22.jpeg)

Example 1: Learned embedding (lookup table).

Example 2: Message passing graph neural network.

# Encoder-decoder perspective

Encoder: Maps a vectex into a latent representation.

![img-23.jpeg](img-23.jpeg)

Example 1: Learned embedding (lookup table).

Example 2: Message passing graph neural network.

Pairwise decoder: Maps a pair of latent variables into a node pair statistic.

$$
\mathrm {D E C}: \mathbb {R} ^ {d} \times \mathbb {R} ^ {d} \to \mathbb {S}
$$

Example 1: Predict link or non-link,  $\mathbb{S} = \{0,1\}$ .

Example 2: Predict a graph-based similarity measure,  $\mathbb{S} = \mathbb{R}_{+} = \{x\in \mathbb{R}|x\geq 0\}$

# Encoder-decoder perspective

Encoder: Maps a vectex into a latent representation.

![img-24.jpeg](img-24.jpeg)

Example 1: Learned embedding (lookup table).

Example 2: Message passing graph neural network.

Pairwise decoder: Maps a pair of latent variables into a node pair statistic.

$$
\mathrm {D E C}: \mathbb {R} ^ {d} \times \mathbb {R} ^ {d} \to \mathbb {S}
$$

Example 1: Predict link or non-link,  $\mathbb{S} = \{0,1\}$ .

Example 2: Predict a graph-based similarity measure,  $\mathbb{S} = \mathbb{R}_{+} = \{x\in \mathbb{R}|x\geq 0\}$

Loss: Measure discrepancy between observed and estimated graph statistics.

# Optimizing an encoder-decoder model

![img-25.jpeg](img-25.jpeg)

39/47

# Node embeddings: Encoder: Shallow embedding

Shallow embeddings

Each node has an embedding vector which is a learned parameter.

Embedding vector for each node

$$
\boldsymbol {Z} = \begin{bmatrix} - \boldsymbol {z} _ {1} - \\ - \boldsymbol {z} _ {2} - \\ \vdots \\ - \boldsymbol {z} _ {N} - \end{bmatrix}
$$

Equivalently, we can think of this as a single linear layer

$$
\operatorname {ENC} (u) = \boldsymbol {z} _ {u} = \boldsymbol {s} _ {u} ^ {\top} \boldsymbol {Z}
$$

where $\boldsymbol{s}_u$ is a one-hot encoding of the node index $\boldsymbol{Z}$ is a weight matrix.

40/47

41/47

# Decoders

Choice of decoder depends on which graph statistic to model.

- **Dot product**: Decodes to a real number, $(-\infty, \infty)$.

$$
\mathrm{DEC}\left(\boldsymbol{z}_u, \boldsymbol{z}_v\right) = \boldsymbol{z}_u^\top \boldsymbol{z}_v
$$

41/47

# Decoders

Choice of decoder depends on which graph statistic to model.

- **Dot product**: Decodes to a real number, $(-\infty, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \boldsymbol{z}_u^\top \boldsymbol{z}_v
$$

- **Squared distance**: Distances decode to a non-negative number, $[0, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \|\boldsymbol{z}_u - \boldsymbol{z}_v\|_2^2
$$

41/47

# Decoders

Choice of decoder depends on which graph statistic to model.

- **Dot product**: Decodes to a real number, $(-\infty, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \boldsymbol{z}_u^\top \boldsymbol{z}_v
$$

- **Squared distance**: Distances decode to a non-negative number, $[0, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \|\boldsymbol{z}_u - \boldsymbol{z}_v\|_2^2
$$

- **Sigmoid**: Decodes to a binary probability, $[0, 1]$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \sigma(\boldsymbol{z}_u^\top \boldsymbol{z}_v + b)
$$

41/47

# Decoders

Choice of decoder depends on which graph statistic to model.

- **Dot product**: Decodes to a real number, $(-\infty, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \boldsymbol{z}_u^\top \boldsymbol{z}_v
$$

- **Squared distance**: Distances decode to a non-negative number, $[0, \infty)$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \|\boldsymbol{z}_u - \boldsymbol{z}_v\|_2^2
$$

- **Sigmoid**: Decodes to a binary probability, $[0, 1]$.

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \sigma(\boldsymbol{z}_u^\top \boldsymbol{z}_v + b)
$$

- **Softmax**: Decodes to a discrete probability distribution (not symmetric).

$$
\mathrm{DEC}(\boldsymbol{z}_u, \boldsymbol{z}_v) = \frac{e^{\boldsymbol{z}_u^\top \boldsymbol{z}_v}}{\sum_{w \in \mathcal{V}} e^{\boldsymbol{z}_u^\top \boldsymbol{z}_w}}
$$

42/47

# Loss function

- Squared error
In the case when the node similarity is a real number.
$$
\mathcal{L}_{\mathrm{mse}} = \sum_{(u,v)\in \mathcal{D}}\left(S_{u,v} - z_{u}^{\top}z_{v}\right)^{2}
$$
Sum over all node pairs in training set

Loss function

- Squared error
In the case when the node similarity is a real number.
$$
\mathcal{L}_{\mathrm{mse}} = \sum_{(u,v)\in \mathcal{D}}\left(S_{u,v} - \boldsymbol{z}_{u}^{\top}\boldsymbol{z}_{v}\right)^{2}
$$
Sum over all node pairs in training set

- Binary cross-entropy
When the node similarity is binary (e.g. the adjacency matrix).
$$
\mathcal{L}_{\mathrm{bce}} = \sum_{(u,v)\in \mathcal{D}} - S_{u,v}\log \left(\sigma (\boldsymbol{z}_{u}^{\top}\boldsymbol{z}_{v} + b)\right) - (1 - S_{u,v})\log \left(1 - \sigma (\boldsymbol{z}_{u}^{\top}\boldsymbol{z}_{v} + b)\right)
$$
Logistic sigmoid

42/47

Loss function

- Squared error

In the case when the node similarity is a real number.

$$
\mathcal{L}_{\mathrm{mse}} = \sum_{(u,v)\in \mathcal{D}}\left(S_{u,v} - z_{u}^{\top}z_{v}\right)^{2}
$$

Sum over all node pairs in training set

- Binary cross-entropy

When the node similarity is binary (e.g. the adjacency matrix).

$$
\mathcal{L}_{\mathrm{bce}} = \sum_{(u,v)\in \mathcal{D}} - S_{u,v} \log \left(\sigma(z_{u}^{\top}z_{v} + b)\right) - (1 - S_{u,v}) \log \left(1 - \sigma(z_{u}^{\top}z_{v} + b)\right)
$$

Logistic sigmoid

- Random walk

$$
\mathcal{L}_{\mathrm{rw}} = \sum_{(u,v)\in \mathcal{W}} - \log \frac{e^{z_{u}^{\top}z_{v}}}{\sum_{w\in \mathcal{V}}e^{z_{u}^{\top}z_{w}}}
$$

Sum over random walks

Expensive to compute

42/47

Squared error

Minimizing squared error with a dot product decoder is a matrix factorization

$$
\mathcal{L}_{\mathrm{mse}} = \sum_{(u,v) \in \mathcal{D}} \left(S_{u,v} - \boldsymbol{z}_{u}^{\top} \boldsymbol{z}_{v}\right)^{2} = \|\boldsymbol{S} - \boldsymbol{Z} \boldsymbol{Z}^{\top}\|_{F}^{2}
$$

43/47

Binary cross entropy loss

- When predicting the adjacency matrix and using the whole graph as training data we have

$$
\mathcal{L}_{\mathrm{bce}} = \sum_{(u,v) \in \mathcal{D}} -S_{u,v} \log \left(\sigma \left(z_u^\top z_v + b\right)\right) - (1 - S_{u,v}) \log \left(1 - \sigma \left(z_u^\top z_v + b\right)\right)
$$

All edges

All non-edges²

$$
= \sum_{(u,v) \in \mathcal{E}} - \log \sigma (z_u^\top z_v + b) + \sum_{(u,v) \notin \mathcal{E}} - \log \sigma (-z_u^\top z_v - b)
$$

$$
= -\sum_{u \in \mathcal{V}} \left(\sum_{v \in \mathcal{N}_u} \log \sigma (z_u^\top z_v + b) + \sum_{v \notin \mathcal{N}_u} \log \sigma (-z_u^\top z_v - b)\right)
$$

Neighbors of $u$

Non-neighbors of $u$

**Idea:**

1. Randomly sample a set of neighbors from a (flexible) distribution in the neighborhood around $u$.
2. Sample a set of non-neighbors uniformly.

2Note that $1 - \sigma(z) = \sigma(-z)$.
44/47

Random walk embeddings

A stochastic approximation of the loss can be computed efficiently using a random walk combined with negative sampling.

$$
\mathcal {L} _ {\mathrm {b c e - r w}} = \underbrace {\sum_ {(u , v) \in \mathcal {W}} - \log \sigma (z _ {u} ^ {\top} z _ {v} + b) - \gamma \mathbb {E} _ {w \sim P (w)} \left[ \log \sigma (- z _ {u} ^ {\top} z _ {w} - b) \right]} _ {\text {N e g a t i v e s a m p l i n g}}
$$

Sum over random walks

45/47

46/47

# Exercises

# Exercises

A Node level statistics
B Random walks
C Shallow embeddings
D Programming exercise (shallow embedding)