1/42

Advanced machine learning

Graph neural networks

Mikkel N. Schmidt

Technical University of Denmark,
DTU Compute, Department of Applied Mathematics and Computer Science.

# Overview

1. Module overview
2. Results from last week's competition
3. Message passing on graphs
- Neighborhood aggregation
- Update methods
- Graph pooling
- Machine learning tasks on graphs
4. Geometric embedding
5. Pytorch implementation
6. Exercises

2/42

3/42

Module overview

4/42

# Module 3: Graph models

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

5/42

Results from last week's competition

6/42

# Last week's competition

![img-0.jpeg](img-0.jpeg)

7/42

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

8/42

Message passing

1. Initialization: Each node has an initial state representation.
2. Messages: Each node sends a message to its neighbors based on its current state.
3. Aggregate: Each node aggregates its received messages in combination with its previous state.
4. Update: Nodes update their state based on aggregated messages.
5. Iteration: Steps 2-4 are repeated to let information propagate through the graph.
6. Readout: The final node states are used to make node level predictions, or aggregated to make graph level predictions.

![img-1.jpeg](img-1.jpeg)

Message passing

- Initialization: Node embeddings are initialized based on an embedding of node features (if available).

$$
\boldsymbol {h} _ {u} ^ {(0)} = \operatorname {E M B E D} \left(\boldsymbol {x} _ {u}\right)
$$

- Message passing: Node embeddings are updated based on aggregated information from neighbors and previous embedding.

$$
\boldsymbol {m} _ {\mathcal {N} (u)} ^ {(k)} = \text {A G G R E G A T E} ^ {(k)} \left(\left\{\boldsymbol {h} _ {v} ^ {(k)}: v \in \mathcal {N} (u) \right\}\right)
$$

$$
\boldsymbol {h} _ {u} ^ {(k + 1)} = \text {U P D A T E} ^ {(k)} \left(\boldsymbol {h} _ {u} ^ {(k)}, \boldsymbol {m} _ {\mathcal {N} (u)} ^ {(k)}\right)
$$

Weight sharing or separate functions per layer.

- Graph level readout: If a graph-level prediction is required, node features are aggregated in a readout.

$$
\boldsymbol {y} = \text {R E A D O U T} \left(\left\{\boldsymbol {h} _ {1}, \dots , \boldsymbol {h} _ {| \mathcal {V} |} \right\}\right)
$$

# Message passing neural network

Example of a 2-layer GNN with graph-level readout on a graph with 3 nodes.

![img-2.jpeg](img-2.jpeg)

The basic GNN

An example of a very basic graph neural network:

$$
\underbrace {h _ {u} ^ {(k)} = \sigma \left(W _ {\mathrm {s e l f}} ^ {(k)} h _ {u} ^ {(k - 1)} + W _ {\mathrm {n e i g h .}} ^ {(k)} \sum_ {v \in \mathcal {N} (u)} h _ {v} ^ {(k - 1)} + b ^ {(k)}\right)}
$$

AGGREGATE

13/42

Message passing on graphs: Neighborhood aggregation

14/42

# Aggregation

- Aggregation function should be permutation invariant.

$$
f(x_1, x_2, \ldots, x_N) = f(x_{\pi_1}, x_{\pi_2}, \ldots, x_{\pi_N}) \quad \text{for any permutation } \pi
$$

Possibilities include sum, product, mean, max, etc.

- Normalization by node degree

$$
m_{\mathcal{N}(u)} = \sum_{v \in \mathcal{N}(u)} \frac{h_v}{|\mathcal{N}(u)|}
$$

$$
m_{\mathcal{N}(u)} = \sum_{v \in \mathcal{N}(u)} \frac{h_v}{\sqrt{|\mathcal{N}(u)\mathcal{N}(v)|}}
$$

- Works better in some applications (e.g. high diversity in node degree)
- Can lead to loss of information.

Aggregation

- Set pooling

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \mathrm {M L P} _ {\theta} \left(\sum_ {v \in \mathcal {N} (u)} \mathrm {M L P} _ {\phi} (\boldsymbol {h} _ {v})\right)
$$

Universal function approximator.

15/42

Aggregation

- Set pooling

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \mathrm {M L P} _ {\theta} \left(\sum_ {v \in \mathcal {N} (u)} \mathrm {M L P} _ {\phi} (\boldsymbol {h} _ {v})\right)
$$

Universal function approximator.

- Janossy pooling (randomized order)

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \mathrm {M L P} _ {\theta} \left(\frac {1}{| \Pi |} \sum_ {\pi \in \Pi} \rho_ {\phi} \left(\overbrace {[ h _ {\pi_ {1}} , h _ {\pi_ {2}} , \ldots , h _ {\pi_ {| \mathcal {N} (u) |}} ]}\right)\right)
$$

Sum over a set of permutations

Average over many permutations of a permutation-sensitive function.

15/42

Aggregation

- Set pooling

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \mathrm {M L P} _ {\theta} \left(\sum_ {v \in \mathcal {N} (u)} \mathrm {M L P} _ {\phi} (\boldsymbol {h} _ {v})\right)
$$

Universal function approximator.

- Janossy pooling (randomized order)

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \mathrm {M L P} _ {\theta} \left(\frac {1}{| \Pi |} \sum_ {\pi \in \Pi} \rho_ {\phi} \left(\overbrace {[ \boldsymbol {h} _ {\pi_ {1}} , \boldsymbol {h} _ {\pi_ {2}} , \dots , \boldsymbol {h} _ {\pi_ {| \mathcal {N} (u) |}} ]}\right)\right)
$$

Sum over a set of permutations

Average over many permutations of a permutation-sensitive function.

- Attention

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \sum_ {v \in \mathcal {N} (u)} \alpha_ {u, v} \boldsymbol {h} _ {v}
$$

$$
\alpha_ {u, v} = \underset {v \in \mathcal {N} (u)} {\operatorname {S o f t m a x}} \left(\boldsymbol {h} _ {u} ^ {\top} \boldsymbol {W h} _ {v}\right)
$$

Example here is bilinear attention—many other possibilities.

15/42

16/42

# Transformer-style attention

- Key, query and value

$$
\boldsymbol {k} _ {u} = \boldsymbol {W} _ {\text {key}} \boldsymbol {h} _ {u} \quad \boldsymbol {q} _ {u} = \boldsymbol {W} _ {\text {query}} \boldsymbol {h} _ {u} \quad \boldsymbol {v} _ {u} = \boldsymbol {W} _ {\text {value}} \boldsymbol {h} _ {u}
$$

- Scaled dot-product attention

$$
\boldsymbol {m} _ {\mathcal {N} (u)} = \sum_ {v \in \mathcal {N} (u)} \alpha_ {u, v} \boldsymbol {v} _ {v}
$$

$$
\alpha_ {u, v} = \operatorname {softmax} _ {v \in \mathcal {N} (u)} \left(\frac {\boldsymbol {k} _ {u} ^ {\top} \boldsymbol {q} _ {v}}{\sqrt {d}}\right)
$$

Dimension of key and query vectors

- Multi-head attention: Concatenate multiple of these mechanisms.

Message passing on graphs: Update methods

Self-loops

- Self-loops are a simple way to include the node itself in the neighborhood aggregation

$$
\mathcal{N}(u) \cup \{u\}
$$

I.e., we consider the node $u$ to be neighbor to itself.

- This way we can skip the update.

18/42

# Over-smoothing

- Node representation can become very similar with increasing depth.
- *Intuitive reason:* Information from neighbors dominate.
- Methods for reducing over-smoothing¹
- Normalizing node features
- Regularizing the training procedure
- More advanced updates, that retain node information.

¹See e.g. Rusch et al. A survey on oversmoothing in graph neural networks.

Concatenate, skip-connections, and gating

- Concatenation

$$
\mathrm {U P D A T E} _ {\text {c o n c a t .}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \left[ \mathrm {U P D A T E} _ {\text {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right), \boldsymbol {h} _ {u} \right]
$$

20/42

Concatenate, skip-connections, and gating

- Concatenation
$$
\mathrm {U P D A T E} _ {\text {c o n c a t .}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \left[ \mathrm {U P D A T E} _ {\text {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right), \boldsymbol {h} _ {u} \right]
$$

- Skip-connections (residual updates)
$$
\mathrm {U P D A T E} _ {\mathrm {s k i p}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \boldsymbol {h} _ {u} + \mathrm {U P D A T E} _ {\mathrm {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right)
$$

20/42

Concatenate, skip-connections, and gating

- Concatenation
$$
\mathrm {U P D A T E} _ {\text {c o n c a t .}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \left[ \mathrm {U P D A T E} _ {\text {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right), \boldsymbol {h} _ {u} \right]
$$

- Skip-connections (residual updates)
$$
\mathrm {U P D A T E} _ {\mathrm {s k i p}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \boldsymbol {h} _ {u} + \mathrm {U P D A T E} _ {\mathrm {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right)
$$

- Gating (linear interpolation)
$$
\mathrm {U P D A T E} _ {\text {i n t e r p .}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right) = \boldsymbol {\alpha} \odot \boldsymbol {h} _ {u} + (1 - \boldsymbol {\alpha}) \odot \mathrm {U P D A T E} _ {\text {b a s e}} \left(\boldsymbol {h} _ {u}, \boldsymbol {m} _ {\mathcal {N} (u)}\right)
$$
Here $\alpha$ is a gating vector which can be learned in many ways.

20/42

# Gated recurrent unit (GRU)

![img-3.jpeg](img-3.jpeg)

# Gated recurrent unit (GRU)

![img-4.jpeg](img-4.jpeg)

Reset:  $\pmb{r} = \sigma (\pmb{W}_{mr}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hr}\pmb{h}_u^{(k)} + \pmb{b}_r)$

# Gated recurrent unit (GRU)

![img-5.jpeg](img-5.jpeg)

Reset:  $r = \sigma (W_{mr}m_{\mathcal{N}(u)}^{(k)} + W_{hr}h_u^{(k)} + b_r)$

Update:  $z = \sigma (W_{mz}m_{\mathcal{N}(u)}^{(k)} + W_{hz}h_u^{(k)} + b_z)$

# Gated recurrent unit (GRU)

![img-6.jpeg](img-6.jpeg)

Reset:  $\pmb{r} = \sigma (\pmb{W}_{mr}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hr}\pmb{h}_{u}^{(k)} + \pmb{b}_{r})$

Update:  $\pmb{z} = \sigma (\pmb{W}_{mz}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hz}\pmb{h}_{u}^{(k)} + \pmb{b}_{z})$

Candidate:  $\bar{\pmb{h}} = \tanh (\pmb{W}_{mh}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hh}(\pmb{r}\odot \pmb{h}_u^{(k)}) + \pmb{b}_h)$

# Gated recurrent unit (GRU)

![img-7.jpeg](img-7.jpeg)

Reset:  $\pmb{r} = \sigma (\pmb{W}_{mr}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hr}\pmb{h}_{u}^{(k)} + \pmb{b}_{r})$

Update:  $\pmb{z} = \sigma (\pmb{W}_{mz}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hz}\pmb{h}_{u}^{(k)} + \pmb{b}_{z})$

Candidate:  $\bar{\pmb{h}} = \tanh (\pmb{W}_{mh}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hh}(\pmb{r}\odot \pmb{h}_{u}^{(k)}) + \pmb{b}_{h})$

State:  $\pmb{h}_u^{(k + 1)} = \pmb {z}\odot \bar{\pmb{h}} +(1 - \pmb {z})\odot \pmb{h}_u^{(k)}$

# Gated recurrent unit (GRU)

![img-8.jpeg](img-8.jpeg)

Reset:  $\pmb{r} = \sigma (\pmb{W}_{mr}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hr}\pmb{h}_{u}^{(k)} + \pmb{b}_{r})$

Update:  $\pmb{z} = \sigma (\pmb{W}_{mz}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hz}\pmb{h}_{u}^{(k)} + \pmb{b}_{z})$

Candidate:  $\bar{\pmb{h}} = \tanh (\pmb{W}_{mh}\pmb{m}_{\mathcal{N}(u)}^{(k)} + \pmb{W}_{hh}(\pmb{r}\odot \pmb{h}_{u}^{(k)}) + \pmb{b}_{h})$

State:  $\pmb{h}_u^{(k + 1)} = \pmb {z}\odot \bar{\pmb{h}} +(1 - \pmb {z})\odot \pmb{h}_u^{(k)}$

22/42

Message passing on graphs: Graph pooling

Graph-level predictions

- Neural message passing produces node embedding
- If we want to make a graph-level prediction:
- Set pooling from final node embedding to compute a graph-level embedding (similar to AGGREGATE).
- Pooling from all layers, e.g. using a LSTM- or GRU-based recurrent neural network.

24/42

# Generalized message passing

- Node states

$$
\boldsymbol {h} _ {u} ^ {(k + 1)} = \operatorname {U P D A T E} _ {\text {n o d e}} \left(\boldsymbol {h} _ {u} ^ {(k)}, \underbrace {\operatorname {A G G} _ {\text {e n}} \left(\left\{\boldsymbol {h} _ {(u , v)} ^ {(k)} \forall v \in \mathcal {N} (u) \right\}\right)} _ {\text {A g g r e g a t e e d g e s t o n o d e}}, \boldsymbol {h} _ {\mathcal {G}} ^ {(k)}\right)
$$

- Edge states

$$
\boldsymbol {h} _ {(u, v)} ^ {(k + 1)} = \operatorname {U P D A T E} _ {\text {e d g e}} \left(\boldsymbol {h} _ {(u, v)} ^ {(k)}, \underbrace {\operatorname {A G G} _ {\text {n e}} \left(\left\{\boldsymbol {h} _ {u} ^ {(k)} , \boldsymbol {h} _ {v} ^ {(k)} \right\}\right)} _ {\text {A g g r e g a t e n o d e s t o e d g e}}, \boldsymbol {h} _ {\mathcal {G}} ^ {(k)}\right)
$$

- Graph states

$$
\boldsymbol {h} _ {\mathcal {G}} ^ {(k + 1)} = \operatorname {U P D A T E} _ {\text {g r a p h}} \left(\boldsymbol {h} _ {\mathcal {G}} ^ {(k)}, \underbrace {\operatorname {A G G} _ {\text {e g}} \left(\left\{\boldsymbol {h} _ {u} ^ {(k)} \forall u \in \mathcal {V} \right\}\right)} _ {\text {A g g r e g a t e n o d e s t o g r a p h}}, \underbrace {\operatorname {A G G} _ {\text {n g}} \left(\left\{\boldsymbol {h} _ {(u , v)} ^ {(k)} \forall (u , v) \in \mathcal {E} \right\}\right)} _ {\text {A g g r e g a t e e d g e s t o g r a p h}}\right)
$$

Message passing on graphs: Machine learning tasks on graphs

26/42

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
- Graph neural networks can be used in all these tasks.

Transduction and induction

In a node prediction task we can distinguish 3 types of nodes

1. Training nodes: Used in message passing and to compute loss.
These are purely training data.
2. Transductive nodes: Nodes to train message passing but not in loss.
We can make node level predictions, having learned their specific embedding.
3. Inductive nodes: Not used in training.
We can make node level predictions by using the trained GNN to infer their embedding.
Contrast to shallow embeddings, where induction is not possible.

28/42

Geometric embedding

Handling symmetries

Consider graphs embedded in a vector space (nodes have coordinates.)

- Physical features change deterministically under certain transformations. We do not need to learn this.
- E.g. energy of a molecule does not change with translation and rotation.
- Forces on atoms translate and rotate with the molecule.
- Coordinate systems are arbitrary. The model should not be sensitive to choice of coordinate system.

Symmetry-aware modeling

Three general ways to handle symmetries

1. Data augmentation (brute-force)
2. Symmetry-aware descriptors (pre-processing, constraining the data space)
3. Symmetry-aware models (built-in, constraining the function space)
We focus on this approach

31/42

# Invariance and equivariance

Example: Invariance and equivariance in sequence translation:

- **Invariant**: Output does not depend on input translation.

$$
[1, 2, 3, 0, 0] \rightarrow [0, 1, 0, 0, 0]
$$

$$
[0, 1, 2, 3, 0] \rightarrow [0, 1, 0, 0, 0]
$$

$$
[0, 0, 1, 2, 3] \rightarrow [0, 1, 0, 0, 0]
$$

- **Equivariant**: Output translates with input translation.

$$
[1, 2, 3, 0, 0] \rightarrow [0, 1, 0, 0, 0]
$$

$$
[0, 1, 2, 3, 0] \rightarrow [0, 0, 1, 0, 0]
$$

$$
[0, 0, 1, 2, 3] \rightarrow [0, 0, 0, 1, 0]
$$

32/42

# Example: Molecular graph classification

- Data is a set of graphs (molecules)
- Each node has a position in a vector space
- We want the GNN to be invariant to rotation and translation

Graph embedded in vector space

![img-9.jpeg](img-9.jpeg)

Graph embedded in vector space

![img-10.jpeg](img-10.jpeg)

# Graph embedded in vector space

![img-11.jpeg](img-11.jpeg)

# Rotation and translation of graph embedded in vector space

- Invariant

$$
\vec {f} (T (\vec {h})) = \vec {f} (\vec {h})
$$

- Equivariant

$$
\vec {f} (T (\vec {h})) = T ^ {\prime} (\vec {f} (\vec {h}))
$$

$\vec{h}$  A node- or graph-level feature

$f$  Operation performed on the feature

$T$  Rotation+translation operator

$T^{\prime}$  Corresponding output operator

$T = T^{\prime}$  when  $x$  and  $f(x)$  are in the same vector space.

![img-12.jpeg](img-12.jpeg)
Adapted from Elise van der Pol, Daniel E. Worrall

How to make a invariant/equivariant GNN

- Initialize node states with *scalar* and *vector* features.
- Let node state consist of scalar and vectorial parts.
- Construct AGGREGATE, UPDATE, and READOUT functions using only invariant/equivariant operations.

This ensures that the entire GNN is invariant/equivariant.

Vector features

Examples of invariant/equivariant operations

$a \cdot \vec{v}$ Scale

$\vec{v}_1 + \vec{v}_2$ Addition/subtraction

$\| \vec{v} \|$ Norm

$\vec{v}_1 \cdot \vec{v}_2$ Dot product $= \| \vec{v}_1 \| \cdot \| \vec{v}_2 \| \cdot \cos(\theta)$

$\vec{v}_1 \times \vec{v}_2$ Cross product (3-d)

$\| \vec{v}_1 \times \vec{v}_2 \|$ Cross product (2-d) $= \| \vec{v}_1 \| \cdot \| \vec{v}_2 \| \cdot \sin(\theta)$

Which operations are rotationally invariant and which are equivariant?

36/42

Why vector features?

- It is limited, what can be expressed with scalar features
- Vector features encode local geometry in more detail
- Even if we want an invariant GNN, vector features are useful.

38/42

Pytorch implementation

Python tools for implementing GNN

- PyG (PyTorch Geometric) is a library built upon PyTorch to easily write and train GNNs
- However, basic GNNs are not so hard to implement in plain PyTorch.

Here, we opt for the latter approach for pedagogical reasons.

40/42

# TORCH.TENSOR.INDEX_ADD_

Tensor.index_add_(dim, index, source, *, alpha=1) → Tensor

Accumulate the elements of `alpha` times `source` into the `self` tensor by adding to the indices in the order given in `index`. For example, if `dim == 0`, `index[i] == j`, and `alpha=-1`, then the `i`th row of `source` is subtracted from the `j`th row of `self`.

The `dim`th dimension of `source` must have the same size as the length of `index` (which must be a vector), and all other dimensions must match `self`, or an error will be raised.

For a 3-D tensor the output is given as:

```
self[index[i], :, :] += alpha * src[i, :, :] # if dim == 0
self[:, index[i], :] += alpha * src[:, i, :] # if dim == 1
self[:, :, index[i]] += alpha * src[:, :, i] # if dim == 2
```

- **NOTE**

This operation may behave nondeterministically when given tensors on a CUDA device. See *Reproducibility* for more information.

## Parameters

- `dim` (int) – dimension along which to index
- `index` (Tensor) – indices of `source` to select from, should have `dtype` either `torch.int64` or `torch.int32`
- `source` (Tensor) – the tensor containing values to add

## Keyword Arguments

`alpha` (Number) – the scalar multiplier for `source`

41/42

# Exercises

# Exercises

- A Invariant aggregation functions
- B Simple graph neural networks
- C Programming exercise (GNN for graph-level classification)