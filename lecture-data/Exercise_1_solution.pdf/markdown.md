1 af 5

# TECHNICAL UNIVERSITY OF DENMARK

ADVANCED MACHINE LEARNING. MODULE 3.

EXERCISE 1. GRAPHS AND NODE EMBEDDINGS.

# CONTENTS

EXERCISE A: NODE LEVEL STATISTICS 2

EXERCISE B: RANDOM WALKS 3

EXERCISE C: SHALLOW EMBEDDINGS 4

EXERCISE D: PROGRAMMING EXERCISE 5

Exercise A

Node level statistics

Consider the following graph

![img-0.jpeg](img-0.jpeg)

The following code prints the adjacency matrix and computes its eigendecomposition:

```txt
&gt;&gt;&gt; print(A)
[[0. 0. 1. 1. 0. 1. 0.]
[0. 0. 0. 0. 1. 1. 1.]
[1. 0. 0. 1. 0. 1. 0.]
[1. 0. 1. 0. 0. 1. 0.]
[0. 1. 0. 0. 0. 1. 1.]
[1. 1. 1. 1. 1. 0. 1.]
[0. 1. 0. 0. 1. 1. 0.]]
```

```txt
&gt;&gt;&gt; lambda, E = np.linalg.eig(A)
```

```txt
&gt;&gt;&gt; print(np.round(lambda, 3))
[3.646 2. -1.646 -1. -1. -1. -1. ]
```

```txt
&gt;&gt;&gt; print(np.round(E, 3))
[[-0.339 -0.408 -0.228 -0.816 0.004 -0.084 0.126]
[-0.339 0.408 -0.228 -0. -0.374 0.69 -0.255]
[-0.339 -0.408 -0.228 0.408 0.511 0.114 -0.493]
[-0.339 -0.408 -0.228 0.408 -0.515 -0.03 0.367]
[-0.339 0.408 -0.228 0. -0.176 -0.709 -0.377]
[-0.558 -0. 0.83 -0. 0. -0. -0. ]
[-0.339 0.408 -0.228 0. 0.55 0.019 0.632]]
```

Question A.1: Determine the eigenvector centrality for each node in the graph.

Answer A.1: By examining the adjacency matrix, we see that all nodes have degree 3, except the node in the 6th row/column which has degree 6. This means that we can identify node 6 as the central node in the graph. The remaining nodes are not distinguishable, and must thus have the same centrality.

The eigenvector centrality for each node are given as the components of the principal eigenvector (corresponding to the largest eigenvalue), thus the centralities are found as the first column in the matrix  $\mathbb{E}$ . However, since eigenvectors are not unique in terms of their sign, and centralities are defined to be positive, we need to flip their sign. Thus, the centralities are 0.339 (for node 1-5 and 7) and 0.558 (for node 6).

Question A.2: Determine the clustering coefficient for each node in the graph.

Answer A.2: Again, since all the non-central nodes are structurally equivalent, they have the same clustering coefficient,  $c_{1-5,7} = \frac{3}{\binom{4}{2}} = 1$  which implies that all of the non-central nodes' neighbors are neighbors to each other. The central node has clustering coefficient  $c_6 = \frac{6}{\binom{6}{2}} = 0.4$ .

Exercise B Random walks

Question B.1: Given a graph with adjacency matrix $\mathbf{A}$ and a starting node chosen randomly according to a discrete distribution $\mathbf{p}$, what is the final node's probability distribution after taking a single step from the starting node along an edge chosen uniformly at random?

Answer B.1:

$$
\mathbf{A} \mathbf{D}^{-1} \mathbf{p} = \mathbf{P} \mathbf{p}
$$

where $\mathbf{D}$ is a matrix with node degrees on the diagonal and we define the stochastic matrix $\mathbf{P} = \mathbf{A} \mathbf{D}^{-1}$.

Question B.2: Given a graph with adjacency matrix $\mathbf{A}$, how many distinct paths of length $t$ can we find starting from a specific node (say node 1)?

Hint: We can represent the initial state as a vector that is one for the start node and zero elsewhere. Where can we end up after taking a single step, and how can this be computed using a matrix-vector product? How can this approach be generalized to $t$ steps?

Answer B.2: Say we start at node 1, and define the initial state vector as $\mathbf{x} = [1,0,\ldots]^{\top}$. We take this to mean, that we have a single path of length zero at node 1. Then the nodes we can end up in after a single step can be computed as $\mathbf{A} \mathbf{x}$. The result counts the number of paths of length 1 from node 1 that end in each node. Pre-multiplying by the adjacency matrix essentially distributes the counts of each node to its neighbors, so by computing $\mathbf{A} \mathbf{A} \mathbf{x}$ we count the number of paths of length 2 from node 1 that end in each node. After $t$ steps, the total number of distinct paths is

$$
\sum_{v=1}^{|V|} (\mathbf{A}^t \mathbf{x})_v
$$

i.e. the sum of the first column of the matrix $\mathbf{A}^t$.

B

3 af 5

Exercise C Shallow embeddings

In this exercise we will use the decoder  $\mathrm{DEC}(\pmb{z}_u, \pmb{z}_v) = \sigma(\pmb{z}_u^\top \pmb{z}_v + b)$ , where  $\sigma(x) = \frac{1}{1 + e^{-x}}$  denotes the sigmoid function.

Question C.1: As a warm-up, show that  $1 - \sigma(x) = \sigma(-x)$ .

Based on this, can you spot a typo in eq. 3.12 on page 37 in the book?

Answer C.1:

$$
1 - \sigma (x) = 1 - \frac {1}{1 + e ^ {- x}} = \frac {1 + e ^ {- x} - 1}{1 + e ^ {- x}} = \frac {e ^ {- x}}{1 + e ^ {- x}} = \frac {1}{e ^ {x} + 1} = \frac {1}{1 + e ^ {x}} = \sigma (- x)
$$

The typo is that  $\log (-\sigma (\pmb{z}_u^\top \pmb{z}_{v_u}))$  should be  $\log (\sigma (-\pmb{z}_u^\top \pmb{z}_{v_u}))$ .

Let  $S_{u,v} \in \{0,1\}$  denote a binary feature corresponding to an non-edge/edge between nodes  $u$  and  $v$  (i.e.  $S_{u,v}$  are the elements of the adjacency matrix.) Let  $P_{u,v} = P(S_{u,v} = 1 | \pmb{z}_u, \pmb{z}_v, b) = \sigma(\pmb{z}_u^\top \pmb{z}_v + b)$  denote the predicted probability that the edge is present, given the latent node embeddings  $\pmb{z}_u, \pmb{z}_v$  and bias  $b$ .

Question C.2: Write the cross entropy loss for the single observation  $S_{u,v}$  (in terms of  $P_{u,v}$  and  $S_{u,v}$ ).

Answer C.2:

$$
L _ {u, v} = - \left(S _ {u, v} \log P _ {u, v} + \left(1 - S _ {u, v}\right) \log \left(1 - P _ {u, v}\right)\right)
$$

Question C.3: Let us assume that the embeddings  $\mathbf{z}_u$  and  $\mathbf{z}_v$  are orthogonal, such that their dot product is zero, and let us further assume that the bias is zero,  $b = 0$ . What is the probability of an edge between node  $u$  and  $v$ ?

Answer C.3: In that case we have

$$
P \left(S _ {u, v} = 1 \mid \boldsymbol {z} _ {u}, \boldsymbol {z} _ {v}, b\right) = \sigma \left(\boldsymbol {z} _ {u} ^ {\top} \boldsymbol {z} _ {v} + b\right) = \sigma (0) = \frac {1}{1 + \exp (- 0)} = 0. 5
$$

In this exercise you will work with a shallow node embedding implemented in the script shallow_embedding.py. The code loads a graph from a file: This graph is simulated from a shallow embedding model, so that we know the ground truth probability of each possible link. In this exercise we will fit a shallow embedding model to the data and see how well we can estimated the ground truth.

###### Question D.1

Examine and run the code for loading the graph data.

- Understand how the graph is represented as a matrix as well as in the form of a set of index pairs and target values.
- It can perhaps help to visualize the adjacency matrix.

###### Question D.2

Examine and run the implementation of the class Shallow.

- Understand how the node embeddings are implemented using torch.nn.Embedding. Look up the documentation if needed.
- Understand what the forward function computes. What exactly is the role of the variables rx and tx?

###### Question D.3

Examine and run the code to fit the model. In this version, the loss is computed on the entire graph (no train/validation split and no mini batching).

- Experiment with different number of max_step.
- Experiment with different embedding dimensions. How does the embedding dimension influence the training loss?

###### Question D.4

Modify the code to use a train/validation split.

- Make a random split of the data (each node pair) into a training set (e.g. 80%) and a validation set (e.g. 20%).
- Modify the code to train on only the training data.
- Write code to compute the loss of the trained model on the validation set.
- Experiment with different embedding dimensions. What is the optimal embedding dimension when computing the loss on the validation set?

###### Question D.5

Hand in your predictions:

- Using the train/validation procedure you have implemented (or any other updates, hacks and modifications) to optimize the model. Compute what you believe is the best possible predicted link probability.
- Using the provided code, save your predictions in a file, link_probabilities.pt, and hand it in on DTU Learn.

I will compute the ground truth loss on your predictions and lowest generalization loss will be honored as the class winner.