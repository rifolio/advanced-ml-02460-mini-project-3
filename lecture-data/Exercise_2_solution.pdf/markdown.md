1 af 5

# TECHNICAL UNIVERSITY OF DENMARK

ADVANCED MACHINE LEARNING. MODULE 3.

EXERCISE 2. GRAPHS NEURAL NETWORKS

## CONTENTS

EXERCISE A: INVARIANT AGGREGATION FUNCTIONS ... 2
EXERCISE B: SIMPLE GRAPH NEURAL NETWORKS ... 3
EXERCISE C: PROGRAMMING EXERCISE ... 4

Exercise A Invariant aggregation functions

In a graph neural network, when aggregating information from neighbors or when aggregating information from all nodes to form a graph-level prediction, it is important that the aggregation function does not depend on the order of the inputs.

Question A.1: Which of the following functions on  $(\pmb{x}_1, \pmb{x}_2, \dots, \pmb{x}_N)$  are permutation invariant:

1. Sum:  $g\left(\sum_{n = 1}^{N}f(\pmb{x}_n)\right) = g\big(f(\pmb{x}_1) + f(\pmb{x}_2) + \dots +f(\pmb{x}_N)\big)$
2. Product:  $g\left(\prod_{n = 1}^{N}f(\pmb{x}_n)\right) = g\big(f(\pmb{x}_1)\cdot f(\pmb{x}_2)\cdot \dots \cdot f(\pmb{x}_N)\big)$
3. Maximum:  $g\left(\max_{n = 1}^{N}f(\pmb{x}_n)\right) = g\Big(\max \big(f(\pmb{x}_1),f(\pmb{x}_2),\dots ,f(\pmb{x}_N)\big)\Big)$
4. Concatenation:  $g\left(\operatorname{concat}_{n=1}^{N} f(\boldsymbol{x}_n)\right) = g\left([f(\boldsymbol{x}_1), f(\boldsymbol{x}_2), \ldots, f(\boldsymbol{x}_N)]\right)$

for arbitrary functions  $f(\cdot)$  and  $g(\cdot)$ .

Answer A.1: The sum, product, and maximum are permutation invarition, whereas the concatenation is not.

A

Exercise B Simple graph neural networks

Consider a GNN defined as

$$
\begin{array}{l} \text {A G G R E G A T E}: m _ {\mathcal {N} (u)} ^ {(k)} = \sum_ {v \in \mathcal {N} (u)} h _ {v} ^ {(k)} \\ \text {U P D A T E}: h _ {u} ^ {(k + 1)} = \frac {m _ {\mathcal {N} (u)} ^ {(k)}}{\sqrt {\sum_ {v \in \mathcal {V}} \left(m _ {\mathcal {N} (v)} ^ {(k)}\right) ^ {2}}} \\ \end{array}
$$

where the node representations are scalar, and initialized randomly.

Question B.1: Assuming that a large number of update rounds is computed, what will the node representations converge to?

Hint

Answer B.1: The update equations can be written in matrix-vector form as

$$
\boldsymbol {h} ^ {(k + 1)} = \frac {\boldsymbol {A h} ^ {(k)}}{\| \boldsymbol {A h} ^ {(k)} \|}
$$

This is the formula for the power iteration algorithm which (under mild assumptions) converges to the dominant eigenvector. In other words, this GNN computes the eigenvector centrality.

Consider the basic GNN, where each round consists of the following update:

$$
\boldsymbol {h} _ {u} ^ {(k)} = \sigma \left(\boldsymbol {W} _ {\text {s e l l}} ^ {(k)} \boldsymbol {h} _ {u} ^ {(k - 1)} + \boldsymbol {W} _ {\text {n e i g h .}} ^ {(k)} \sum_ {v \in \mathcal {N} (u)} \boldsymbol {h} _ {v} ^ {(k - 1)} + \boldsymbol {b} ^ {(k)}\right)
$$

Question B.2: If a graph contains  $|\mathcal{V}| = 10$ , the dimension of the node representation is  $D = 32$  (i.e.  $h_u^{(k)} \in \mathbb{R}^{32}$ ), the GNN performs 5 message passing rounds, and weight matrices are not shared between rounds, what is the total number of parameters in the GNN?

Answer B.2: The parameters are  $\mathbf{W}_{\mathrm{self}}^{(k)} \in \mathbb{R}^{32 \times 32}$ ,  $\mathbf{W}_{\mathrm{neigh.}}^{(k)} \in \mathbb{R}^{32 \times 32}$ , and  $\mathbf{b}^k \in \mathbb{R}^{32}$ , so the total number of parameters is

$$
(3 2 \cdot 3 2 + 3 2 \cdot 3 2 + 3 2) \cdot 5 = 1 0 4 0 0
$$

The number of parameters does not depend on the number of nodes in the graph data.

Exercise C Programming exercise

In this exercise you will work with a graph neural network for graph-level classification implemented in the script gnn_graph_classification.py.

We will use the MUTAG dataset introduced by Debnath et al.: a collection of nitroaromatic compounds (molecular graphs) and the task is to predict their mutagenicity on Salmonella typhimurium (graph-level binary classification). Vertices represent atoms and edges represent bonds, and the 7 discrete node labels represent the atom type (one-hot encoded). There are a total of 188 graphs in the dataset.

Question C.1: Examine and run the code for loading the graph data.

- Extract a single batch from the training loader using the code data_batch = next(iter(train_loader)).
- The variable data_batch will then contain the following important variables which you should examine to make sure you understand:

data_batch.x: Node features

data_batch.edge_index: Edges

data_batch.batch: Index of which graph in the batch each node belongs to.

Question C.2: Examine and run the code that defines the graph neural network SimpleGNN.

- Based on the components defined in the __init__ function and the computations carried out in the forward function, sketch a diagram of the graph neural network architecture.
- What are the aggregate and update functions that are implemented?
- Where and how are residual connections used?
- The messages are aggregated using a sum. To do this, the code uses the function torch.index_add. Make sure you understand this function, and look up its documentation if necessary. The same function is used to compute the graph level aggregation.
- What are the dimensions and purpose of the inputs and the output of the forward function?

Answer C.2:

- The aggregate function is:

$$
m _ {\mathcal {N} (u)} = \sum_ {v \in \mathcal {N} (u)} \operatorname {R e L U} \left(\boldsymbol {W} _ {\text {a g g .}} \boldsymbol {h} _ {v} ^ {(k)} + \boldsymbol {b} _ {\text {a g g .}}\right)
$$

The update function is:

$$
\boldsymbol {h} _ {u} ^ {(k + 1)} = \boldsymbol {h} _ {u} ^ {(k)} + \operatorname {R e L U} \left(\boldsymbol {W} _ {\text {u p d .}} \boldsymbol {m} _ {\mathcal {N} (u)} + \boldsymbol {b} _ {\text {u p d .}}\right)
$$

- The update function has a skip-connection (residual update) because it adds the update to the previous value of the node state.
- Let  $B$  denote the number of nodes in a batch and let  $V$  denote the number of edges:  $x$  are node features of dimension  $B \times 7$ . edge_index contains all edges and is of dimension  $2 \times V$ . batch is and index of which graph each node belongs to in the batch and is of dimension  $B$ .

Question C.3: Examine and run the remaining code to fit the GNN. Make sure you understand the following:

- Which loss function, optimizer, and learning rate are used?
- What does the learning rate scheduler do?

4 af 5

- How is the training/validation loss and accuracy computed.

After having fitted the GNN, examine the two generated plots. Does the model seem to overfit or underfit?

Answer C.3:

- We use binary cross entropy loss, the Adam optimizer and a learning rate initially set to $10^{-2}$.
- The learning rate scheduler decreases the learning rate by a factor of 0.995 after each epoch.
- When computing the loss, the lossfunction computes the mean over the batch. This is accumulated in a variable and weighted by the batch size divided by the total number of observations, to give the total mean loss. Similarly for the accuracy.

Question C.4: Modify the code to achieve the best possible validation loss. Do not change the training/validation split, and do no look at the test set. You might consider the following modifications:

- Change the model hyperparameters (the state dimension and number of message passing rounds)
- Change optimizer hyperparameters (learning rate schedule and number of epochs).
- Regularize by adding weight decay or dropout layers.
- Change the model architecture, for example by introducing a GRU update.

Question C.5: Using the provided code, save your test set predictions in a file test_predictions.pt, and hand it in on DTU Learn. I will compute the test loss on your predictions and lowest loss will be honored as the class winner.