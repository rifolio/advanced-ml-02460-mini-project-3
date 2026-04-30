1/50

Advanced machine learning

Theory and generative models

Mikkel N. Schmidt

Technical University of Denmark,
DTU Compute, Department of Applied Mathematics and Computer Science.

# Overview

1. Module overview
2. Results from last week's competition
3. Convolutions and Fourier transform
- Discrete convolutions
- Discrete Fourier transform
- Graph convolutions and Fourier transform
4. GNNs and probabilistic models
5. Generative models
- Traditional methods
- Variational autoencoders
- Generative adversarial networks (GANs)
- Evaluating graph generation
6. Exercises

3/50

Module overview

4/50

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

5/50

Results from last week’s competition

6/50

# Last week's competition

![img-0.jpeg](img-0.jpeg)

7/50

Convolutions and Fourier transform

8/50

Convolutions and Fourier transform: Discrete convolutions

# Convolution

- Convolution of discrete time series

$$
y[n] = (x * h)[n] = \sum_{k = -\infty}^{\infty} x[k]h[n - k]
$$

- Convolution of continuous signals

$$
g(\boldsymbol{x}) = (f * h)(\boldsymbol{x}) = \int_{\mathbb{R}^d} f(\boldsymbol{y}) h(\boldsymbol{x} - \boldsymbol{y}) \, \mathrm{d}\boldsymbol{y}
$$

We only look at discrete convolutions in this course.

9/50

10/50

# Causal filter

- When the filter coefficients are zero for $n &lt; 0$, i.e. $h[n] = 0 \ \forall \ n &lt; 0$, we say the filter is causal.

This means that the filter does not respond before the input signal is applied.

$$
y[n] = (x * h)[n] = \sum_{k = -\infty}^{n} x[k]h[n - k] = \sum_{k' = 0}^{\infty} x[n - k']h[k']
$$

10/50

# Causal filter

- When the filter coefficients are zero for $n &lt; 0$, i.e. $h[n] = 0 \ \forall \ n &lt; 0$, we say the filter is causal.

This means that the filter does not respond before the input signal is applied.

$$
y[n] = (x * h)[n] = \sum_{k = -\infty}^{n} x[k]h[n - k] = \sum_{k' = 0}^{\infty} x[n - k']h[k']
$$

- We can then write the convolution as

$$
y[n] = h[0]x[n] + h[1]x[n - 1] + h[2]x[n - 2] + \dots
$$

This shows how the convolution is a weighted linear combination of delayed versions of the input signal.

10/50

# Causal filter

- When the filter coefficients are zero for $n &lt; 0$, i.e. $h[n] = 0 \ \forall \ n &lt; 0$, we say the filter is causal.

This means that the filter does not respond before the input signal is applied.

$$
y[n] = (x * h)[n] = \sum_{k = -\infty}^{n} x[k]h[n - k] = \sum_{k' = 0}^{\infty} x[n - k']h[k']
$$

- We can then write the convolution as

$$
y[n] = h[0]x[n] + h[1]x[n - 1] + h[2]x[n - 2] + \dots
$$

This shows how the convolution is a weighted linear combination of delayed versions of the input signal.

- If we define a unit delay operator

$$
Dx[n] = x[n - 1]
$$

we can further write it as

$$
y[n] = (h[0] + h[1]D + h[2]D^2 + \dots)x[n]
$$

This is a nice parallel to graph convolutions as we will see later.

Circular convolution

- Circular discrete-time convolution

$$
y[n] = (x * h)[n] = \sum_{k=0}^{N-1} x[k] \cdot h[(n-k) \mod N]
$$

- We can define a (Toeplitz structured) filter matrix

$$
\boldsymbol{H} = \begin{bmatrix}
h[0] &amp; h[N-1] &amp; h[N-2] &amp; \cdots &amp; h[1] \\
h[1] &amp; h[0] &amp; h[N-1] &amp; \cdots &amp; h[2] \\
h[2] &amp; h[1] &amp; h[0] &amp; \cdots &amp; h[3] \\
\vdots &amp; \vdots &amp; \vdots &amp; \ddots &amp; \vdots \\
h[N-1] &amp; h[N-2] &amp; h[N-3] &amp; \cdots &amp; h[0]
\end{bmatrix}
\quad \text{i.e. } H_{kn} = h[(n-k) \mod N]
$$

a compute the convolution as

$$
\boldsymbol{y} = \boldsymbol{H} \boldsymbol{x}
$$

11/50

12/50

Convolutions and Fourier transform: Discrete Fourier transform

Discrete Fourier transform

- The discrete Fourier transform is defined as

$$
\tilde{x}[k] = \sum_{n=0}^{N-1} x[n] \overbrace{e^{-i \frac{2\pi k}{N} n}}^{{\cos\left(\frac{2\pi kn}{N}\right)} - i \sin\left(\frac{2\pi kn}{N}\right)} = \sum_{n=0}^{N-1} x[n] \underbrace{\omega_N^{k \cdot n}}_{\omega_N = e^{-i \frac{2\pi}{N}}}
$$

13/50

13/50

# Discrete Fourier transform

The discrete Fourier transform is defined as

$$
\tilde{x}[k] = \sum_{n=0}^{N-1} x[n] \overbrace{e^{-i \frac{2\pi k}{N} n}}^{{\cos\left(\frac{2\pi kn}{N}\right)} - i \sin\left(\frac{2\pi kn}{N}\right)} = \sum_{n=0}^{N-1} x[n] \underbrace{\omega_N^{k \cdot n}}_{\omega_N = e^{-i \frac{2\pi}{N}}}
$$

We can define a DFT basis matrix

$$
\boldsymbol{U}^H = \frac{1}{\sqrt{N}} \begin{bmatrix}
\omega_N^{0 \cdot 0} &amp; \omega_N^{0 \cdot 1} &amp; \dots &amp; \omega_N^{0 \cdot (N-1)} \\
\omega_N^{1 \cdot 0} &amp; \omega_N^{1 \cdot 1} &amp; \dots &amp; \omega_N^{1 \cdot (N-1)} \\
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \\
\omega_N^{(N-1) \cdot 0} &amp; \omega_N^{(N-1) \cdot 1} &amp; \dots &amp; \omega_N^{(N-1) \cdot (N-1)}
\end{bmatrix}
\quad \text{i.e.} \quad
U_{kn}^H = \omega_N^{k \cdot n}
$$

The DFT basis is a unitary matrix, $\boldsymbol{U}^H \boldsymbol{U} = \boldsymbol{I}$.

13/50

# Discrete Fourier transform

The discrete Fourier transform is defined as

$$
\tilde{x}[k] = \sum_{n=0}^{N-1} x[n] \overbrace{e^{-i \frac{2\pi k}{N} n}}^{{\cos\left(\frac{2\pi kn}{N}\right)} - i \sin\left(\frac{2\pi kn}{N}\right)} = \sum_{n=0}^{N-1} x[n] \underbrace{\omega_N^{k \cdot n}}_{\omega_N = e^{-i \frac{2\pi}{N}}}
$$

We can define a DFT basis matrix

$$
\boldsymbol{U}^H = \frac{1}{\sqrt{N}} \begin{bmatrix}
\omega_N^{0 \cdot 0} &amp; \omega_N^{0 \cdot 1} &amp; \dots &amp; \omega_N^{0 \cdot (N-1)} \\
\omega_N^{1 \cdot 0} &amp; \omega_N^{1 \cdot 1} &amp; \dots &amp; \omega_N^{1 \cdot (N-1)} \\
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \\
\omega_N^{(N-1) \cdot 0} &amp; \omega_N^{(N-1) \cdot 1} &amp; \dots &amp; \omega_N^{(N-1) \cdot (N-1)}
\end{bmatrix}
\quad \text{i.e.} \quad
U_{kn}^H = \omega_N^{k \cdot n}
$$

The DFT basis is a unitary matrix, $\boldsymbol{U}^H \boldsymbol{U} = \boldsymbol{I}$.

The DFT and inverse DFT can be computed as

$$
\tilde{\boldsymbol{x}} = \boldsymbol{U}^H \boldsymbol{x} \quad \boldsymbol{x} = \boldsymbol{U} \tilde{\boldsymbol{x}}
$$

DFT maps a signal onto a basis of complex sinusoids with different frequencies.

14/50

# Filtering with the DFT

- Convolution in the "time" domain

$$
y[n] = (x * h)[n] = \sum_{k=0}^{N-1} x[k] \cdot h[(n - k) \mod N]
$$

corresponds to multiplication in the "frequency" domain

$$
\tilde{y}[k] = \tilde{x}[k] \cdot \tilde{h}[k]
$$

14/50

# Filtering with the DFT

- Convolution in the "time" domain

$$
y[n] = (x * h)[n] = \sum_{k=0}^{N-1} x[k] \cdot h[(n-k) \mod N]
$$

corresponds to multiplication in the "frequency" domain

$$
\tilde{y}[k] = \tilde{x}[k] \cdot \tilde{h}[k]
$$

- In matrix form

$$
\boldsymbol{y} = \boldsymbol{H} \boldsymbol{x} \quad \Leftrightarrow \quad \tilde{\boldsymbol{y}} = \tilde{\boldsymbol{h}} \odot \tilde{\boldsymbol{x}}
$$

i.e.

$$
\boldsymbol{y} = \boldsymbol{U} \left( \underbrace{\boldsymbol{U}^H \boldsymbol{h}} \odot \underbrace{\boldsymbol{U}^H \boldsymbol{x}} \right)
$$

$$
\tilde{h} = \mathrm{DFT}(h) \quad \tilde{x} = \mathrm{DFT}(x)
$$

Proof: Convolution equals DFT multiplication

$$
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x \circledast h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n}
\end{array}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x * h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
\end{array}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x \odot h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
n = m + \ell \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{m=-\ell}^{N-1-\ell} h[m \mod N] \omega_N^{k \cdot (m + \ell)}
\end{array}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x * h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
n = m + \ell \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{m=-\ell}^{N-1-\ell} h[m \mod N] \omega_N^{k \cdot (m+\ell)} \\
= \sum_{\ell=0}^{N-1} x[\ell] \underbrace{\sum_{m=0}^{N-1} h[m] \omega_N^{k \cdot m} \omega_N^{k \cdot \ell}}_{h[k]} \\
\end{array}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x * h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
n = m + \ell \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{m=-\ell}^{N-1-\ell} h[m \mod N] \omega_N^{k \cdot (m+\ell)} \\
= \sum_{\ell=0}^{N-1} x[\ell] \underbrace{\sum_{m=0}^{N-1} h[m] \omega_N^{k \cdot m} \omega_N^{k \cdot \ell}}_{\tilde{h}[k]} \\
= \sum_{\ell=0}^{N-1} x[\ell] \tilde{h}[k] \omega_N^{k \cdot \ell}
\end{array}
$$

15/50

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x \odot h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
n = m + \ell \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{m=-\ell}^{N-1-\ell} h[m \mod N] \omega_N^{k \cdot (m+\ell)} \\
= \sum_{\ell=0}^{N-1} x[\ell] \underbrace{\sum_{m=0}^{N-1} h[m] \omega_N^{k \cdot m} \omega_N^{k \cdot \ell}}_{\tilde{h}[k]} \\
= \sum_{\ell=0}^{N-1} x[\ell] \tilde{h}[k] \omega_N^{k \cdot \ell} = \underbrace{\sum_{\ell=0}^{N-1} x[\ell] \omega_N^{k \cdot \ell} \tilde{h}[k]}_{\tilde{x}[k]}
\end{array}
$$

# Proof: Convolution equals DFT multiplication

$$
\begin{array}{l}
\tilde{y}[k] = \sum_{n=0}^{N-1} y[n] \omega_N^{k \cdot n} \\
y[n] = (x \odot h)[n] \\
= \sum_{n=0}^{N-1} \sum_{\ell=0}^{N-1} x[\ell] \cdot h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{n=0}^{N-1} h[(n - \ell) \mod N] \omega_N^{k \cdot n} \\
n = m + \ell \\
= \sum_{\ell=0}^{N-1} x[\ell] \sum_{m=-\ell}^{N-1-\ell} h[m \mod N] \omega_N^{k \cdot (m+\ell)} \\
= \sum_{\ell=0}^{N-1} x[\ell] \underbrace{\sum_{m=0}^{N-1} h[m] \omega_N^{k \cdot m} \omega_N^{k \cdot \ell}}_{\tilde{h}[k]} \\
= \sum_{\ell=0}^{N-1} x[\ell] \tilde{h}[k] \omega_N^{k \cdot \ell} = \underbrace{\sum_{\ell=0}^{N-1} x[\ell] \omega_N^{k \cdot \ell} \tilde{h}[k]}_{\tilde{x}[k]} = \tilde{x}[k] \cdot \tilde{h}[k]
\end{array}
$$

15/50

DFT basis

- DFT basis vectors

$$
\boldsymbol {u} _ {k} ^ {*} = \left[ \omega_ {N} ^ {k, 0}, \omega_ {N} ^ {k, 1}, \dots , \omega_ {N} ^ {k, (N - 1)} \right] ^ {\top}
$$

are complex exponentials with normalized frequency  $\frac{k}{N}$

$$
\left(\boldsymbol {u} _ {k} ^ {*}\right) _ {n} = \omega_ {N} ^ {k, n} = e ^ {- i \frac {2 \pi k}{N} n} = \cos \left(\frac {2 \pi k}{N} n\right) - i \sin \left(\frac {2 \pi k}{N} n\right)
$$

16/50

DFT basis

- DFT basis vectors

$$
\boldsymbol {u} _ {k} ^ {*} = \left[ \omega_ {N} ^ {k, 0}, \omega_ {N} ^ {k, 1}, \dots , \omega_ {N} ^ {k, (N - 1)} \right] ^ {\top}
$$

are complex exponentials with normalized frequency  $\frac{k}{N}$

$$
\left(\boldsymbol {u} _ {k} ^ {*}\right) _ {n} = \omega_ {N} ^ {k, n} = e ^ {- i \frac {2 \pi k}{N} n} = \cos \left(\frac {2 \pi k}{N} n\right) - i \sin \left(\frac {2 \pi k}{N} n\right)
$$

- Come in complex conjugate pairs

$$
\left(\boldsymbol {u} _ {N - k} ^ {*}\right) _ {n} = e ^ {- i \frac {2 \pi (N - k)}{N} n} = \overbrace {e ^ {- i \frac {2 \pi N}{N} n}} ^ {= 1} e ^ {i \frac {2 \pi k}{N} n} = e ^ {i \frac {2 \pi k}{N} n} = \left(\boldsymbol {u} _ {k}\right) _ {n} \tag {1}
$$

i.e. there are two complex conjugate components per frequency.

16/50

17/50

Convolutions and Fourier transform: Graph convolutions and Fourier transform

18/50

# Graph convolutions

- For a discrete time signal, we could write a convolution as a weighted sum of delayed versions of the input signal.

$$
y[n] = \left(h[0] + h[1]D + h[2]D^2 + \cdots\right) x[n]
$$

Here, the shift operator is the unit delay $Dx[n] = x[n - 1]$. On a chain graph, it would correspond to moving on to the next node (e.g. clockwise).

18/50

# Graph convolutions

- For a discrete time signal, we could write a convolution as a weighted sum of delayed versions of the input signal.

$$
y[n] = \left(h[0] + h[1]D + h[2]D^2 + \cdots\right) x[n]
$$

Here, the shift operator is the unit delay $Dx[n] = x[n - 1]$. On a chain graph, it would correspond to moving on to the next node (e.g. clockwise).

- For a signal propagating on a graph, we can use the adjacency matrix as a "neighborhood shift" operator.

Thus we can define a graph convolution as:

$$
\boldsymbol{y} = \left(h[0]\boldsymbol{I} + h[1]\boldsymbol{A} + h[2]\boldsymbol{A}^2 + \cdots\right) \boldsymbol{x}
$$

19/50

# Eigendecomposition of the cycle graph adjacency matrix

- Eigenvalue decomposition is given by

$$
\boldsymbol {A} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {H}
$$

where the eigenvectors satisfy

$$
\boldsymbol {A} \boldsymbol {u} _ {k} = \lambda_ {k} \boldsymbol {u} _ {k}
$$

# Eigendecomposition of the cycle graph adjacency matrix

Eigenvalue decomposition is given by

$$
\boldsymbol {A} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {H}
$$

where the eigenvectors satisfy

$$
\boldsymbol {A} \boldsymbol {u} _ {k} = \lambda_ {k} \boldsymbol {u} _ {k}
$$

It turns out that the eigenvectors are equal to the DFT basis vectors.

You will show this in the exercises.

![img-1.jpeg](img-1.jpeg)

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c c c c} 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 \\ 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 \\ 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 \end{array} \right]
$$

# Eigendecomposition of the cycle graph adjacency matrix

Eigenvalue decomposition is given by

$$
\boldsymbol {A} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {H}
$$

where the eigenvectors satisfy

$$
\boldsymbol {A} \boldsymbol {u} _ {k} = \lambda_ {k} \boldsymbol {u} _ {k}
$$

It turns out that the eigenvectors are equal to the DFT basis vectors.

You will show this in the exercises.

![img-2.jpeg](img-2.jpeg)

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c c c c} 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 \\ 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 &amp; 0 \\ 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 &amp; 1 \\ 1 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 &amp; 1 &amp; 0 \end{array} \right]
$$

Note: The eigendecomposition not unique

- Conjugate pairs can be rotated together.
- We can rotate to sine and cosine components, yielding a real matrix  $\mathbf{U}$ .

20/50

# Graph Fourier transform

- We can now define a generalized *graph Fourier transform*, given as the eigendecomposition of the adjacency matrix.

$$
\boldsymbol {A} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {H}
$$

- The basis vectors are no longer *frequencies* but more general *spatial patterns* on the graph vertices.

Example: Grid graph

![img-3.jpeg](img-3.jpeg)

# Example: Spatial components of the grid graph

Twelve "low-frequency" components.

![img-4.jpeg](img-4.jpeg)

23/50

# Summary of graph covolution

- A graph convolution is defined as

$$
\boldsymbol {y} = \sum_ {k = 0} ^ {N} h [ k ] \boldsymbol {A} ^ {k} \boldsymbol {x}
$$

- It can equivalently be computed using the graph Fourier transform

$$
\boldsymbol {y} = \boldsymbol {U} \left(\sum_ {k = 0} ^ {N} h [ k ] \boldsymbol {\Lambda} ^ {k}\right) \boldsymbol {U} ^ {H} \boldsymbol {x}
$$

where

$$
\boldsymbol {A} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {H}
$$

GNNs without message passing

- We can now define a graph convolution neural network

Node-level output features

$$
\widehat {\boldsymbol {Z}} = \operatorname {M L P} _ {\text {o u t}} \left(\underbrace {\sum_ {k = 0} ^ {N} h [ k ] \boldsymbol {A} ^ {k} \overbrace {\operatorname {M L P} _ {\text {i n}} (\boldsymbol {X})} ^ {\text {M L P o n n o d e i n p u t f e a t u r e s}}}\right)
$$

Graph convolution

Computations can also be performed in the Fourier domain.

- No message passing, but information flows through $N$ steps from nodes to neighbors through the graph convolution.

24/50

25/50

GNNs and probabilistic models

Probabilistic graphical models

- *Probabilistic graphical models:* Represent statistical relationships between random variables using a graph.
- *Approximate inference:* Finding the exact distributions of hidden variables is difficult.
- *Message passing:* Iteratively pass messages between nodes that updates beliefs about node’s distribution based on neighbors.
- *Variational approach:* Using a simpler, approximate distribution for each variable.
- *Conjugate models:* Classical variational message passing uses conjugate distributions so updates can be performed exactly.

# Kernel mean embedding

- Probability distributions can be embedded in a Hilbert space.
Hilbert space = A complete vector space with an inner product.
- Mean embedding using a feature map $\phi$

$$
\boldsymbol{\mu}_x = \int_{\mathbb{R}^d} \phi(\boldsymbol{x}) p(\boldsymbol{x}) \, \mathrm{d}\boldsymbol{x}
$$

If the feature map is “complex enough” the embedding is injective.

- In practice, we do not work directly with the feature map, but define it implicitly using a kernel (inner product).

$$
\left\langle \phi(\boldsymbol{x}), \phi(\boldsymbol{x}') \right\rangle = k(\boldsymbol{x}, \boldsymbol{x}')
$$

27/50

# Graphs as graphical models

A graph defines a Markov random field

Node features (observed)

$$
p(\{x_v\}, \{z_v\}) \propto \prod_{v \in \mathcal{V}} \Phi(x_v, z_v) \prod_{(u,v) \in \mathcal{E}} \Psi(z_u, z_v)
$$

Node embedding (latent variables)

28/50

# Graphs as graphical models

A graph defines a Markov random field

Node features (observed)

$$
p \left(\left\{x _ {v} \right\}, \left\{z _ {v} \right\}\right) \propto \prod_ {v \in \mathcal {V}} \Phi \left(x _ {v}, z _ {v}\right) \prod_ {(u, v) \in \mathcal {E}} \Psi \left(z _ {u}, z _ {v}\right)
$$

Node embedding (latent variables)

Example

![img-5.jpeg](img-5.jpeg)

Nodes could be vector-valued, but I will assume scalars for simplicity.

Mean field variational inference

- Approximate posterior distribution of latent variables

$$
p \left(\left\{z _ {v} \right\} \mid \left\{x _ {v} \right\}\right) \approx q \left(\left\{z _ {v} \right\}\right) = \prod_ {v \in \mathcal {V}} q _ {v} \left(z _ {v}\right)
$$

- Minimize KL to true posterior

$$
\mathrm {K L} (q | p) = \int \prod_ {v \in \mathcal {V}} q _ {v} (z _ {v}) \log \frac {\prod_ {v \in \mathcal {V}} q _ {v} (z _ {v})}{p (\{z _ {v} \} | \{x _ {v} \})} \prod_ {v \in \mathcal {V}} \mathrm {d} z _ {v}
$$

- Optimum satisfies fixed point equations

$$
\log q _ {v} ^ {(t + 1)} (z _ {v}) = c _ {v} + \log \Phi (x _ {v}, z _ {v}) + \sum_ {u \in \mathcal {N} (v)} \int q _ {u} ^ {(t)} (z _ {u}) \log \Psi (z _ {u}, z _ {v}) \mathrm {d} z _ {u}
$$

29/50

# Mean field variational message passing

- Optimum satisfies fixed point equations

$$
\log q _ {v} ^ {(t + 1)} (z _ {v}) = c _ {v} + \log \Phi (x _ {v}, z _ {v}) + \sum_ {u \in \mathcal {N} (v)} \int q _ {u} ^ {(t)} (z _ {u}) \log \Psi (z _ {u}, z _ {v}) \mathrm {d} z _ {u}
$$

Example (node 4)

![img-6.jpeg](img-6.jpeg)

$$
\log q _ {4} ^ {(t)} (z _ {4}) = c _ {4} + \log \Phi (x _ {4}, z _ {4}) + \int q _ {2} ^ {(t)} (z _ {2}) \log \Psi (z _ {2}, z _ {4}) d z _ {2} + \int q _ {3} ^ {(t)} (z _ {3}) \log \Psi (z _ {3}, z _ {4}) d z _ {3}
$$

31/50

# Mean field VI embedded in Hilbert space

Suppose we represent $q_{v}(z_{v})$ as mean embeddings

$$
\boldsymbol {\mu} _ {v} = \int \phi (z) q _ {v} (z _ {v}) \mathrm {d} z
$$

the fixed point equations can be written as

$$
\boldsymbol {\mu} _ {v} ^ {(t + 1)} = \boldsymbol {c} + f \left(\boldsymbol {\mu} _ {v} ^ {(t)}, x _ {v}, \left\{\boldsymbol {\mu} _ {u}, \forall u \in \mathcal {N} (v) \right\}\right)
$$

This looks very much like message passing GNN updates.

We can now use this to either

- Define potential functions $\Phi$ and $\Psi$ and a kernel $k(\cdot, \cdot)$ and analytically derive updates.
- Learn embeddings and assume that they could correspond to some probabilistic model.

32/50

# Generative models

33/50

Generative models: Traditional methods

34/50

# Traditional models

- Erdős-Rényi model
Generate each possible edge independently with a fixed probability
$$
P(A_{u,v} = 1) = r, \quad \forall u, v \in \mathcal{V}, u \neq v.
$$

Traditional models

## Erdős-Rényi model

Generate each possible edge independently with a fixed probability

$$
P(A_{u,v} = 1) = r, \quad \forall u, v \in \mathcal{V}, u \neq v.
$$

## Stochastic blockmodels

1. Specify a categorical distribution $p_1, \ldots, p_\gamma$ over blocks $\mathcal{C}_1, \ldots, \mathcal{C}_\gamma$.
2. Assign each node randomly to a block according to $p_i$.
3. Specify probabilities of edges between (and within) each block $r_{i,j}$
4. Generate edges between nodes in blocks $i$ and $j$ according to

$$
P(A_{u,v} = 1) = r_{i,j}, \quad \forall u \in \mathcal{C}_i, v \in \mathcal{C}_j, u \neq v, \forall i, j \in (1, \ldots, \gamma)
$$

Possible simplification

$$
r_{i,j} = \begin{cases} r_{\mathrm{in}} &amp; i = j \\ r_{\mathrm{out}} &amp; i \neq j \end{cases}
$$

34/50

Traditional models

## Preferential attachment

1. Start with a seed graph, e.g. a fully connected graph with $m_0$ nodes (Possibly just a single node)
2. Iteratively add nodes to the graph and form edges according to

$$
P(A_{u,v} = 1) = \frac{d_v^{(t)}}{\sum_{v' \in \mathcal{V}^{(t)}} d_v^{(t)}}
$$

Nodes that are already highly connected are more likely to connect.

There are also linear and non-linear extensions that allows more detailed control of the distribution.

Applications of traditional methods

- Generating synthetic benchmark data: Testing algorithms, validating hypotheses, and studying network phenomena in controlled settings.
- Creating null models: If observed properties of a real graph significantly differ from those of the null model, it suggests that the real graph exhibits non-random structural characteristics or underlying processes.

37/50

Generative models: Variational autoencoders

38/50

# Latent variable models

- We have already worked with a generative model: *Shallow node embedding*.
- Each node is embedded in a space, i.e. is endowed with a latent variable.
- The graph edges are generated according to a pairwise decoder.

# Encoder-decoder perspective

Encoder: Maps a vectex into a latent representation.

![img-7.jpeg](img-7.jpeg)

Example 1: Learned embedding (lookup table).

Example 2: Message passing graph neural network.

# Encoder-decoder perspective

Encoder: Maps a vectex into a latent representation.

![img-8.jpeg](img-8.jpeg)

Example 1: Learned embedding (lookup table).

Example 2: Message passing graph neural network.

Pairwise decoder: Maps a pair of latent variables into a node pair statistic.

$$
\mathrm {D E C}: \mathbb {R} ^ {d} \times \mathbb {R} ^ {d} \to \mathbb {S}
$$

Example 1: Predict link or non-link,  $\mathbb{S} = \{0,1\}$ .

Example 2: Predict a graph-based similarity measure,  $\mathbb{S} = \mathbb{R}_+ = \{x\in \mathbb{R}|x\geq 0\}$

# Variational autoencoders

- **Probabilistic encoder**: Takes graph as input and computes parameters of a probabilistic latent embedding.

$$
\boldsymbol {Z} \sim \mathcal {N} \left(\boldsymbol {\mu} _ {\phi} (\mathcal {G}), \boldsymbol {\Sigma} _ {\phi} (\mathcal {G})\right)
$$

For example the mean and variance of a Normal distribution.

- **Probabilistic decoder**: Takes a latent representation and produces a conditional distribution over entries of the adjacency matrix.

$$
\boldsymbol {A} _ {u, v} \sim \operatorname {B e r n o u l l i} \left(p _ {u, v} (\boldsymbol {Z})\right)
$$

For example the parameters of independent Bernoulli variables for each possible link.

- **A prior distribution**:

$$
p (\boldsymbol {Z}) = \mathcal {N} (\boldsymbol {0}, \boldsymbol {I})
$$

Typically this is a standard Normal.

40/50

Variational autoencoders

- Probabilistic encoder: $\mathbf{Z} \sim \mathcal{N}(\boldsymbol{\mu}_{\phi}(\mathcal{G}), \boldsymbol{\Sigma}_{\phi}(\mathcal{G}))$
- Probabilistic decoder: $A_{u,v} \sim \mathrm{Bernoulli}\big(p_{u,v}(\mathbf{Z})\big)$
- A prior distribution: $p(\mathbf{Z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

Given these components we train the model by maximizing the evidence lower bound (ELBO)

$$
\mathcal{L} = \sum_{\mathcal{G} \in \mathcal{D}} \mathbb{E}_{q_{\phi}(\mathbf{Z} | \mathcal{G})} \left[ p_{\theta}(\mathcal{G} | \mathbf{Z}) \right] - \mathrm{KL} \left[ q_{\phi}(\mathbf{Z} | \mathcal{G}) \| p(\mathbf{Z}) \right]
$$

41/50

# Node-level latents

- Encoder: Probabilistic node embedding

GNN outputs two vectors per node

$$
\underbrace{\mu_{Z}, \log \sigma_{Z}}_{\text{Mean and variance for each node}} = \overbrace{\operatorname{GNN}(A, X)}^{\text{GNN}(A, X)}
$$

Sample latent node embedding using the reparametrization trick

$$
Z = \mu_{Z} + \epsilon \odot \log \sigma_{Z}
$$

# Node-level latents

- Encoder: Probabilistic node embedding

GNN outputs two vectors per node

$$
\underbrace{\mu_{Z}, \log \sigma_{Z}}_{\text{Mean and variance for each node}} = \overbrace{\operatorname{GNN}(A, X)}^{\text{GNN}(A, X)}
$$

Sample latent node embedding using the reparametrization trick

$$
Z = \mu_{Z} + \epsilon \odot \log \sigma_{Z}
$$

- Decoder: Probabilistic pairwise decoder compute a binary probability

$$
P(A_{u,v} = 1) = \sigma(z_{u}^{\top} z_{v} + b n)
$$

Same decoder we used for shallow embeddings.

42/50

# Node-level latents

- Encoder: Probabilistic node embedding

GNN outputs two vectors per node

$$
\underbrace{\mu_{Z}, \log \sigma_{Z}}_{\text{Mean and variance for each node}} = \overbrace{\operatorname{GNN}(A, X)}^{\text{GNN}(A, X)}
$$

Sample latent node embedding using the reparametrization trick

$$
Z = \mu_{Z} + \epsilon \odot \log \sigma_{Z}
$$

- Decoder: Probabilistic pairwise decoder compute a binary probability

$$
P(A_{u,v} = 1) = \sigma(z_{u}^{\top} z_{v} + b n)
$$

Same decoder we used for shallow embeddings.

Note: The basic node-level VAE has very limited performance, because the latent variable model (decoder) is very weak. This can be improved by using a GNN as decoder also.

42/50

# Graph-level latents

- Encoder: Probabilistic graph embedding

GNN outputs two vectors

$$
\underbrace{\boldsymbol{\mu}_{z}, \log \sigma_{z}}_{\text{Mean and variance for the entire graph}} = \overbrace{\operatorname{GNN}(\boldsymbol{A}, \boldsymbol{X})}^{\text{GNN}(A, X)}
$$

Sample latent graph embedding using the reparametrization trick

$$
z = \boldsymbol{\mu}_{z} + \epsilon \odot \log \sigma_{z}
$$

43/50

# Graph-level latents

- Encoder: Probabilistic graph embedding

GNN outputs two vectors

$$
\underbrace{\boldsymbol{\mu}_{z}, \log \sigma_{z}}_{\text{Mean and variance for the entire graph}} = \overbrace{\operatorname{GNN}(\boldsymbol{A}, \boldsymbol{X})}^{\text{GNN}(A, X)}
$$

Sample latent graph embedding using the reparametrization trick

$$
z = \boldsymbol{\mu}_{z} + \epsilon \odot \log \sigma_{z}
$$

- Decoder: Neural network

Multilayer perceptron

$$
\tilde{\boldsymbol{A}} = \overbrace{\sigma(\operatorname{MLP}(z))}^{\text{Multilayer perceptron}}
$$

A matrix of edge probabilities

43/50

43/50

# Graph-level latents

- **Encoder**: Probabilistic graph embedding

GNN outputs two vectors

$$
\underbrace{\boldsymbol{\mu}_z, \log \sigma_z}_{\text{Mean and variance for the entire graph}} = \overbrace{\operatorname{GNN}(\boldsymbol{A}, \boldsymbol{X})}^{\text{GNN}(A, X)}
$$

Sample latent graph embedding using the reparametrization trick

$$
z = \boldsymbol{\mu}_z + \epsilon \odot \log \sigma_z
$$

- **Decoder**: Neural network

Multilayer perceptron

$$
\tilde{\boldsymbol{A}} = \overbrace{\sigma(\operatorname{MLP}(z))}^{\text{Multilayer perceptron}}
$$

A matrix of edge probabilities

**Note**: Two problems:

1. We need to assume a fixed number of nodes: This can be handled by assuming a maximum number and masking.
2. We do not know the order of the nodes: This can be handled by (heuristic) graph matching to optimize the node order on the fly.

44/50

Generative models: Generative adversarial networks (GANs)

45/50

# Adversarial approaches

- Generator: Generates data from a (random) latent variable

$$
g_{\theta}: \mathbb{R}^{d} \to \mathcal{X}
$$

- Discriminator: Distinguish between real and generated data

$$
d_{\phi}: \mathcal{X} \to [0, 1]
$$

- Adversarial optimization: Train the discriminator to distinguish real and generated, while training the generator to fool the discriminator

$$
\min_{\theta} \max_{\phi} \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x})} \left[ \log(1 - d_{\phi}(\boldsymbol{x})) \right] + \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})} \left[ \log d_{\phi}(g_{\theta}(\boldsymbol{z})) \right]
$$

Sample from data distribution
Sample from prior

GAN for graph generation

- Generator: Could be as simple multilayer perceptron.

$$
\tilde{\boldsymbol{A}} = \sigma\left(\mathrm{MLP}(\boldsymbol{z})\right)
$$

- Discriminator: GNN based graph classification.

This approach does not require a specific node ordering: The discriminator is permutation invariant.

46/50

47/50

# Generative models: Evaluating graph generation

Evaluating graph generation

- Difficult to directly compare generative models.
No uniform definition of likelihood across different methods.

48/50

# Evaluating graph generation

- Difficult to directly compare generative models.
No uniform definition of likelihood across different methods.
- Analyze graph statistics: We can define a number of statistics such as the distributions of
- Degree
- Clustering coefficient
- Eigenvector centrality
Measure a distributional distance between generated graphs and real test graphs.

48/50

# Evaluating graph generation

- Difficult to directly compare generative models.
No uniform definition of likelihood across different methods.

- **Analyze graph statistics**: We can define a number of statistics such as the distributions of
- Degree
- Clustering coefficient
- Eigenvector centrality
Measure a distributional distance between generated graphs and real test graphs.

- **Distance measure**: Total variation
$$
d(s_{\text{gen}}, s_{\text{test}}) = \sup_{x} |P(s_{\text{gen}} \in x), P(s_{\text{test}} \in x)|
$$
Measure the maximum disagreement in probability assignment. For discrete distributions this is simply half the $L_1$ distance.

49/50

# Exercises

50/50

# Exercises

A Graph convolutions
B Graph Fourier transform
C Programming exercise (Graph Fourier transform)