# SVD-Efficient-Computation-and-Hard-Threshold
# Investigating Geometry of Recursive Dyadic Sums in SVD


## Project Overview

This repository contains a Jupyter notebook that delves into the geometric characteristics of recursive dyadic sums formed through Singular Value Decomposition (SVD). The primary focus is on observing the behavior of the Frobenius norm as a matrix undergoes successive SVD decompositions and dyadic summations. The goal is to understand how the "importance" of matrix components changes with decomposition depth and to explore pathways for potentially faster matrix reconstruction techniques.

Additionally, the project explores and benchmarks different methods for computing SVD, including the "Snapshot Method" and a "Randomized Method." It also investigates the application of optimal hard thresholds for singular values in both known and unknown noise scenarios, for both square and rectangular matrices.

## Introduction

Singular Value Decomposition (SVD) is a powerful matrix factorization technique with broad applications in data science, image processing, and numerical analysis. It decomposes a matrix into three constituent matrices: `U` (left singular vectors), `S` (singular values), and `Vt` (right singular vectors, transposed).

A key aspect of SVD is that any matrix can be represented as a sum of rank-one matrices, known as the dyadic sum. This project specifically investigates the *recursive* nature of this dyadic sum, examining the geometric implications, particularly concerning the Frobenius norm of the resulting matrices at different depths of decomposition.

The core idea is that as a matrix is recursively decomposed and summed in this manner, its Frobenius norm tends to approach zero. This trend can be interpreted as a "life signal" of the matrix, indicating components that become progressively less significant.

## Key Concepts

* **Singular Value Decomposition (SVD)**: A factorization of a real or complex matrix. It is the generalization of the eigendecomposition of a positive semidefinite normal matrix (e.g., a symmetric matrix with positive eigenvalues) to any `m x n` matrix.
    `A = U @ S @ Vt`
    Where:
    * `U`: Unitary matrix (left singular vectors).
    * `S`: Diagonal matrix containing singular values.
    * `Vt`: Unitary matrix (right singular vectors, transposed).
* **Dyadic Sum**: The sum of rank-one matrices obtained from the SVD components. Specifically, it's the sum of the `r`-th singular value multiplied by the outer product of the `r`-th column vector of `U` and the `r`-th row vector of `Vt`.
* **Frobenius Norm**: A measure of the "size" or "magnitude" of a matrix. For a matrix `A`, the Frobenius norm is defined as `||A||_F = sqrt(sum(sum(abs(A_ij)^2)))`. It quantifies the overall magnitude of the matrix elements.

## Methodology

The project's methodology can be broken down into two main parts:

1.  **Recursive Dyadic Sum Analysis**: The `rec_dyatic_sum` function is implemented to recursively decompose a matrix through SVD and then sum its dyadic components. At each level of recursion, the Frobenius norm of the resulting sum is calculated and stored. This process aims to illustrate the "life signal" of the matrix, showing how the norm converges to zero.

2.  **SVD Computation Methods Benchmarking**: Various methods for computing SVD are explored and benchmarked for efficiency. This involves:
    * Generating random matrices of varying sizes.
    * Measuring the execution time for each SVD method.
    * Comparing their performance across different matrix dimensions.

## Thresholding for Singular Values

The project also explores methods for determining optimal hard thresholds for singular values, which are crucial for noise reduction and data compression.

* **Known Noise Scenario**:
    * For a square `n x n` matrix with known noise magnitude `sigma`, the optimal hard threshold for singular values is calculated using the formula:
        `Threshold_SQ = (4 / sqrt(3)) * sqrt(n) * sigma`
    * For a rectangular `n x m` matrix (where `n >> m`) with known noise magnitude `sigma`, a more complex formula involving `LambdaFunction(B)` is used:
        `Threshold_RT = LambdaFunction(B) * sqrt(n) * sigma`
        Where `B = m / n` and
