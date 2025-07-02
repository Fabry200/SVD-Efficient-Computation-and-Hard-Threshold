
# Computationally efficient method for SVD. Investigating Geometry of Recursive Dyadic Sums in SVD and hard threeshold for limiting singular value truncation


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
        Where `B = m / n` and `LambdaFunction(B) = sqrt(2(B+1) + (8B / (B+1 + sqrt(B^2 + 13B + 1))))`.

* **Unknown Noise Scenario (Rectangular Matrix)**:
    * When the noise is unknown for a rectangular matrix, the optimal hard threshold is determined using a method that involves the median of the singular values. This requires solving an integral equation to find `mu_beta`, which represents the median of the Marchenko-Pastur distribution.
        `Threshold_RT_UnknownNoise = (LambdaFunction(B) / mu_beta) * S[len(S)//2]`
        Where `S[len(S)//2]` is the median singular value of the noised matrix. The `mu_beta` is found by numerically solving an equation involving the integral of a specific function related to the Marchenko-Pastur distribution.

## SVD Computation Methods

The following methods for computing SVD are analyzed and compared in this project:

### 1. Regular SVD
This refers to the standard `np.linalg.svd` function provided by NumPy. It performs a full and exact Singular Value Decomposition. While highly accurate, its computational cost can be prohibitive for very large matrices, typically scaling as `O(min(N*M^2, N^2*M))` for an `N x M` matrix. It serves as a baseline for comparison with more optimized methods.

### 2. Snapshot Method (also known as Method of Snapshots or Proper Orthogonal Decomposition (POD))
The Snapshot Method is an efficient technique particularly suited for tall-skinny matrices (i.e., when the number of rows `N` is much greater than the number of columns `M`, `N >> M`). Instead of directly computing the SVD of the original `N x M` data matrix `X`, it leverages the properties of the covariance matrix.

* **How it works:**
    1.  It computes the smaller covariance matrix `C = X.T @ X`, which has dimensions `M x M`.
    2.  It then performs eigenvalue decomposition on `C` to find its eigenvalues and eigenvectors.
    3.  The singular values of `X` are the square roots of the eigenvalues of `C`.
    4.  The right singular vectors (`Vt`) of `X` are directly the eigenvectors of `C`.
    5.  The left singular vectors (`U`) of `X` are then derived by multiplying `X` with `V` and scaling by the inverse of the singular value matrix (`U = X @ V @ S_inv`).
* **Efficiency:** This approach significantly reduces the computational cost from `O(N*M^2)` for direct SVD to `O(M^3)` for the eigenvalue decomposition of `C`, followed by `O(N*M^2)` for calculating `U`. When `N` is much larger than `M`, this provides a substantial speedup.
* **Implementation:** The `snapshot_svd(data)` function in the notebook implements this method.

### 3. Randomized Method
The Randomized SVD is an approximate SVD algorithm specifically designed to handle extremely large matrices that are too big for traditional SVD methods to be computationally feasible. It sacrifices a small amount of accuracy for a massive gain in speed.

* **How it works (High-Level):**
    1.  **Random Projection:** Instead of working with the full data matrix `A` (of size `N x M`), a smaller, random projection matrix `P` (of size `M x k`, where `k` is typically `r + p` for target rank `r` and oversampling parameter `p`) is used.
    2.  **Form a Smaller Matrix:** The original matrix `A` is multiplied by this random projection matrix `P` to create a much smaller "sketch" or "range matrix" `Z = A @ P` (of size `N x k`). The column space of `Z` is a good approximation of the column space of `A`.
    3.  **QR Decomposition:** A QR decomposition is performed on `Z` to obtain an orthonormal basis `Q` for the column space of `Z`. So, `Z = Q @ R`.
    4.  **SVD on Projected Data:** A smaller matrix `Y = Q.T @ A` (of size `k x M`) is formed. SVD is then performed on this smaller matrix `Y` to obtain `Uy`, `Sy`, `Vty`.
    5.  **Reconstruction:** The full `U` matrix (approximate) for the original matrix `A` is then reconstructed as `U = Q @ Uy`. The singular values `S` and `Vt` are taken directly from the SVD of `Y`.
* **Efficiency:** The primary computational cost shifts to operations on much smaller matrices (`Z`, `Y`), making it significantly faster than exact SVD for large `N` and `M`.
* **Trade-offs:** It provides an approximation, meaning the results are not exact, but the error can be controlled by parameters like the oversampling `p`. It's ideal when a near-optimal low-rank approximation is sufficient.
* **Implementation:** The `randomized_q(data, r)` function in the notebook demonstrates this technique, where `r` specifies the target rank of the approximation. It also includes a calculation of the loss (error) between the singular values obtained by the randomized method and the regular SVD.

## Results and Observations

The analysis demonstrates that as a matrix is recursively decomposed through dyadic summation, its Frobenius norm systematically decreases and tends towards zero. This behavior provides a quantifiable measure of the diminishing information content or "importance" within the matrix components at deeper levels of decomposition.

The benchmarking results for the different SVD computation methods show the trade-offs between accuracy and computational time. The plot of time vs. number of elements illustrates how "Regular SVD", "Snapshot method", and "Randomized method" scale differently with increasing matrix size, highlighting scenarios where each method offers advantages. The Snapshot method is notably efficient for certain matrix dimensions, and the Randomized method provides a good balance for large-scale approximations.

The calculated optimal hard thresholds provide concrete values for filtering out noise in singular values, improving the quality of low-rank approximations.

## Potential Applications and Future Work

* **Matrix Reconstruction and Data Compression**: The insights from recursive dyadic sums and thresholding can lead to more efficient and specialized methods for reconstructing original matrices from their compressed SVD components and developing advanced data compression algorithms.
* **Noise Reduction**: The thresholding techniques are directly applicable to denoising datasets by effectively distinguishing between signal and noise components in the singular values.
* **Big Data Analytics**: The benchmarked efficient SVD methods (Snapshot and Randomized) are critical for handling and processing large-scale datasets where traditional SVD might be computationally prohibitive.
* **Signal Processing**: The "life signal" concept and singular value thresholding could have implications for analyzing and filtering signals where components of varying importance need to be identified and processed.
* **Further Optimization**: Continued exploration into optimizing the numerical integration for unknown noise scenarios and refining the randomized SVD parameters could further enhance performance.

## Setup and Usage

To run the Jupyter notebook and explore the project:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Ensure you have Jupyter Notebook installed.** If not, you can install it via pip:
    ```bash
    pip install jupyter
    ```
3.  **Install necessary Python libraries:**
    ```bash
    pip install numpy matplotlib scipy scikit-image
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  **Open `svd_efficiency.ipynb`** in your browser.
6.  Run the cells sequentially to reproduce the analysis.

## Dependencies

* Python 3.x
* `numpy`
* `matplotlib`
* `scipy` (for `integrate` and `fsolve`)
* `scikit-image` (for `imread`, if images are processed in your notebook)

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please open an issue or submit a pull request.

