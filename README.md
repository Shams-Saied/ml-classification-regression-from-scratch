# Alexandria University | Faculty of Engineering

## Introduction to Machine Learning — Assignment 1

This repository contains the complete implementation and analysis for **Assignment 1**, developed in a Python Jupyter Notebook environment. The project explores foundational supervised learning workflows across classification and regression domains, contrasting custom, from-scratch mathematical implementations with Scikit-Learn benchmarks.

---

## 👥 Team & Environment Details

- **Course Instructor:** Dr. Marwan Torki
- **Teaching Assistant:** Eng. Youssef El-Ebiary
- **Environment Requirement:** Python `3.13.5`
- **Academic Integrity:** Built natively using explicit mathematical implementations and core library optimizations. No code-generation tools or external platform distributions were utilized during development.

---

## 🛠️ Key Architectural Features

### 📊 Task 1: KNN Classification (MAGIC Gamma Telescope)

The objective is to classify simulated high-energy gamma particles into signal (`g`) and background (`h`) classes.

- **Class Imbalance Handling:** Calculated an original imbalance margin of **5644** entries. Using a deterministic random seed (`69`), we downsampled the majority class (`g`) to create perfectly balanced distributions.
- **Deterministic Stratified Contiguous Splitting:** Sectioned datasets manually using precise contiguous indexing parameters to enforce exact proportional splits (70% Train / 15% Val / 15% Test) across both target classes.
- **Custom Array-Broadcasting KNN:** Implemented a scratch-built KNN classifier utilizing NumPy array broadcasting for fast Euclidean squared-distance computation alongside vectorized top-k processing via `np.argpartition`.
- **Hyperparameter Convergence Parity:** Tuned $k$ across a range of 1 to 149 (stepping by 2). Both our scratch-built algorithm and Scikit-Learn's `KNeighborsClassifier` reached unified minimum validation errors at **$k = 51$**.

#### Final Performance Metrics ($k = 51$):

| Metric               |   Scratch Implementation   |   Scikit-Learn Framework   |
| :------------------- | :------------------------: | :------------------------: |
| **Accuracy**         |           77.54%           |           77.54%           |
| **Precision (`g`)**  |           73.06%           |           73.06%           |
| **Recall (`g`)**     |           87.25%           |           87.25%           |
| **F1-Score**         |           79.53%           |           79.53%           |
| **Confusion Matrix** | `[[876, 128], [323, 681]]` | `[[876, 128], [323, 681]]` |

---

### 🏠 Task 2: Linear Regression (California Housing Prices)

Predicts `Median_House_Value` using structural, geographic, and demographic inputs via matrix calculations and iterative training loops.

- **Feature Standardization:** Integrated `StandardScaler` to uniformize varying numerical distributions, followed by programmatic bias-column stacking ($X_0 = 1$) to ensure seamless matrix dot-products.
- **Analytical Closed-Form Solver:** Implemented standard Linear and Ridge solutions using the normal equation: $w = (X^T X)^{-1} X^T y$
- **Iterative Matrix Optimization:** Built an optimization loop from scratch to update weights incrementally based on the standard regression gradient vector.
- **Non-Analytical Regularization ($L_1$ Penalty):** Handled Lasso parameters by applying structural gradient descent updates via sign vectors: $\Delta w = \alpha \cdot (\text{grad} + \lambda \cdot \text{sign}(w))$
- **Hyperparameter Validation Sweeps:** Plotted empirical changes using log-spaced arrays across a $\lambda$ window ($[10^{-5}, 10^{10}]$) to inspect the exact regularization threshold behaviors of Ridge and Lasso models.

#### Model Comparison Matrix (Test Data Errors):

| Model Variant         | Scikit-Learn MSE | Manual/Scratch MSE | Scikit-Learn MAE | Manual/Scratch MAE |
| :-------------------- | :--------------: | :----------------: | :--------------: | :----------------: |
| **Linear Regression** |  4400953150.61   |   4400953150.61    |     48782.03     |      48782.03      |
| **Ridge Regression**  |  4400538182.34   |   4400582047.51    |     48784.32     |      48820.89      |
| **Lasso Regression**  |  4400942412.81   |   4407404667.04    |     48782.08     |      48990.92      |

---
