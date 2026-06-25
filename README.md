# Machine Learning Fundamentals: Classification & Regression

## Overview

This project was developed for the **Introduction to Machine Learning** course at **Alexandria University, Faculty of Engineering**.

The assignment focuses on two fundamental machine learning problems:

1. **Classification** using the MAGIC Gamma Telescope Dataset and K-Nearest Neighbors (K-NN).
2. **Regression** using the California Housing Prices Dataset with Linear, Ridge, and Lasso Regression.

Both tasks were implemented **from scratch using NumPy** and then reimplemented using **Scikit-Learn** for comparison and validation.

---

## Objectives

- Understand classification and regression tasks.
- Implement machine learning algorithms from scratch.
- Perform train, validation, and test splitting.
- Tune model hyperparameters.
- Compare custom implementations with Scikit-Learn models.
- Evaluate model performance using standard metrics.
- Analyze the effects of regularization.

---

# Task 1: Classification with K-Nearest Neighbors (K-NN)

## Dataset

### MAGIC Gamma Telescope Dataset

The dataset contains telescope observations classified into:

- **Gamma (g)** → Signal events
- **Hadron (h)** → Background events

Original dataset distribution:

| Class | Samples |
|---------|---------|
| Gamma | 12,332 |
| Hadron | 6,688 |

Since the dataset is imbalanced, the gamma class was randomly undersampled to create a balanced dataset.

---

## Data Preprocessing

### Class Balancing

- Randomly removed excess gamma samples.
- Fixed random seed used for reproducibility.

### Data Splitting

The dataset was divided into:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

Each subset contains equal numbers of both classes.

---

## Manual K-NN Implementation

Implemented entirely from scratch using NumPy.

### Features Implemented

- Euclidean Distance Calculation
- K-Nearest Neighbor Search
- Majority Voting Classification

### Hyperparameter Tuning

Tested odd values of k:

```python
k = 1, 3, 5, ..., 149
```

Validation error was calculated for every value.

### Best Result

```text
Best k = 51
Minimum Validation Error = 22.48%
```

---

## Manual K-NN Test Performance

| Metric | Value |
|----------|----------|
| Accuracy | 77.54% |
| Precision | 73.06% |
| Recall | 87.25% |
| F1 Score | 79.53% |

### Confusion Matrix

```text
[[876 128]
 [323 681]]
```

---

## Scikit-Learn K-NN

Implemented using:

```python
from sklearn.neighbors import KNeighborsClassifier
```

### Best Result

```text
Best k = 51
Validation Accuracy = 77.52%
```

### Test Performance

| Metric | Value |
|----------|----------|
| Accuracy | 77.54% |
| Precision | 73.06% |
| Recall | 87.25% |
| F1 Score | 79.53% |

### Confusion Matrix

```text
[[876 128]
 [323 681]]
```

---

## Classification Analysis

### Observations

- Manual and Scikit-Learn implementations achieved nearly identical results.
- Small k values showed overfitting behavior.
- Very large k values led to underfitting.
- The optimal value was found around k = 51.
- The custom implementation successfully replicated the behavior of Scikit-Learn's K-NN classifier.

---

# Task 2: Regression on California Housing Prices

## Dataset

### California Housing Prices Dataset

The objective is to predict:

```text
Median_House_Value
```

using demographic and geographic features such as:

- Median Income
- Median Age
- Population
- Total Rooms
- Total Bedrooms
- Latitude
- Longitude
- Distance to Coast
- Distance to Major California Cities

---

## Data Preprocessing

### Feature Scaling

All features were standardized using:

```python
StandardScaler()
```

### Data Splitting

The dataset was split into:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

---

## Models Implemented

### From Scratch

#### Linear Regression

Implemented using:

##### Normal Equation

\[
w = (X^TX)^{-1}X^Ty
\]

##### Gradient Descent

Used as an alternative optimization method.

---

#### Ridge Regression

Implemented using L2 Regularization:

\[
w = (X^TX + \lambda I)^{-1}X^Ty
\]

---

#### Lasso Regression

Implemented using Gradient Descent with L1 Regularization:

\[
Loss = MSE + \lambda |w|
\]

---

## Hyperparameter Tuning

For Ridge and Lasso Regression:

```python
lambdas = np.logspace(-5, 10, 50)
```

Validation Mean Squared Error (MSE) was evaluated across different regularization strengths.

---

## Scikit-Learn Implementations

Implemented using:

```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```

---

## Results

### Linear Regression

| Implementation | MSE | MAE |
|---------------|---------------|---------------|
| Manual | 4,400,953,150.61 | 48,782.03 |
| Scikit-Learn | 4,400,953,150.61 | 48,782.03 |

---

### Ridge Regression

| Implementation | MSE | MAE |
|---------------|---------------|---------------|
| Manual | 4,400,582,047.51 | 48,820.89 |
| Scikit-Learn | 4,400,538,182.34 | 48,784.32 |

---

### Lasso Regression

| Implementation | MSE | MAE |
|---------------|---------------|---------------|
| Manual | 4,407,404,667.04 | 48,990.92 |
| Scikit-Learn | 4,400,942,412.81 | 48,782.08 |

---

## Regression Analysis

### Observations

- Manual Linear Regression matched Scikit-Learn exactly.
- Ridge Regression slightly improved generalization through L2 regularization.
- Lasso Regression achieved comparable performance while encouraging sparse solutions.
- Manual implementations closely matched Scikit-Learn outputs, validating the correctness of the mathematical implementations.

---

## Normal Equation vs Gradient Descent

### Normal Equation

**Advantages**

- Produces an exact analytical solution.
- No learning rate tuning required.
- Efficient for small and medium-sized datasets.

**Limitations**

- Computationally expensive for high-dimensional data.

### Gradient Descent

**Advantages**

- Scales well to large datasets.
- Suitable for iterative optimization.

**Limitations**

- Requires learning rate tuning.
- May converge slowly if hyperparameters are poorly chosen.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

# Repository Structure

```text
├── Classification/
│   ├── manual_knn.ipynb
│   ├── sklearn_knn.ipynb
│   └── telescope_data.csv
│
├── Regression/
│   ├── manual_regression.ipynb
│   ├── sklearn_regression.ipynb
│   └── California_Houses.csv
│
├── README.md
└── requirements.txt
```

---

# Key Learning Outcomes

- Implemented K-NN classification from scratch.
- Built Linear, Ridge, and Lasso Regression using matrix operations.
- Applied Gradient Descent optimization.
- Performed hyperparameter tuning using validation data.
- Evaluated classification and regression models using standard metrics.
- Compared custom implementations against Scikit-Learn.
- Gained practical experience with regularization techniques and model evaluation.

---

## Course Information

**Course:** Introduction to Machine Learning  
**Institution:** Alexandria University – Faculty of Engineering  
**Instructor:** Dr. Marwan Torki  
**Teaching Assistant:** Eng. Youssef El-Ebiary

**Assignment:** Classification and Regression Fundamentals using NumPy and Scikit-Learn
