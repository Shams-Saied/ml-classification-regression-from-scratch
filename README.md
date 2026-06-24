# Alexandria University | Faculty of Engineering

## Introduction to Machine Learning — Assignment 1

### Submission Details

- **Instructor:** Dr. Marwan Torki
- **TA:** Eng. Youssef El-Ebiary
- **Environment:** Python 3.13.5
- **Project Deliverables:** Jupyter Notebook (.ipynb) and exported PDF wrapped in a single zip file following the `id_assignment.zip` naming format.

---

### Task 1: KNN Classification (MAGIC Gamma Telescope)

This task focuses on balancing a data set of high-energy gamma particles and classifying them into signal (`g`) or background (`h`).

- **Data Preprocessing:** Identifed an initial imbalance of 5644 records. Used a fixed random seed (`69`) to drop excess majority class records to ensure a 1:1 ratio.
- **Data Splitting:** Divided the data manually via contiguous slicing into exactly 70% training, 15% validation, and 15% testing splits.
- **Scratch Implementation:** Implemented Euclidean distance calculation, nearest neighbor sorting using `np.argpartition`, and classification using a pandas majority vote mechanism.
- **Scikit-Learn Verification:** Evaluated code against standard `KNeighborsClassifier`. Both pipelines identified **k = 51** as the optimal hyperparameter value.

#### Final Classification Performance (k = 51)

- **Accuracy:** 77.54%
- **Precision:** 73.06%
- **Recall:** 87.25%
- **F1 Score:** 79.53%
- **Confusion Matrix:** ```text
  [[876 128]
 [323 681]]
