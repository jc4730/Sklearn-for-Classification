# Sklearn for Classification
## Spec

This repository is to use the support vector classifiers in the sklearn package to learn a classification model for a chessboard-like dataset. The input3.csv contains a series of data points.

SVM is used with different kernels to build a classifier. Data is split into training (60%) and testing(40%). Stratified sampling (i.e. same ratio of positive to negative in both the training and testing datasets) is used. Cross validation (with the number of folds k = 5) is used instead of a validation set.

In addition to SVM, Logistic Regression, k-NN, Decision Trees and Random Forests are used for benchmarking. Below is the list of models and parameters used for this project.

### SVM with Linear Kernel
C = [0.1, 0.5, 1, 5, 10, 50, 100]


### SVM with Polynomial Kernel
C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 1]


### SVM with RBF Kernel
C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10]


### Logistic Regression
C = [0.1, 0.5, 1, 5, 10, 50, 100]


### k-Nearest Neighbors
n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60]


### Decision Trees
max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10]


### Random Forest
max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10]
