# Quantum-Kernel-SVM-project
# Quantum Kernel SVM for Classification

## Overview

This project demonstrates the use of Quantum Kernel Estimation (QKE) with a Support Vector Machine (SVM) for classifying data generated using the `make_circles` dataset from scikit-learn. The QKE method leverages a variational quantum circuit to compute a kernel matrix, which is then used by the SVM classifier.

## Installation

To run this project, you need to install the following dependencies:

```sh
pip install pennylane pennylane-lightning numpy scikit-learn scipy joblib matplotlib
Code Explanation
Step 1: Import necessary libraries
The necessary libraries for this project are imported, including PennyLane for quantum computing, scikit-learn for dataset generation and machine learning, and other utilities like joblib and matplotlib for parallel computing and plotting, respectively.

Step 2: Load and visualize the dataset
The make_circles dataset is generated and plotted to visualize the two classes.


from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('make_circles Dataset')
plt.show()
Step 3: Define the quantum feature map and variational circuit
A quantum feature map and variational circuit are defined using PennyLane. The circuit is parameterized and will be optimized to compute the kernel matrix.


import pennylane as qml
from pennylane import numpy as np

num_qubits = X.shape[1]
device = qml.device('default.qubit', wires=num_qubits)

def quantum_feature_map(x, params):
    for i in range(num_qubits):
        qml.RY(x[i % len(x)], wires=i)
    for i in range(num_qubits):
        qml.RZ(params[i], wires=i)
    qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern="ring")

@qml.qnode(device)
def variational_circuit(x, params):
    quantum_feature_map(x, params)
    return qml.state()
Step 4: Compute the Quantum Kernel Matrix (QEK)
The QEK matrix is computed by measuring the overlap between the quantum states of different data points.


from joblib import Parallel, delayed

def compute_qek_matrix(X, params):
    n_samples = len(X)
    qek_matrix = np.zeros((n_samples, n_samples))

    def compute_element(i, j):
        state_i = variational_circuit(X[i], params)
        state_j = variational_circuit(X[j], params)
        return np.abs(np.dot(np.conj(state_i), state_j))**2

    results = Parallel(n_jobs=-1)(delayed(compute_element)(i, j) for i in range(n_samples) for j in range(n_samples))
    qek_matrix = np.array(results).reshape(n_samples, n_samples)

    return qek_matrix

params = np.random.uniform(0, np.pi, num_qubits)
qek_matrix = compute_qek_matrix(X, params)
Step 5: Optimize the quantum kernel parameters
The parameters of the quantum feature map are optimized to maximize the kernel-target alignment.


from scipy.optimize import minimize

def kernel_target_alignment(params, X, y):
    qek_matrix = compute_qek_matrix(X, params)
    y_matrix = np.outer(y, y)
    alignment = np.sum(qek_matrix * y_matrix)
    return -alignment  # Negate for minimization

result = minimize(kernel_target_alignment, params, args=(X, y), method='COBYLA')
optimized_params = result.x
optimized_qek_matrix = compute_qek_matrix(X, optimized_params)
Step 6: Train and evaluate the SVM classifier
An SVM classifier is trained using the optimized QEK matrix, and its performance is evaluated on the test set.


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_indices = np.array([np.where((X == train_instance).all(axis=1))[0][0] for train_instance in X_train])
test_indices = np.array([np.where((X == test_instance).all(axis=1))[0][0] for test_instance in X_test])

K_train = optimized_qek_matrix[np.ix_(train_indices, train_indices)]
K_test = optimized_qek_matrix[np.ix_(test_indices, train_indices)]

svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

accuracy = svm.score(K_test, y_test)
print(f'Classification accuracy: {accuracy:.2f}')
Step 7: Plot the SVM decision boundary
The decision boundary of the SVM classifier with the quantum kernel is plotted.


h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def predict_with_quantum_kernel(X_train, X_new, optimized_qek_matrix):
    n_train = len(X_train)
    n_new = len(X_new)
    K_new = np.zeros((n_new, n_train))

    for i in range(n_new):
        for j in range(n_train):
            state_i = variational_circuit(X_new[i], optimized_params)
            state_j = variational_circuit(X_train[j], optimized_params)
            K_new[i, j] = np.abs(np.dot(np.conj(state_i), state_j))**2

    return K_new

Z = predict_with_quantum_kernel(X_train, np.c_[xx.ravel(), yy.ravel()], optimized_qek_matrix)
Z = svm.predict(Z)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Quantum Kernel (make_circles Dataset)')
plt.show()
Results
The quantum kernel SVM classifier achieved a classification accuracy of approximately accuracy on the make_circles dataset. The decision boundary plot shows the regions classified as different classes by the SVM using the quantum kernel.

Conclusion
This project showcases how to integrate quantum computing with classical machine learning algorithms to perform kernel-based classification. By leveraging the power of quantum feature maps, the SVM classifier can potentially achieve better performance on complex datasets.

References
PennyLane Documentation
scikit-learn Documentation
Quantum Kernel Estimation
