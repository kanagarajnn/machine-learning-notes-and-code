"""
gradient_descent.py - Implementation of the Gradient Descent Algorithm in Python

Description:
- This module implements the **Gradient Descent** algorithm, an essential optimization 
  technique used in machine learning and deep learning.
- Gradient Descent is used to iteratively update model parameters (weights `w` and bias `b`) 
  to minimize the **cost function** and improve predictions.
- It is widely used in **Linear Regression, Neural Networks, Logistic Regression**, and other 
  machine learning models where optimization is required.

Why Gradient Descent?
- When training machine learning models, manually adjusting parameters is impractical.
- Gradient Descent automates parameter tuning by **computing the slope of the cost function** 
  and updating parameters accordingly.
- Without an efficient optimization method, models may fail to converge or take excessive 
  time to learn.

Features:
- **Computes the Cost Function**: Evaluates how well the model fits the data.
- **Implements Gradient Computation**: Calculates the derivatives (gradients) needed for optimization.
- **Performs Gradient Descent**: Adjusts `w` and `b` in small steps to **reduce errors over time**.
- **Visualizes Cost Function Convergence**: Generates a plot to show how the cost function 
  decreases with each iteration.
- **Supports Different Learning Rates**: Experiment with different step sizes (alpha) 
  to see how it affects convergence.

Real-World Applications:
- **Finance**: Predicting stock prices using regression models.
- **Healthcare**: Training models to detect diseases from medical images.
- **E-commerce**: Optimizing recommendation algorithms for better product suggestions.
- **Autonomous Vehicles**: Tuning AI models to improve self-driving car navigation.

Usage:
- Run as a standalone script to see Gradient Descent in action:
    ```bash
    python gradient_descent.py
    ```
- Import as a module to integrate with other machine learning models:
    ```python
    from gradient_descent import gradient_descent
    optimized_w, optimized_b, cost_history = gradient_descent(X, y, w, b, alpha, num_iters)
    ```

Metadata:
- Author: Kanagaraj N N
- Date: February 8, 2025
- Version: 2.0
- License: MIT (see LICENSE file for details)
"""


import numpy as np
import matplotlib.pyplot as plt


# Compute Cost Function for Linear Regression
def compute_cost(X, y, w, b):
    """
    Compute the cost function (Mean Squared Error) for linear regression.

    Args:
        X (ndarray): Feature matrix (m examples, n features)
        y (ndarray): Target values (m,)
        w (ndarray): Weight parameters (n,)
        b (float): Bias parameter

    Returns:
        cost (float): Computed cost function value
    """
    m = X.shape[0]  # Number of training examples
    cost = 0

    # Compute sum of squared errors
    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Compute prediction
        cost += (f_wb - y[i]) ** 2  # Compute squared error

    # Average cost over all training examples
    cost = cost / (2 * m)
    return cost


# Compute Gradient for Linear Regression
def compute_gradient(X, y, w, b):
    """
    Compute the gradient (partial derivatives) of the cost function
    with respect to weights (w) and bias (b).

    Args:
        X (ndarray): Feature matrix (m examples, n features)
        y (ndarray): Target values (m,)
        w (ndarray): Weight parameters (n,)
        b (float): Bias parameter

    Returns:
        dj_dw (ndarray): Gradient of cost w.r.t. weights (n,)
        dj_db (float): Gradient of cost w.r.t. bias
    """
    m, n = X.shape  # Get number of examples (m) and features (n)
    dj_dw = np.zeros(n)  # Initialize gradient for weights
    dj_db = 0  # Initialize gradient for bias

    # Compute gradients by iterating over each training example
    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Compute prediction
        err = f_wb - y[i]  # Compute error

        # Accumulate gradients
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i, j]  # Compute gradient for weight w[j]

    # Average gradients over all examples
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# Gradient Descent Optimization Algorithm
def gradient_descent(X, y, w, b, alpha, num_iters):
    """
    Perform gradient descent to optimize parameters.

    Args:
        X (ndarray): Feature matrix (m examples, n features)
        y (ndarray): Target values (m,)
        w (ndarray): Initial weight parameters (n,)
        b (float): Initial bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations

    Returns:
        w (ndarray): Optimized weight parameters (n,)
        b (float): Optimized bias parameter
        J_history (list): Cost function values over iterations
    """
    J_history = []  # Store cost function values over iterations

    for i in range(num_iters):
        # Compute gradients
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        # Update parameters using gradient descent rule
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Store the cost function value for tracking convergence
        J_history.append(compute_cost(X, y, w, b))

    return w, b, J_history


# Function to Plot Cost Function Over Iterations
def plot_cost(J_history):
    """
    Plot the cost function over gradient descent iterations.

    Args:
        J_history (list): List of cost function values
    """
    plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("Cost Function Convergence")
    plt.show()


# Example Usage of Gradient Descent
if __name__ == "__main__":
    # Define training data: Feature matrix X and target values y
    X = np.array([[1], [2], [3], [4], [5]])  # Single feature (column vector)
    y = np.array([2, 3, 4, 5, 6])  # Target values

    # Initialize model parameters
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    b = 0  # Initialize bias to zero
    alpha = 0.01  # Learning rate
    num_iters = 100  # Number of gradient descent iterations

    # Run gradient descent optimization
    w, b, J_history = gradient_descent(X, y, w, b, alpha, num_iters)

    # Print optimized parameters
    print(f"Optimized Parameters: w = {w}, b = {b}")

    # Plot cost function to visualize convergence
    plot_cost(J_history)
