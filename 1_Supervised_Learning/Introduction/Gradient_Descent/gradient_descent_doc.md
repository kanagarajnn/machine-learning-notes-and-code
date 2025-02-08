# Gradient Descent Algorithm Implementation

## Description
This module implements the **Gradient Descent** algorithm, an essential optimization technique used in machine learning and deep learning.

### What is Gradient Descent?
- Gradient Descent is used to iteratively update model parameters (weights `w` and bias `b`) to minimize the **cost function** and improve predictions.
- It is widely used in **Linear Regression, Neural Networks, Logistic Regression**, and other machine learning models where optimization is required.

### Why Gradient Descent?
- When training machine learning models, manually adjusting parameters is impractical.
- Gradient Descent automates parameter tuning by **computing the slope of the cost function** and updating parameters accordingly.
- Without an efficient optimization method, models may fail to converge or take excessive time to learn.

## Features
- **Computes the Cost Function**: Evaluates how well the model fits the data.
- **Implements Gradient Computation**: Calculates the derivatives (gradients) needed for optimization.
- **Performs Gradient Descent**: Adjusts `w` and `b` in small steps to **reduce errors over time**.
- **Visualizes Cost Function Convergence**: Generates a plot to show how the cost function decreases with each iteration.
- **Supports Different Learning Rates**: Experiment with different step sizes (alpha) to see how it affects convergence.

## Real-World Applications
- **Finance**: Predicting stock prices using regression models.
- **Healthcare**: Training models to detect diseases from medical images.
- **E-commerce**: Optimizing recommendation algorithms for better product suggestions.
- **Autonomous Vehicles**: Tuning AI models to improve self-driving car navigation.

## Usage
- Run as a standalone script to see Gradient Descent in action:
    ```bash
    python gradient_descent.py
    ```
- Import as a module to integrate with other machine learning models:
    ```python
    from gradient_descent import gradient_descent
    optimized_w, optimized_b, cost_history = gradient_descent(X, y, w, b, alpha, num_iters)
    ```

## Metadata
- **Author**: Kanagaraj N N
- **Date**: February 8, 2025
- **Version**: 2.0
- **License**: MIT (see LICENSE file for details)

---

## Python Code Implementation

### Compute Cost Function
```python
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, w, b):
    """
    Compute the cost function (Mean Squared Error) for linear regression.
    """
    m = X.shape[0]  # Number of training examples
    cost = 0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Compute prediction
        cost += (f_wb - y[i]) ** 2  # Compute squared error
    
    cost = cost / (2 * m)  # Average cost over all training examples
    return cost
```

### Compute Gradient
```python
def compute_gradient(X, y, w, b):
    """
    Compute the gradient (partial derivatives) of the cost function.
    """
    m, n = X.shape  # Get number of examples (m) and features (n)
    dj_dw = np.zeros(n)  # Initialize gradient for weights
    dj_db = 0  # Initialize gradient for bias

    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Compute prediction
        err = f_wb - y[i]  # Compute error
        
        dj_db += err  # Accumulate bias gradient
        for j in range(n):
            dj_dw[j] += err * X[i, j]  # Compute weight gradient
    
    dj_dw /= m  # Average gradients
    dj_db /= m
    return dj_dw, dj_db
```

### Gradient Descent Optimization Algorithm
```python
def gradient_descent(X, y, w, b, alpha, num_iters):
    """
    Perform gradient descent to optimize parameters.
    """
    J_history = []  # Store cost function values over iterations

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)  # Compute gradients

        w -= alpha * dj_dw  # Update weights
        b -= alpha * dj_db  # Update bias
        
        J_history.append(compute_cost(X, y, w, b))  # Store cost function value
    
    return w, b, J_history
```

### Plot Cost Function Over Iterations
```python
def plot_cost(J_history):
    """
    Plot the cost function over gradient descent iterations.
    """
    plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("Cost Function Convergence")
    plt.show()
```

### Example Usage
```python
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
```

---

