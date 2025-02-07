# Gradient Descent for Multiple Linear Regression

## Introduction
Now that we've explored **gradient descent**, **multiple linear regression**, and **vectorization**, it's time to combine these concepts to efficiently implement gradient descent for **multiple linear regression**.

## Recap: Multiple Linear Regression in Vector Form
Instead of treating each weight **w** as a separate parameter, we collect them into a **vector w**. This allows us to express our prediction function more concisely:

\[
 f(w, b, X) = w \cdot X + b
\]

where:
- **w**: A vector of weights (one for each feature)
- **X**: A vector of input features
- **b**: A scalar bias term

Using **vectorized notation** helps optimize computations and simplifies implementation in machine learning frameworks like NumPy.

## Gradient Descent for Multiple Linear Regression
The cost function \( J(w, b) \) measures the difference between the predicted and actual values. To minimize this cost function, gradient descent updates **w** and **b** iteratively:

\[
 w_j := w_j - \alpha \frac{d}{dw_j} J(w, b)
\]

\[
 b := b - \alpha \frac{d}{db} J(w, b)
\]

where **\( \alpha \)** is the learning rate.

### Understanding the Update Process
- The formula for **one feature** remains similar when extended to **multiple features**.
- The **error term** (\( f(X) - Y \)) is computed across all training examples.
- Each **w** is updated individually for all features **X_1, X_2, ... X_n**.

## Implementation in Python (Vectorized Form)
Instead of using a loop, we can **vectorize** gradient descent to optimize performance:
```python
import numpy as np

def gradient_descent(X, y, w, b, alpha, num_iterations):
    m = X.shape[0]  # Number of training examples
    for _ in range(num_iterations):
        predictions = np.dot(X, w) + b
        errors = predictions - y
        w -= (alpha / m) * np.dot(X.T, errors)
        b -= (alpha / m) * np.sum(errors)
    return w, b
```
- **np.dot(X, w) + b** computes predictions efficiently.
- **np.dot(X.T, errors)** computes the gradient for all weights simultaneously.
- This approach scales well with large datasets.

## Alternative Approach: The Normal Equation
Gradient descent is not the only way to find **w** and **b**. The **Normal Equation** provides an explicit solution for linear regression:

\[
 w = (X^T X)^{-1} X^T Y
\]

### Pros and Cons of the Normal Equation
- **Advantage**: No need for iterative optimization.
- **Disadvantage**: Computationally expensive for large feature sets.

Most machine learning libraries (e.g., Scikit-Learn’s `LinearRegression`) use the **Normal Equation** behind the scenes when solving for regression coefficients.

## Real-World Applications
- **Finance**: Predicting stock prices using multiple economic indicators.
- **Healthcare**: Estimating disease risk based on multiple patient parameters.
- **E-commerce**: Amazon and Netflix use multiple linear regression to predict customer preferences based on historical data.
- **Self-driving cars**: Tesla’s autopilot system adjusts vehicle behavior using multiple sensor inputs.

## Key Takeaways
- **Gradient descent** allows iterative updates for finding optimal values of **w** and **b**.
- **Vectorization** improves efficiency when handling multiple features.
- **The Normal Equation** offers a direct mathematical solution but has scalability limitations.
- **Multiple linear regression** is widely used across industries for predictive modeling.

---
## Next Section
  - ### [Feature Scaling Part 1](../Gradient_in_Practice/Feature_Scaling_Part_1.md)
