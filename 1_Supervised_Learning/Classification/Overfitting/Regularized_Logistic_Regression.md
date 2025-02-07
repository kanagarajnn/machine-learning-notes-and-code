# Regularized Logistic Regression

## Introduction
Regularized logistic regression is a powerful technique that prevents overfitting when dealing with models containing **many features**. This ensures that the model generalizes well to new data rather than memorizing training data.

---

## Why Regularization is Important
- Logistic regression can **overfit** when trained on **high-order polynomial features**.
- Regularization helps prevent this by **penalizing large parameter values**, ensuring the decision boundary remains **smooth and generalizable**.

### Real-World Analogy
Think of regularization as **budgeting for groceries**:
- **No regularization**: You buy everything, even unnecessary snacks (overfitting to noise).
- **With regularization**: You focus on essentials, avoiding waste (preventing overfitting by prioritizing useful features).

---

## Regularized Cost Function
The cost function for logistic regression is modified by adding a **regularization term**:

\[
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \log(f(x^{(i)})) - (1 - y^{(i)}) \log(1 - f(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
\]

where:
- **\( \lambda \)**: Regularization parameter (controls penalty strength).
- **If \( \lambda \) is too high** → The model underfits (too simple).
- **If \( \lambda \) is too low** → The model overfits (too complex).

### Effect on Decision Boundaries
- **No Regularization** → Complex, wiggly boundary that memorizes training data.
- **With Regularization** → Smoother, more generalizable boundary that works for new examples.

---

## Regularized Gradient Descent Update
Gradient descent for **regularized logistic regression** follows the same update rule as linear regression, except that **weights (w) are penalized**:

\[
 w_j = w_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right)
\]
\[
 b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})
\]

- The additional **\( \frac{\lambda}{m} w_j \)** term **shrinks weights**, preventing them from growing too large.
- **Bias term (b) is not regularized** because it does not contribute to overfitting.

### Industry Example
- **Fraud Detection (Banks like PayPal & Visa)**: Regularization helps models focus on **real fraud signals** while ignoring noise.
- **Medical Diagnosis (AI in Healthcare)**: Regularization ensures models don’t overfit to irrelevant patient features.

---

## Implementing Regularized Logistic Regression in Python

```python
import numpy as np

def compute_cost_with_regularization(X, y, w, b, lambda_):
    m = len(y)
    f_x = 1 / (1 + np.exp(-np.dot(X, w) - b))  # Sigmoid function
    loss = -y * np.log(f_x) - (1 - y) * np.log(1 - f_x)
    cost = np.sum(loss) / m + (lambda_ / (2 * m)) * np.sum(w**2)
    return cost

def gradient_descent_with_regularization(X, y, w, b, alpha, lambda_, num_iters):
    m = len(y)
    for _ in range(num_iters):
        f_x = 1 / (1 + np.exp(-np.dot(X, w) - b))
        dw = (1/m) * np.dot(X.T, (f_x - y)) + (lambda_/m) * w
        db = (1/m) * np.sum(f_x - y)
        w -= alpha * dw
        b -= alpha * db
    return w, b
```

---

## Key Takeaways
- **Regularization penalizes large weights**, making models more generalizable.
- **Gradient descent remains similar**, except that weight updates include a penalty term.
- **Choosing the right \( \lambda \)** is critical for avoiding underfitting or overfitting.
- **Real-world applications**: Fraud detection, healthcare diagnostics, spam classification, and search ranking.

---

## Next Section
- [To be Added Soon]
