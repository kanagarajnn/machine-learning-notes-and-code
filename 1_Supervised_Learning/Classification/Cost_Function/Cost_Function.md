# Cost Function for Logistic Regression

## Introduction
The **cost function** is a crucial component of logistic regression, helping us determine how well our model is performing. Unlike linear regression, which uses the **squared error cost function**, logistic regression requires a different approach to ensure **convexity** and **proper convergence**.

---

## Why Not Use Squared Error Cost Function?
- In **linear regression**, the cost function is based on the **mean squared error (MSE)**:
  \[
  J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2
  \]
- This works well for **continuous outputs** but **not for classification**.
- If we try to apply this to **logistic regression**, we get a **non-convex cost function**, meaning gradient descent may **get stuck in local minima**, leading to poor optimization.

### Real-World Analogy
Imagine you are **hiking down a mountain**. If the landscape is **smooth (convex function)**, you will always reach the lowest valley. However, if the terrain has **multiple dips and bumps (non-convex function)**, you might get stuck in a smaller dip instead of reaching the actual lowest point.

---

## The Logistic Regression Cost Function
To ensure proper convergence, we use the **log loss (logarithmic loss) function**, which is:
\[
J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f(x^{(i)})) + (1 - y^{(i)}) \log(1 - f(x^{(i)})) \right]
\]
where:
- **f(x) = sigmoid(wX + b)** → This ensures the output is between 0 and 1.
- **log(f(x))** → Penalizes incorrect predictions.
- **log(1 - f(x))** → Ensures the cost is high when the model is confident but incorrect.

---

## Understanding the Loss Function
The loss function tells us how **bad** a single prediction is:
- **If y = 1**, the loss is:
  \[
  L(f(x), y) = -\log(f(x))
  \]
  - If **f(x) is close to 1**, loss is **small** (good prediction).
  - If **f(x) is close to 0**, loss is **high** (bad prediction).

- **If y = 0**, the loss is:
  \[
  L(f(x), y) = -\log(1 - f(x))
  \]
  - If **f(x) is close to 0**, loss is **small** (good prediction).
  - If **f(x) is close to 1**, loss is **high** (bad prediction).

### Example: Tumor Classification
- If a tumor is **malignant (y=1)** and the model predicts **f(x) = 0.9**, the loss is small.
- If a tumor is **malignant (y=1)** and the model predicts **f(x) = 0.1**, the loss is large.
- If a tumor is **benign (y=0)** and the model predicts **f(x) = 0.9**, the loss is large.

### Why This Works Better Than Squared Error
- **Avoids non-convexity**, ensuring gradient descent reliably converges.
- **Punishes overconfident incorrect predictions** more than squared error.

---

## Gradient Descent and Optimization
- The goal is to **minimize J(w, b)** using **gradient descent**.
- Unlike linear regression, the updates use:
  \[
  w = w - \alpha \frac{d}{dw} J(w, b)
  \]
  \[
  b = b - \alpha \frac{d}{db} J(w, b)
  \]
- This guarantees **global convergence** because the log-loss function is convex.

---

## Summary
- **Squared error loss is not ideal for logistic regression** because it creates a non-convex cost function.
- **Log-loss ensures convexity**, allowing gradient descent to converge efficiently.
- **Minimizing log-loss** improves classification accuracy by reducing incorrect confident predictions.

---

## Next Section
- ### [Simplified Cost Function](Simplified_Cost_Function.md)
