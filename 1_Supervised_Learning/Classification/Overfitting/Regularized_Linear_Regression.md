# Regularized Linear Regression

## Introduction
Regularization helps prevent **overfitting** in linear regression by adding a penalty to large parameter values. In this section, we will discuss how to implement **gradient descent with regularization** to improve the generalization of our models.

---

## 1. Understanding the Regularized Cost Function

The regularized cost function for **linear regression** consists of two parts:
- **Squared error term**: Measures how well the model fits the training data.
- **Regularization term**: Penalizes large values of **w** to reduce overfitting.

```
J(w, b) = (1 / 2m) * Σ (f(x) - y)² + (λ / 2m) * Σ w_j²
```

where:
- **λ** (lambda) is the **regularization parameter**.
- **Large λ** shrinks weights aggressively (risk of underfitting).
- **Small λ** allows larger weights (risk of overfitting).

### Real-World Analogy
Imagine you're **packing a suitcase**:
- If you pack **too much** (overfitting), the suitcase becomes too heavy and inefficient.
- If you pack **too little** (underfitting), you miss essential items.
- **Regularization ensures you carry only the necessary items**, keeping the model simple yet effective.

---

## 2. Updating Parameters with Regularized Gradient Descent
Gradient descent adjusts parameters iteratively to **minimize the cost function**.

### **Gradient Descent Without Regularization**
```
w_j = w_j - α * (1/m) * Σ (f(x) - y) * x_j
b = b - α * (1/m) * Σ (f(x) - y)
```

### **Gradient Descent With Regularization**
```
w_j = w_j - α * ((1/m) * Σ (f(x) - y) * x_j + (λ/m) * w_j)
```

- The additional `(λ/m) * w_j` term **shrinks** the weights slightly with each update.
- **Note**: The **bias term b is NOT regularized** since it doesn’t contribute to overfitting.

---

## 3. What Does Regularization Actually Do?
Regularization gradually reduces large parameter values.

If we rewrite the update rule:
```
w_j = w_j * (1 - α * λ/m) - α * gradient term
```

- The term **(1 - small value)** causes **w_j to shrink slightly in every iteration**.
- This prevents over-reliance on individual features.

### Example: House Price Prediction
- Without regularization, the model might focus too much on **irrelevant features** (e.g., number of swimming pools).
- Regularization forces the model to **prioritize important features** (e.g., square footage, number of bedrooms).

---

## 4. Choosing the Right **λ**
Selecting **λ** correctly is essential:

| **λ Value** | **Effect** |
|--------------|----------------------|
| **Too small** | Overfitting (complex model) |
| **Too large** | Underfitting (simple model) |
| **Optimal** | Best generalization |

### Real-World Example
- **Amazon's recommendation system** uses regularization to avoid overfitting to niche buying patterns.
- **Google Maps ETA prediction** applies regularization to improve route time estimation by prioritizing key traffic features.

---

## 5. Implementation in Code
Here’s how we implement **regularized gradient descent** in Python using NumPy:

```python
# Regularized gradient descent for linear regression
w = np.zeros(n)  # Initialize weights
b = 0  # Initialize bias
alpha = 0.01  # Learning rate
lambda_ = 0.1  # Regularization parameter
m = len(y)  # Number of training examples

for _ in range(num_iterations):
    f_x = np.dot(X, w) + b  # Model prediction
    gradient_w = (1/m) * np.dot(X.T, (f_x - y)) + (lambda_/m) * w
    gradient_b = (1/m) * np.sum(f_x - y)
    
    w -= alpha * gradient_w
    b -= alpha * gradient_b
```

---

## Summary
- **Regularization reduces overfitting** by shrinking large weights.
- **Gradient descent updates remain similar**, but an additional **regularization term** is included.
- **Choosing the right λ ensures the best model performance**.
- Regularization is widely used in **finance, e-commerce, and self-driving technology** to prevent overfitting.

---

## Next Section
  - ### [Regularized Logistic Regression](Regularized_Logistic_Regression.md)

