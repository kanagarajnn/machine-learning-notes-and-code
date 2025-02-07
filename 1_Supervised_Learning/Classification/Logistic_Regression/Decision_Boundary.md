# Decision Boundary in Logistic Regression

## Introduction
A **decision boundary** is the line or curve that separates different classes in a **classification problem**. In logistic regression, this boundary helps determine whether the model predicts **class 0 or class 1** based on the input features.

---

## How Logistic Regression Makes Predictions
Logistic regression makes predictions in two steps:
1. **Compute the linear combination of inputs**:
   \[
   z = wX + b
   \]
2. **Apply the Sigmoid function**:
   \[
   f(x) = \frac{1}{1 + e^{-z}}
   \]
   - If **f(x) ≥ 0.5**, predict **y = 1**.
   - If **f(x) < 0.5**, predict **y = 0**.

---

## Understanding the Decision Boundary
- The decision boundary is the region **where the model is uncertain** (where the probability of classifying as 0 or 1 is 50%).
- This happens when:
  \[
  wX + b = 0
  \]
- This forms a **straight line** in the case of two features.

### Example: Email Spam Classification
- If an email’s spam probability is **above 50%**, classify it as spam.
- If it’s **below 50%**, classify it as not spam.
- The decision boundary is the threshold where the probability is exactly **50%**.

---

## Visualizing Decision Boundaries
### 1. **Linear Decision Boundary**
- When using only basic input features (e.g., email word count), the decision boundary is a **straight line**.
- Example: Predicting if a customer will buy a product based on their age and income.

### 2. **Non-Linear Decision Boundary**
- If we introduce **polynomial features** (e.g., x² or x³), we can get **curved boundaries**.
- Example: Predicting whether an image contains a dog or cat based on complex pixel patterns.

---

## Example: Logistic Regression with Two Features
Consider a dataset where we classify data points based on **two features, x₁ and x₂**.
- Red crosses (❌) represent **positive examples (y = 1)**.
- Blue circles (🔵) represent **negative examples (y = 0)**.
- The decision boundary is the point where:
  \[
  w_1 x_1 + w_2 x_2 + b = 0
  \]
  If **w₁ = 1, w₂ = 1, and b = -3**, the decision boundary is:
  \[
  x_1 + x_2 = 3
  \]
  This represents a **straight line** separating the two classes.

---

## More Complex Decision Boundaries
### 1. **Quadratic Boundaries**
- If we introduce **squared terms**, we can model **circular** decision boundaries:
  \[
  x_1^2 + x_2^2 = 1
  \]
  - Example: Predicting if a tumor is malignant based on its **size and roundness**.

### 2. **Higher-Order Boundaries**
- Adding **interaction terms** like **x₁x₂** allows for **elliptical** or more complex shapes.
- Example: Identifying fraud in transactions based on **transaction amount and frequency**.

---

## Summary
- The **decision boundary** determines whether logistic regression predicts **class 0 or 1**.
- **Linear decision boundaries** work well for simple problems.
- **Non-linear decision boundaries** can capture more complex patterns using polynomial features.
- Adding polynomial features allows logistic regression to fit **more complex datasets**.

---

## Next Section
- ### [Cost Function for Logistic Regression](../Cost_Function/Cost_Function.md)
