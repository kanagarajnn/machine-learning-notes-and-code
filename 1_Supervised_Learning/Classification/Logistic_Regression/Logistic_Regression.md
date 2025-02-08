# Logistic Regression: Classification Algorithm

## Introduction
Logistic regression is one of the most widely used **classification algorithms**. Unlike linear regression, which predicts **continuous values**, logistic regression is used for **binary classification problems**, where the outcome is either **0 or 1**.

---

## Why Use Logistic Regression?
- **Predicting Categories**: It is ideal for problems like **spam detection (spam or not spam)** and **medical diagnosis (malignant or benign tumor)**.
- **Ensures Outputs are Between 0 and 1**: Unlike linear regression, which can predict values outside this range, logistic regression outputs **probabilities**, making it ideal for classification.

### Real-World Analogy
Think of **weather forecasting**. Instead of predicting the exact temperature, logistic regression predicts the **probability of rain** (e.g., 70% chance of rain). Similarly, logistic regression predicts the probability that an instance belongs to a certain class.

---

## Logistic Regression vs. Linear Regression
| **Feature**            | **Linear Regression**              | **Logistic Regression** |
|------------------------|----------------------------------|------------------------|
| **Type of Output**     | Continuous values (e.g., price) | Binary values (e.g., spam or not spam) |
| **Prediction Range**   | Can be any real number          | Between 0 and 1 (probability) |
| **Use Case**          | Forecasting, stock prices       | Spam detection, fraud detection |
| **Function Used**      | Straight-line equation         | Sigmoid function |

---

## The Sigmoid Function: Key to Logistic Regression
- Logistic regression applies the **Sigmoid function** (also called the **logistic function**) to map predictions to a probability range **between 0 and 1**.
- **Formula for Sigmoid Function:**
  ```
  g(z) = 1 / (1 + e^(-z))
  ```
  - **z** is the weighted sum of inputs: `z = wX + b`
  - **e** is the mathematical constant (~2.718)

### Behavior of the Sigmoid Function
- **When z is very large (e.g., 100):** The function approaches **1**.
- **When z is very small (e.g., -100):** The function approaches **0**.
- **When z = 0:** The function outputs **0.5** (uncertainty between two classes).

### Example: Predicting Tumor Malignancy
- Input: **Tumor size** (X)
- If the model outputs **0.7**, it means there is a **70% chance** the tumor is malignant.
- If the model outputs **0.3**, it means there is a **30% chance** the tumor is malignant.

---

## How Logistic Regression Makes Predictions
1. Compute **z = wX + b** (linear combination of weights and inputs).
2. Apply the **sigmoid function** to convert it into a probability.
3. **Classify based on a threshold** (e.g., if probability > 0.5, classify as 1; otherwise, classify as 0).

### Example: Fraud Detection in Online Payments
- **X (input features):** Transaction amount, location, time of day
- **Output (y):** Fraudulent (1) or not fraudulent (0)
- If the model predicts **0.85**, it suggests an **85% chance of fraud**, so the transaction is flagged.

---

## Why Logistic Regression is Used in Industry
- **Internet Advertising**: Determines whether a user will click an ad (1 = click, 0 = no click).
- **Finance**: Predicts whether a loan applicant will default (1 = default, 0 = no default).
- **Healthcare**: Diagnoses diseases based on symptoms.
- **Self-Driving Cars**: Determines whether an obstacle is a pedestrian or not.

---

## Next Steps
- Learn about **decision boundaries** in logistic regression.
- Understand **how to evaluate classification models** using precision, recall, and accuracy.

---

## Next Section
- ### [Decision Boundary](Decision_Boundary.md)
