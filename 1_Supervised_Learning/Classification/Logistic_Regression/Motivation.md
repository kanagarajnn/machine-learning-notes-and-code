# Motivation (Intro to Classification and Logistic Regression)

## Understanding Classification
- **Classification** is a type of supervised learning where the goal is to predict discrete categories rather than continuous values.
- Unlike linear regression, which predicts a **numeric value**, classification models predict **categories** (e.g., spam vs. not spam).

### Real-World Analogy
Think of sorting emails into "Inbox" and "Spam." Each email is either **spam (1)** or **not spam (0)**—this is a **binary classification** problem.

## Why Linear Regression is Not Ideal for Classification
- If we apply **linear regression** to a classification problem, it can predict values **outside the range of 0 and 1**, making it unsuitable for classifying discrete categories.
- Example:
  - Predicting whether a **tumor is malignant or benign** based on its size.
  - A linear regression model might output **values like -0.3 or 1.5**, which don’t make sense for classification.

## Binary Classification
- In binary classification, the model predicts one of **two possible categories**.
- Examples:
  - **Email Spam Detection**: Spam (1) or Not Spam (0).
  - **Fraud Detection**: Fraudulent (1) or Not Fraudulent (0).
  - **Medical Diagnosis**: Malignant (1) or Benign (0).
- By convention, we refer to these classes as **positive (1)** and **negative (0)**.

## The Issue with Linear Regression in Classification
- Linear regression **fits a straight line** to the data, which is problematic when classifying categories.
- If we use a **fixed threshold** (e.g., if prediction > 0.5, classify as 1), adding an outlier can shift the entire decision boundary.
- This can cause misclassification, making **linear regression unreliable for classification problems**.

### Example: Tumor Classification
- Suppose we classify tumors as **malignant (1) or benign (0)** based on their size.
- Using **linear regression**, a tumor with a very large size can shift the decision boundary, leading to incorrect classifications.
- **Solution**: We need a model that ensures **predictions are always between 0 and 1** and that outputs probabilities instead of raw numbers.

## Introduction to Logistic Regression
- **Logistic Regression** is specifically designed for **binary classification**.
- It applies a function that **keeps predictions between 0 and 1**, making it well-suited for classification problems.
- Even though it has "regression" in its name, **logistic regression is used for classification**.

## What’s Next?
- In the next section, we will explore **logistic regression in detail** and see how it overcomes the issues of using linear regression for classification.
- We will also introduce **decision boundaries** and how logistic regression **models probabilities** instead of raw numerical outputs.

---

## Next Section
- ### [Logistic Regression](Logistic_Regression.md)
