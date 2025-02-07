# Supervised Machine Learning Part 2: Classification Introduction

## Definition of Classification
- Classification is a type of supervised learning that predicts **categories** rather than **continuous values**.
- Unlike regression, classification predicts from a **finite set of possible outputs** instead of numerical values.
- **Real-World Example**: Email spam detection classifies emails as **spam or not spam** using past email patterns.

## Example: Breast Cancer Detection
- A classification model can determine whether a tumor is **benign (0)** or **malignant (1)** based on historical medical data.
- The model learns from patient data, such as tumor size and shape, to predict the correct category.
- **Real-World Application**: IBM Watson Health and Google's DeepMind use ML-based classification for cancer detection.

## Difference Between Classification and Regression
- **Regression**: Predicts numerical values (e.g., house prices, sales forecasts).
- **Classification**: Predicts categorical values (e.g., fraudulent vs. non-fraudulent transactions, spam vs. non-spam emails).
- **Example**:
  - **Regression**: Predicting **stock prices** based on market data.
  - **Classification**: Determining whether a **credit card transaction** is fraudulent.

## Multi-Class Classification
- Classification is not limited to two categories.
- A model can classify data into multiple groups (e.g., different types of cancer, various customer segments in marketing).
- **Example**: A cancer detection model may classify tumors as **benign, type 1 cancer, or type 2 cancer**.

## Non-Numeric and Numeric Categories
- Categories can be **non-numeric** (e.g., "cat" vs. "dog") or **numeric labels** (e.g., 0, 1, 2), where the values represent distinct categories but have no numerical significance.
- **Example**: Classifying news articles into categories like **sports, politics, and entertainment** in Google News.

## Using Multiple Input Features
- Models can take **multiple input variables** to make better predictions.
- **Example**: Predicting tumor malignancy using **tumor size, patient age, and genetic factors**.
- **Real-World Application**: Tesla’s Autopilot system classifies road conditions based on sensor data from multiple cameras and radar.

## Decision Boundaries in Classification
- A learning algorithm determines a **decision boundary** that separates different classes in the dataset.
- This boundary helps classify new inputs based on prior learned patterns.
- **Example**: In facial recognition, decision boundaries help distinguish between different faces.

## Complexity of Classification Models
- Advanced classification models use multiple features beyond just one variable.
- **Example**: A breast cancer detection model may incorporate **tumor thickness, cell size uniformity, cell shape, and genetic markers**.
- **Real-World Application**: Fraud detection in banking systems combines **transaction amount, location, user behavior, and spending patterns**.

## Supervised Learning Recap
- **Supervised learning maps input (X) to output (Y) using labeled data**.
- The two major types are:
  - **Regression**: Predicts continuous values (e.g., house prices, sales revenue).
  - **Classification**: Predicts discrete categories (e.g., spam vs. not spam, disease presence vs. absence).

## Next Steps
- The next topic explores **unsupervised learning**, where models learn from **unlabeled data**.
- Learners will understand how clustering and pattern detection work in machine learning.
- **Real-World Case Study**: Customer segmentation in e-commerce platforms like **Amazon and Shopify** using unsupervised learning.

---
## Next Section
- ### [Unsupervised Machine Learning Part 1](Unsupervised_Machine_Learning_Part_1.md)
