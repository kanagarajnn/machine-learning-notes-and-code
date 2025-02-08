# Multiple Linear Regression: Introduction to Multiple Features

## Moving Beyond a Single Feature
- In the initial linear regression model:
  - **Single Feature**: The input feature was the size of the house (\(X_1\)).
  - **Output**: The price of the house (\(Y\)).
- Real-World Analogy: Think of predicting a car's resale value based solely on its mileage.
- **Enhancement**: By including additional features like the car’s brand, condition, and model year, predictions become more accurate.

## Introduction to Multiple Features
- **Features**: Additional inputs such as:
  - Number of bedrooms (\(X_2\))
  - Number of floors (\(X_3\))
  - Age of the house (\(X_4\))
- **Model Update**:
  \[
  f(w, b, X) = w_1X_1 + w_2X_2 + w_3X_3 + w_4X_4 + b
  \]
- Example: A **real estate platform** like Zillow could use these features to estimate house prices more precisely.
- To visualize this, think of a **recipe for a cake**. Each ingredient (like flour, sugar, and eggs) contributes to the final taste. In multiple linear regression, each feature (like size, bedrooms, etc.) is like an ingredient that helps you calculate the final price of the house. The model uses a formula that combines these features with specific weights (like how much of each ingredient to use) to give you the best prediction.

## Understanding the Parameters
- **Base Price** (\(b\)): Reflects the price with no features (e.g., $80,000 starting price for a property).
- **Weights** (\(w_1, w_2, \dots\)):
  - \(w_1 = 0.1\): Price increases by $100 per square foot.
  - \(w_2 = 4\): Price increases by $4,000 for each additional bedroom.
  - \(w_3 = 10\): Price increases by $10,000 for each additional floor.
  - \(w_4 = -2\): Price decreases by $2,000 for every additional year of the house’s age.

## Notation Simplification
- Features:
  \[
  X = [X_1, X_2, X_3, \dots, X_n]
  \]
  - Example: \(X = [1416, 3, 2, 40]\) for a house with:
    - 1,416 square feet
    - 3 bedrooms
    - 2 floors
    - 40 years old.
- Parameters:
  \[
  W = [w_1, w_2, w_3, \dots, w_n]
  \]
- Model in Compact Form:
  \[
  f(w, b, X) = W \cdot X + b
  \]
  - **Dot Product**:
    \[
    W \cdot X = w_1X_1 + w_2X_2 + w_3X_3 + \dots + w_nX_n
    \]

## Real-World Applications of Multiple Linear Regression
- **E-Commerce**: Platforms like **Amazon** use multiple features such as price, reviews, and shipping time to rank products.
- **Healthcare**: Predicting disease risk using features like age, blood pressure, and cholesterol levels.
- **Finance**: Predicting loan approval likelihood based on income, credit score, and repayment history.

## Naming the Model
- **Multiple Linear Regression**:
  - Refers to models with multiple input features.
  - **Not** to be confused with "multivariate regression," which addresses predicting multiple outputs.

## Benefits of Using Multiple Features
- **Enhanced Predictions**: Adding more features increases the accuracy of predictions.
- **Real-World Analogy**: Predicting a student’s performance based on:
  - Study hours
  - Class attendance
  - Quality of study material.

---

## Next Section
- ### [Vectorization Part 1](Vectorization_Part_1.md)
