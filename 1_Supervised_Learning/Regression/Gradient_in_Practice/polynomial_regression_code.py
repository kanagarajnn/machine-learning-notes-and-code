"""
polynomial_regression_code.py - Implementation of Polynomial Regression with Feature Engineering

Description:
- This module demonstrates the use of **Polynomial Regression** for modeling non-linear relationships.
- Polynomial regression is useful when data shows a **curved** trend that cannot be captured by simple linear regression.
- Feature Engineering is applied by transforming input features into polynomial features to improve model performance.

Why Polynomial Regression?
- Many real-world datasets exhibit **non-linear relationships** that cannot be well-represented by a straight line.
- By adding polynomial terms (x^2, x^3, etc.), we allow the model to capture **curved trends**.
- This improves the predictive accuracy of regression models in domains such as **finance, healthcare, and real estate**.

Features:
- **Generates synthetic data** with a quadratic relationship.
- **Transforms input features** into polynomial representations of any specified degree.
- **Trains a Polynomial Regression model** using Scikit-Learn’s Linear Regression.
- **Visualizes the original dataset** and the fitted polynomial curve.
- **Prints model coefficients** to interpret the effect of polynomial terms.

Real-World Applications:
- **Finance**: Predicting stock price trends.
- **Healthcare**: Modeling patient recovery rates over time.
- **E-commerce**: Optimizing pricing based on historical sales data.
- **Self-Driving Cars**: Predicting road curvature from sensor data.

Usage:
- Run as a standalone script to see Polynomial Regression in action:
    ```bash
    python feature_engineering_poly.py
    ```
- Import as a module to integrate with other machine learning pipelines:
    ```python
    from feature_engineering_poly import train_polynomial_regression
    model, poly = train_polynomial_regression(X, y, degree=3)
    ```

Metadata:
- Author: Kanagaraj N N
- Date: February 8, 2025
- Version: 1.0
- License: MIT (see LICENSE file for details)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Generate synthetic data for demonstration
def generate_data():
    """
    Generates synthetic dataset with a quadratic relationship and some noise.
    Returns:
        X (ndarray): Random feature values.
        y (ndarray): Quadratic function of X with noise.
    """
    np.random.seed(42)  # Ensuring reproducibility
    X = np.random.rand(100, 1) * 10  # Feature: Random values between 0 and 10
    # Quadratic relationship with noise
    y = 3 * X**2 + 2 * X + 5 + np.random.randn(100, 1) * 5
    return X, y


# Visualize data
def plot_data(X, y):
    """
    Plots the original dataset.
    Args:
        X (ndarray): Feature values.
        y (ndarray): Target values.
    """
    plt.scatter(X, y, color='blue', label='Data')
    plt.xlabel("Feature X")
    plt.ylabel("Target y")
    plt.title("Original Data Distribution")
    plt.legend()
    plt.show()


# Perform polynomial feature transformation
def transform_features(X, degree=2):
    """
    Transforms input features into polynomial features of the given degree.
    Args:
        X (ndarray): Feature values.
        degree (int): Polynomial degree.
    Returns:
        X_poly (ndarray): Transformed polynomial features.
        poly (PolynomialFeatures): Fitted polynomial transformer.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


# Train polynomial regression model
def train_polynomial_regression(X, y, degree=2):
    """
    Trains a polynomial regression model with the given polynomial degree.
    Args:
        X (ndarray): Feature values.
        y (ndarray): Target values.
        degree (int): Polynomial degree.
    Returns:
        model (LinearRegression): Trained polynomial regression model.
        poly (PolynomialFeatures): Fitted polynomial transformer.
    """
    X_poly, poly = transform_features(X, degree)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly


# Visualize polynomial regression results
def plot_polynomial_regression(X, y, model, poly):
    """
    Plots the polynomial regression model's predictions against the original data.
    Args:
        X (ndarray): Feature values.
        y (ndarray): Target values.
        model (LinearRegression): Trained model.
        poly (PolynomialFeatures): Polynomial feature transformer.
    """
    X_sorted = np.sort(X, axis=0)  # Sort X for smooth plotting
    # Transform sorted X using polynomial transformer
    X_poly_sorted = poly.transform(X_sorted)
    y_pred = model.predict(X_poly_sorted)  # Predict using trained model

    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X_sorted, y_pred, color='red', linewidth=2,
             label=f'Polynomial Regression')
    plt.xlabel("Feature X")
    plt.ylabel("Target y")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y = generate_data()  # Generate synthetic dataset
    plot_data(X, y)  # Visualize dataset

    degree = 2  # Adjust polynomial degree as needed
    model, poly = train_polynomial_regression(X, y, degree)  # Train model
    plot_polynomial_regression(X, y, model, poly)  # Plot results

    # Print model coefficients
    print(f"Polynomial Regression Model Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")  # Print intercept
