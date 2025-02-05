# Cost Function Formula in Linear Regression

## Introduction to Cost Function
- The **cost function** measures how well a linear regression model fits the training data.
- It helps adjust model parameters (**w and b**) to minimize errors.
- **Real-World Example**: Cost functions are used in AI-driven **financial forecasting models** to minimize prediction errors in stock price predictions.

## Understanding the Linear Regression Model
- The model is represented as:
  
  **f(x) = wx + b**
  
  where:
  - **w (weight/parameter)**: Determines the slope of the line.
  - **b (bias/intercept)**: Determines where the line crosses the y-axis.
- **Different values of w and b result in different prediction lines**.
- **Industry Example**: **Zillow's home price estimation tool** uses a linear regression model to predict real estate prices based on house size, location, and historical prices.

## How w and b Affect the Model
- **When w = 0 and b = 1.5**, the function predicts a constant value of 1.5 (a horizontal line).
- **When w = 0.5 and b = 0**, the line has a slope of 0.5, increasing steadily.
- **When w = 0.5 and b = 1**, the line shifts upwards but maintains the same slope.
- **Real-World Application**: **Google Maps' ETA estimation** uses linear regression models with different parameter values to adjust estimated travel times based on traffic.

## What Makes a Good Fit?
- The goal of linear regression is to **find values of w and b that best fit the training data**.
- A good fit means the line **closely follows the trend of data points**, minimizing error.
- **Example**: **Netflix’s recommendation system** fits a regression model to user watch history, making more accurate predictions for movie recommendations.

## Measuring Prediction Error
- The difference between predicted and actual values is called the **error**:
  
  **Error = ŷ - y**  (where ŷ is predicted, y is actual)
  
- **Squared Error**: To measure how far off predictions are from actual values, we compute:
  
  **(ŷ - y)²**
  
- **Real-World Use Case**: **Credit card fraud detection models** minimize prediction error by reducing false positives in fraud alerts.

## Summing Up Errors Over the Dataset
- To measure overall model performance:
  - Sum all squared errors:  
    
    **∑ (ŷ - y)²** (Summation across all training examples)
  
  - Average the errors by dividing by **m (number of examples)**:
    
    **J(w, b) = (1/2m) * ∑ (ŷ - y)²**
  
  - The **(1/2)** factor is included to make future calculations easier.
- **Industry Example**: **Amazon’s demand forecasting model** minimizes cost function errors to optimize inventory and reduce overstock.

## Purpose of the Cost Function
- The cost function **quantifies how well a model fits**.
- A **small J(w, b)** means a better model; a **large J(w, b)** means poor predictions.
- **Example**: **Autonomous driving models** use cost functions to minimize lane-detection errors.

## Next Steps
- The next topic will focus on **minimizing J(w, b) using gradient descent**.
- This helps optimize the model to achieve better predictions.
- **Upcoming Industry Case Study**: How **Tesla optimizes self-driving models** by minimizing the cost function.

---
## Next Section
- ### [Cost Function Intuition](Cost_Function_Intuition.md)
