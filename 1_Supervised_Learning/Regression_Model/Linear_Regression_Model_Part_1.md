# Linear Regression Model Part 1

## Introduction to Supervised Learning and Linear Regression
- Supervised learning involves training a model using input-output pairs.
- **Linear Regression** is one of the most widely used supervised learning algorithms.
- It fits a straight line to the data to make predictions.

## House Price Prediction Example
- Predicting house prices based on square footage is a classic regression problem.
- **Dataset Used**: Houses in Portland, with size (in square feet) as input and price (in thousands of dollars) as output.
- Example:
  - A house of **1250 square feet** is predicted to be around **$220,000** using linear regression.

## Why It’s Called Supervised Learning
- The model learns from labeled data, meaning both **input (X)** and **correct output (Y)** are provided.
- The model uses examples of houses with known prices to make future predictions.

## Regression vs. Classification
- **Regression**: Predicts continuous numerical values (e.g., house prices, temperatures).
- **Classification**: Predicts discrete categories (e.g., spam detection, medical diagnosis).

## Visual Representation of Data
- Data is plotted as points on a graph:
  - **X-axis**: House size in square feet.
  - **Y-axis**: House price in thousands of dollars.
- The goal is to fit a line that best predicts prices based on the house size.

## Representing Data in Tables
- Data can also be represented in a table:
  - **First column (X)**: House size.
  - **Second column (Y)**: Price of the house.
  - Each row represents a **training example**.

## Standard Notation in Machine Learning
- **Input variable (X)**: Also called a **feature** (e.g., house size).
- **Output variable (Y)**: Also called the **target** (e.g., house price).
- **Number of training examples (m)**: Denoted as **m**, referring to the total number of data points.
- **(X, Y) Pair Notation**:
  - Each training example is represented as **(x^(i), y^(i))**.
  - Example: **(2104, 400)** represents a 2104 sq. ft. house priced at $400,000.

## Training Set and Model Learning
- The dataset used to train the model is called the **training set**.
- The model learns from this data to predict house prices.
- The client’s house is **not part of the training set** since its price is unknown.

## Notation for Training Examples
- **x^(i)**: The **i-th** input example (house size).
- **y^(i)**: The **i-th** output example (house price).
- **m**: The total number of training examples.

## Next Steps
- The next topic will explain how to feed this training data into a learning algorithm.
- The algorithm will learn from the dataset to make accurate predictions.

---

## Next Section
- ### [Linear Regression Model Part 2](Linear_Regression_Model_Part_2.md)
