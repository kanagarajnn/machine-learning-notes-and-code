# Linear Regression Model Part 1

## Introduction to Supervised Learning and Linear Regression
- Supervised learning involves training a model using input-output pairs.
- **Linear Regression** is one of the most widely used supervised learning algorithms for predicting continuous values.
- It fits a **straight line** to the data to make predictions based on past observations.
- **Real-World Example**: Financial analysts at **Goldman Sachs** use linear regression to predict stock market trends based on historical data.

## Supervised Learning Explained
Supervised learning is like having a **teacher guide you while you learn**. In this case, the "teacher" is a dataset that contains examples with **known answers**. For example, if you want to predict the price of a house based on its size, you would use a dataset that includes **various house sizes and their corresponding prices**. 

**Linear regression** is a method that helps draw a **straight line** through this data to make predictions. Imagine you have a scatter of dots on a graph representing houses; linear regression helps you find the **best straight line** that fits those dots, allowing you to estimate the price of a house just by knowing its size.

### **Visualizing Linear Regression**
Think of it like trying to **find the best path through a forest of trees (data points)**. The straight line you draw is like a trail that helps you navigate through the trees to reach your destination (the predicted price). 
- If you measure a **house that is 1,250 square feet**, you can follow your trail to see that it might sell for around **$220,000**.
- This is how **linear regression works in supervised learning!**

## House Price Prediction Example
- Predicting house prices based on square footage is a classic regression problem.
- **Dataset Used**: Houses in Portland, with size (in square feet) as input and price (in thousands of dollars) as output.
- Example:
  - A house of **1250 square feet** is predicted to be around **$220,000** using linear regression.
- **Real-World Application**: **Zillow's Zestimate** uses linear regression models to estimate home values across the U.S.

## Why It’s Called Supervised Learning
- The model learns from **labeled data**, meaning both **input (X)** and **correct output (Y)** are provided.
- The model uses examples of houses with known prices to make future predictions.
- **Industry Application**: Banks like **Wells Fargo** use supervised learning models to predict mortgage approval odds.

## Regression vs. Classification
- **Regression**: Predicts continuous numerical values (e.g., house prices, temperatures, stock prices).
- **Classification**: Predicts discrete categories (e.g., spam detection, medical diagnosis, loan approval).
- **Example**:
  - **Regression**: Predicting **salary based on years of experience** in HR analytics.
  - **Classification**: Classifying a **loan applicant as low-risk or high-risk** for approval.

## Visual Representation of Data
- Data is plotted as points on a graph:
  - **X-axis**: House size in square feet.
  - **Y-axis**: House price in thousands of dollars.
- The goal is to fit a line that best predicts prices based on the house size.
- **Real-World Example**: **Tesla's Autopilot** uses regression to estimate the **distance of objects** from the vehicle based on camera and radar data.

## Representing Data in Tables
- Data can also be represented in a structured format:
  - **First column (X)**: House size.
  - **Second column (Y)**: Price of the house.
  - Each row represents a **training example**.
- **Example**: E-commerce platforms like **Amazon** use structured data tables to predict customer purchase trends based on past purchases.

## Standard Notation in Machine Learning
- **Input variable (X)**: Also called a **feature** (e.g., house size).
- **Output variable (Y)**: Also called the **target** (e.g., house price).
- **Number of training examples (m)**: Denoted as **m**, referring to the total number of data points.
- **(X, Y) Pair Notation**:
  - Each training example is represented as **(x^(i), y^(i))**.
  - Example: **(2104, 400)** represents a **2104 sq. ft. house** priced at **$400,000**.
- **Industry Use**: **Uber and Lyft** use regression models trained on ride history to estimate trip fares in real time.

## Training Set and Model Learning
- The dataset used to train the model is called the **training set**.
- The model learns from this data to make price predictions for unseen houses.
- The client’s house is **not part of the training set**, and its price is unknown.
- **Business Case**: **Airbnb** trains ML models on past rental data to predict pricing for new property listings.

## Notation for Training Examples
- **x^(i)**: The **i-th** input example (house size).
- **y^(i)**: The **i-th** output example (house price).
- **m**: The total number of training examples.
- **Example**: In healthcare, researchers train regression models using patient data (X) and disease progression (Y) to predict patient outcomes.

---
## Next Section
- ### [Linear Regression Model Part 2](Linear_Regression_Model_Part_2.md)
