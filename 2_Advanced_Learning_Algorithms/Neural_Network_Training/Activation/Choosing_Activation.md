# Choosing the Right Activation Function in Neural Networks

## Overview
- Activation functions determine how signals propagate through a neural network.
- The choice of activation function depends on the **type of problem** being solved.
- **Why is this important?**
  - A poor choice of activation function can slow down training.
  - Some activation functions help neural networks learn more efficiently.

---

## Choosing an Activation Function for the **Output Layer**
- The **output layer’s activation function** depends on the type of prediction task.

### **1. Binary Classification (Yes/No, Spam/Not Spam, etc.)**
- Use **Sigmoid** activation.
- Formula:
  ```
  g(z) = 1 / (1 + e^(-z))
  ```
- **Why?** It outputs a probability between `0` and `1`, making it ideal for binary problems.
- **Example**: Email spam classification (probability that an email is spam).

### **2. Regression (Predicting Continuous Values, e.g., Stock Price)**
- Use **Linear** activation.
- Formula:
  ```
  g(z) = z
  ```
- **Why?** Allows predictions to take on any value, positive or negative.
- **Example**: Predicting tomorrow’s stock price change.

### **3. Non-Negative Regression (Predicting Prices, Counts, etc.)**
- Use **ReLU (Rectified Linear Unit)** activation.
- Formula:
  ```
  g(z) = max(0, z)
  ```
- **Why?** Ensures outputs are non-negative (since negative prices don’t make sense).
- **Example**: Predicting house prices.

---

## Choosing an Activation Function for **Hidden Layers**
- **Default choice: ReLU**
- **Why?**
  - Faster to compute (`max(0, z)` is simpler than exponentials in sigmoid).
  - Avoids **vanishing gradient** issues found in sigmoid/tanh functions.
- **Example**: Used in **image recognition models (Google Photos, Facebook Face Recognition)**.

---

## Why Not Always Use Sigmoid?
- Sigmoid slows down learning in deep networks due to **vanishing gradients**.
- **ReLU** is preferred because it doesn’t saturate like sigmoid (except for negative values, which get zeroed out).

---

## Other Activation Functions You Might Encounter
- **LeakyReLU**: A small slope for negative values instead of zero.
- **Swish**: A more recent function, sometimes better than ReLU.
- **Softmax**: Used in multi-class classification.

---

## Summary: Choosing the Right Activation Function
| Problem Type | Recommended Activation Function |
|-------------|--------------------------------|
| Binary Classification | Sigmoid |
| Regression (Positive/Negative) | Linear |
| Regression (Only Positive) | ReLU |
| Hidden Layers (Default) | ReLU |

---

By choosing the right activation function, you ensure your neural network learns efficiently and provides accurate predictions—just like choosing the right **engine for a car** ensures optimal performance!



---
## Next Section
- ### [Why Activation](Why_Activation.md)
