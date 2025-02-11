# Introduction to Backpropagation in Neural Networks

## Overview
- **Backpropagation** is the core algorithm that enables neural networks to learn by computing derivatives of the **cost function**.
- It allows **gradient descent** or **Adam optimizer** to adjust parameters efficiently.
- Backpropagation relies on **calculus** to determine how much each parameter influences the final output.
- **Real-world applications**:
  - Image recognition (Google Photos, Facebook Face ID)
  - Speech processing (Siri, Alexa, Google Assistant)
  - Autonomous systems (Self-driving cars, medical AI)

---

## How Backpropagation Works
### **1. Computing the Cost Function’s Derivative**
- Given a cost function `J(w)`, we want to find how `J` changes when `w` changes.
- Example: If `J(w) = w^2` and `w = 3`, then:
  ```
  J(w) = 3^2 = 9
  ```
- If `w` increases slightly by `ε = 0.001`, then:
  ```
  J(3.001) = 3.001^2 = 9.006001
  ```
- The change in `J(w)`:
  ```
  ΔJ ≈ 6 * ε
  ```
- This means the **derivative of J with respect to w is 6**, written as:
  ```
  dJ/dw = 6
  ```
- This tells us **how much J changes when w changes**, a fundamental concept in **gradient descent**.

---

## Why Derivatives Matter in Neural Networks
- In a neural network, **each parameter (weight/bias) affects the final output**.
- **Gradient descent** updates weights based on their influence on the cost function.
- If the derivative is **large**, update weights significantly.
- If the derivative is **small**, make smaller updates.
- **Example analogy**:
  - Think of training as **hiking down a mountain**.
  - If the slope (gradient) is steep, take big steps (large weight updates).
  - If the slope is flat, take small steps (smaller updates to avoid overshooting).

---

## Using Python to Compute Derivatives
### **1. Using SymPy for Automatic Differentiation**
- Instead of manually computing derivatives, Python’s **SymPy** can do it for us:
```python
import sympy as sp
w = sp.Symbol('w')
J = w**2
J_derivative = sp.diff(J, w)
print(J_derivative)  # Output: 2w
```
- Plug in `w = 3` to compute:
```python
J_derivative.subs(w, 3)  # Output: 6
```

### **2. Common Derivatives in Machine Learning**
| Function | Derivative |
|----------|------------|
| `J(w) = w^2` | `2w` |
| `J(w) = w^3` | `3w^2` |
| `J(w) = 1/w` | `-1/w^2` |

- These derivatives help adjust weights efficiently in **neural networks**.

---

## Conclusion
- Backpropagation **computes derivatives automatically** to update weights.
- Gradient descent uses these derivatives to minimize **cost functions**.
- Python libraries like **SymPy** simplify derivative calculations.
- **Next Step:** Understanding **computation graphs** to visualize backpropagation.

By mastering backpropagation, you’ll be able to train **more effective deep learning models** for tasks like **autonomous driving, medical diagnosis, and AI-powered assistants**!

---
## Next Section
- ### [Computation Graph](Computation_Graph.md)
