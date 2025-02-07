# Gradient Descent Convergence: How to Know If It's Working

## Introduction
One of the most important questions when running **gradient descent** is: **How do we know if it is converging?** In other words, how can we tell if it's helping us find values for **w** and **b** that minimize the cost function **J(w, b)**?

## Understanding Gradient Descent Behavior
- The goal of **gradient descent** is to find the optimal values for **w** and **b** that minimize the cost function **J**.
- A good way to monitor its performance is by **plotting the cost function** at each iteration.
- This plot is called a **learning curve**, which helps us determine if gradient descent is working properly.

## The Learning Curve: A Key Indicator
- The **x-axis** represents the **number of iterations** (how many times we've updated w and b).
- The **y-axis** represents the **cost function J(w, b)** (how well our model is performing).
- A properly running gradient descent algorithm should show a **decreasing J value** at every iteration.
- If **J increases at any iteration**, this may indicate:
  - **A bug in the code**.
  - **A poorly chosen learning rate (α)**—usually too large.

### Real-World Analogy
Imagine you're **hiking down a mountain** at night. Each step you take is an update to **w** and **b**, and the goal is to reach the lowest point (global minimum). If your steps are **too large**, you might overshoot and climb back up (J increases). If your steps are **too small**, it will take forever to reach the bottom.

## Detecting Convergence
- As **iterations increase**, J should **decrease steadily**.
- At some point, **J levels off and no longer decreases significantly**.
- This means **gradient descent has converged** and further updates won’t improve the model much.
- Example:
  - **At 300 iterations**, J is still decreasing.
  - **At 400 iterations**, J has nearly flattened, indicating convergence.

## How Many Iterations Are Needed?
- The number of iterations required for convergence **varies widely**:
  - Some models may converge in **30 iterations**.
  - Others may take **1,000 or even 100,000 iterations**.
- **There is no universal rule**—this depends on the dataset and problem.
- To determine convergence, we **plot a learning curve** and visually inspect it.

## Using Automatic Convergence Tests
- One way to automate convergence detection is by defining a **threshold (ε)**.
- If **J decreases by less than ε (e.g., 0.001) in an iteration**, we assume convergence.
- However, **choosing the right ε is difficult**, so manual inspection is often preferred.

## Summary
- **Gradient descent should always reduce J over time.**
- **A well-chosen learning rate α prevents overshooting or slow convergence.**
- **Plotting J vs. iterations (learning curve) helps determine when training is done.**
- **Automatic convergence tests can help, but visual confirmation is often more reliable.**

---

## Next Section
- ### [Choosing Learning Rate](Choosing_Learning_Rate.md)
