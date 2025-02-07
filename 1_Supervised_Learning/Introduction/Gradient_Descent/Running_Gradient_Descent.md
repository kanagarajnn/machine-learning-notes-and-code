# Running Gradient Descent for Linear Regression

## Introduction
Gradient Descent is a key optimization algorithm that iteratively updates parameters **w** and **b** to minimize the cost function. In this guide, we'll explore how gradient descent works visually and apply it to a **real-world scenario of predicting house prices**.

## Step-by-Step Execution of Gradient Descent
1. **Initializing Parameters**
   - In this demonstration, we initialize parameters as:
     
     \( w = -0.1, \quad b = 900 \)
     
   - This gives the initial function:
     
     \( f(x) = -0.1x + 900 \)
     
   - At this point, the **line does not fit the data well**.

2. **First Step of Gradient Descent**
   - We compute the cost function’s gradient and take a small step **downhill** in the cost function.
   - The model parameters **w** and **b** adjust slightly, moving towards a better fit.
   - This is like **tuning a radio station**: **each adjustment gets closer to the best frequency**.

3. **Multiple Iterations & Convergence**
   - With each step, **w and b follow a trajectory towards the global minimum**.
   - The **straight-line fit continuously improves**, reducing prediction errors.
   - Imagine learning a new sport—**each practice session improves accuracy and efficiency**.

4. **Reaching the Global Minimum**
   - Once we reach the optimal values of **w and b**, gradient descent stops.
   - This corresponds to the best-fit line, which accurately models the data.
   - Example: In **real estate pricing**, this ensures we predict house values with high accuracy.

## What is Batch Gradient Descent?
Batch Gradient Descent computes updates by considering **all training examples at each step**.

### Weight Update Formula
\[
 w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} (f(x^i) - y^i) x^i
\]

### Bias Update Formula
\[
 b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f(x^i) - y^i)
\]

These formulas ensure a **smooth descent** toward the minimum.

**Example**: Netflix’s recommendation system processes **entire user histories** to update personalized content predictions.

## Alternative Approaches to Gradient Descent
- **Stochastic Gradient Descent (SGD)**: Updates parameters using **only one example per step**, leading to faster but noisier learning.
- **Mini-Batch Gradient Descent**: Uses **small subsets of training data**, balancing efficiency and stability.
- **Example**: E-commerce platforms like **Amazon** fine-tune pricing dynamically using real-time mini-batch updates.

## Key Takeaways
- **Gradient Descent iteratively adjusts w and b to minimize errors.**
- **Visualizing gradient descent helps understand parameter updates step by step.**
- **Batch Gradient Descent considers the entire dataset, ensuring stable learning.**
- **Used in real-world applications like house pricing models, recommendation systems, and AI-powered finance models.**

---
## Next Section
- ### [Mutliple Features](../../Regression/Multiple_Linear_Regression/Mutliple_Features.md)
