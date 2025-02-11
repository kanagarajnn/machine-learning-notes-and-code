# Multi-Class Classification with Neural Networks

## Overview
- **Multi-class classification** is used when an input belongs to one of several categories.
- Unlike **binary classification**, where only two outcomes exist, multi-class classification has **more than two possible outputs**.
- **Real-world applications**:
  - **Object recognition** (identifying cats, dogs, cars in images)
  - **Language processing** (classifying words into nouns, verbs, adjectives)
  - **Medical diagnosis** (predicting one of multiple diseases)

---

## Understanding Multi-Class Classification
- Neural networks classify data by **mapping input features to output categories**.
- **Example:**
  - Input: Image of an animal.
  - Output: **One** category among {Dog, Cat, Horse, Other}.
- **Key difference from multi-label classification**:
  - Multi-class → **Only one category per input**.
  - Multi-label → **Multiple categories possible per input**.

---

## Data Preparation and Visualization
- **Dataset:** Using `make_blobs` to create a **4-class dataset**.
  ```python
  from sklearn.datasets import make_blobs
  X_train, y_train = make_blobs(n_samples=100, centers=4, cluster_std=1.0, random_state=30)
  ```
- **Visualizing the dataset:**
  ```python
  plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.title("Training Data Distribution")
  plt.show()
  ```
- Each **dot** represents a training example, and the **color** represents its class.

---

## Neural Network Model for Multi-Class Classification
### **1. Network Architecture**
- A **2-layer network** with:
  - **Input layer:** Takes `x0` and `x1` features.
  - **Hidden layer:** 2 neurons with **ReLU activation**.
  - **Output layer:** 4 neurons (one per category) with **linear activation**.
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(2, activation='relu', name="L1"),
      Dense(4, activation='linear', name="L2")
  ])
  ```
- **Why Linear Activation for the Output Layer?**
  - Instead of applying **Softmax** in the final layer, we use **linear activation**.
  - Softmax is applied inside the **loss function** for better **numerical stability**.

### **2. Compiling and Training the Model**
```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)
model.fit(X_train, y_train, epochs=200)
```
- **SparseCategoricalCrossentropy (`from_logits=True`)**:
  - Indicates the outputs are raw logits, and Softmax is applied internally.
  - Improves numerical stability and prevents computation issues.
- The model learns how to **separate the classes** by minimizing the loss over 200 epochs.

---

## How Neural Networks Make Predictions
### **1. Decision Boundaries**
- A trained model divides the **input space** into regions.
- When a new data point is given, the network **assigns it to the most probable class**.
```python
plt_cat_mc(X_train, y_train, model, classes)
```
- The **decision boundary** visualization shows how well the model classifies different regions.

### **2. How Each Layer Works**
- **First layer (`L1`) transforms the data**:
  - Uses **ReLU activation**.
  - Creates intermediate features for classification.
- **Second layer (`L2`) classifies the transformed data**:
  - Outputs 4 logits (one for each class).
  - The **Softmax function** assigns probabilities to each class.

### **3. Selecting the Most Likely Category**
- The **highest probability** category is chosen using:
```python
predicted_class = np.argmax(model.predict(X_test), axis=1)
```
- Ensures the model always returns a **single predicted category**.

---

## Summary
| Feature | Multi-Class Classification |
|---------|----------------------------|
| Output | One category per input |
| Example | Dog, Cat, Horse, Other |
| Activation Function | ReLU (Hidden) + Linear (Output) |
| Loss Function | Sparse Categorical Crossentropy |
| Softmax Placement | Applied inside the loss function |

---

## Conclusion
- Multi-class classification enables neural networks to **assign one label per input**.
- Using **linear activation in the final layer** and applying **Softmax inside the loss function** improves numerical stability.
- **Real-World Applications:**
  - **Google Photos** (object classification)
  - **Amazon Alexa** (intent classification)
  - **Medical AI** (disease detection)

By mastering multi-class classification, you can build **robust AI models** for various real-world challenges!


---
## Next Section
- ### [Advanced Optimization](../../Additional_Concepts/Advanced_Optimization.md)
