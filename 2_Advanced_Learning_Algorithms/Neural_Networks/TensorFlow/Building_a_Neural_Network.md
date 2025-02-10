# **Building Neural Networks in TensorFlow**

## **Introduction**
Neural networks are the foundation of modern **AI applications**, powering technologies like **speech recognition** (Siri, Google Assistant), **image classification** (Google Photos, Facebook AI), and **recommendation systems** (Netflix, Amazon).

This guide explains how to **build and train** a neural network using **TensorFlow**, covering key concepts like **forward propagation, training, inference,** and the **Sequential API**.

---

## **1. Forward Propagation in TensorFlow**
### **What is Forward Propagation?**
**Forward propagation** is the process where **input data** passes through the network **layer by layer**, applying **weights** and **activations** to compute the final **output**.

### **Explicit Approach to Forward Propagation**
Initially, we manually perform computations **layer by layer**:
1. **Initialize** the input `X`
2. **Create Layer 1**, compute activations `a1`
3. **Create Layer 2**, compute activations `a2`

This method requires explicitly passing outputs from one layer to the next.

### **Simplified Approach with TensorFlow**
TensorFlow provides an easier way using the **Sequential API**, which automatically **links layers together**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple neural network with 2 layers
model = Sequential([
    Dense(3, activation='sigmoid', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
```
This approach **automates layer connectivity**, making model creation **faster** and **more readable**.

---

## **2. Training a Neural Network in TensorFlow**
### **Dataset Representation**
A dataset consists of **features (`X`)** and **labels (`Y`)**. Example:
```python
import numpy as np

X_train = np.array([[0.5, 1.2], [1.3, 3.1], [0.9, 2.3], [2.0, 4.5]])
Y_train = np.array([0, 0, 1, 1])
```
Here, `X_train` contains **2 features per example**, and `Y_train` holds **binary labels (0 or 1)**.

### **Compiling and Training the Model**
To train a **neural network**, we:
1. **Compile the model** (define **loss function, optimizer, metrics**)
2. **Fit the model** (train it on data)

```python
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=2)
```
- `binary_crossentropy` is used for **classification tasks**.
- `adam` **optimizer** adjusts weights **efficiently**.

---

## **3. Making Predictions (Inference)**
Once trained, the model can **predict new data**:
```python
X_new = np.array([[1.1, 2.4]])
prediction = model.predict(X_new)
print(prediction)
```
This runs **forward propagation** on `X_new` and outputs a **predicted probability**.

### **Real-World Example:**
- **Spam detection**: Input **email text features**, output **probability of spam**.
- **Medical AI**: Input **patient symptoms**, output **risk of disease**.

---

## **4. Digit Classification Example**
Applying TensorFlow's **Sequential API** to classify **handwritten digits (MNIST dataset)**:
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
Y_train = to_categorical(Y_train)

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=32)
```
### **Key Points**
- **Flatten images** from **28×28** to a **1D vector (784 features)**.
- **Softmax activation** ensures outputs represent **probability distribution** over **10 digits (0-9)**.
- Used in **AI-powered OCR systems** like **Google Lens**.

---

## **5. Understanding the Code Beneath the Libraries**
TensorFlow simplifies **neural networks** into **5-10 lines of code**, but understanding how **forward propagation** and **training** work **internally** is crucial.

### **Why Should You Know the Math Behind AI?**
- **Debugging models**: Understanding **internals** helps fix **errors** in complex networks.
- **Optimizing performance**: **Fine-tuning hyperparameters** requires deep knowledge.
- **Building custom models**: **AI researchers and ML engineers** often develop new architectures.

---

## **6. Key Takeaways**
- **Forward propagation** can be done **manually** or via **TensorFlow’s Sequential API**.
- **Model training** requires **compilation** (**loss function, optimizer**) and **fitting on data**.
- **Predictions** are performed using `model.predict()`, enabling **AI-powered applications**.
- Understanding **internal workings** of **TensorFlow** allows for **better debugging and customization**.


---
## Next Section
- ### [Lab: Coffee Roasting in Tensorflow](Lab_Coffee_Roasting_in_Tensorflow.md)
