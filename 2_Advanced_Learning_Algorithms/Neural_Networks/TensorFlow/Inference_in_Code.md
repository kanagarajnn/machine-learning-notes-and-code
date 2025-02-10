# **Neural Network Inference Using TensorFlow**

## **Introduction**
Neural networks have become an integral part of **image recognition, natural language processing, self-driving cars, and more**. The beauty of neural networks lies in their versatility—the **same algorithm** can be applied to a wide range of applications.

In this document, we explore how to implement **forward propagation (inference)** in a neural network using **TensorFlow**. We will illustrate this with real-world examples, such as optimizing **coffee roasting** and **handwritten digit classification**.

---

## **Why TensorFlow?**
**TensorFlow** is a deep learning framework developed by Google. It is widely used for training and deploying **AI models** across various industries. Some real-world applications include:
- **Google Photos**: Automatically categorizes images using deep learning.
- **Tesla Autopilot**: Uses neural networks for real-time object detection.
- **Netflix Recommendations**: Predicts what to watch next based on user history.

In this lab, we focus on **using TensorFlow for neural network inference**.

---

## **Example 1: Coffee Roasting Optimization**
### **Can AI Help in Making the Perfect Coffee?**
Imagine you are roasting coffee beans. Two parameters affect the quality of your coffee:
1. **Temperature (°C)**: Too low, and the beans are undercooked; too high, and they burn.
2. **Duration (minutes)**: Roasting for too little or too long also affects taste.

By using machine learning, we can **train a neural network** to predict whether a coffee batch will be **good or bad** based on these two parameters.

### **Setting Up the Input Data**
- Define **feature vector `x`** containing **temperature and duration**.
- Example: `x = [200, 17]` (200°C for 17 minutes).

### **Building the Neural Network**
1. **First Hidden Layer**
   ```python
   Layer_1 = tf.keras.layers.Dense(units=3, activation='sigmoid')
   a1 = Layer_1(x)  # Produces a list of 3 activation values
   ```
   This layer processes the input features and extracts relevant patterns.

2. **Second Hidden Layer**
   ```python
   Layer_2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
   a2 = Layer_2(a1)  # Produces a single value
   ```
   The final output `a2` represents the **probability** of the coffee being good.

3. **Final Prediction**
   ```python
   y_hat = 1 if a2 >= 0.5 else 0  # Thresholding at 0.5
   ```
   - If `y_hat = 1`, the coffee is **good**.
   - If `y_hat = 0`, the coffee is **bad**.

### **Real-World Application**
While this example is simplified, **real machine learning models are used in the coffee industry** to optimize roasting and ensure consistent quality. AI-driven coffee roasters, such as **Ikawa and Bellwether Coffee**, use **data-driven roasting profiles** to enhance taste.

---

## **Example 2: Handwritten Digit Classification**
### **Recognizing Digits with AI**
AI can recognize handwritten digits, such as those found in **postal mail sorting systems** and **bank check scanning**. The input to our model is a list of **pixel intensity values** from an image of a handwritten number.

### **Building the Neural Network**
1. **Input Layer (Pixel Values)**
   ```python
   x = np.array([pixel_intensities])  # Convert image to array
   ```
2. **Hidden Layers**
   ```python
   Layer_1 = tf.keras.layers.Dense(units=25, activation='sigmoid')
   a1 = Layer_1(x)
   
   Layer_2 = tf.keras.layers.Dense(units=10, activation='sigmoid')
   a2 = Layer_2(a1)
   ```
3. **Output Layer (Predicted Digit)**
   ```python
   prediction = np.argmax(a2)  # Find the index with the highest probability
   ```
   - If `prediction = 7`, the network thinks the image contains a **7**.

### **Industry Usage**
- **Google Lens**: Recognizes text in images and converts it into editable text.
- **Apple Face ID**: Uses neural networks to recognize faces securely.
- **Tesla’s Self-Driving AI**: Uses image classification to detect traffic signs and pedestrians.

---

## **Key Takeaways**
1. **Neural networks can be used for diverse applications**—from coffee roasting to image recognition.
2. **Forward propagation** is the process where inputs pass through layers to generate predictions.
3. **TensorFlow makes it easy** to define and run deep learning models.
4. **Thresholding at 0.5** helps classify binary outputs in classification tasks.
5. **Real-world applications include AI-powered coffee roasters, self-driving cars, and text recognition tools.**

---

## **What’s Next?**
Now that you understand forward propagation, the next step is learning how to **train a neural network** using **backpropagation and optimization techniques**.



---
## Next Section
- ### [Data in TensorFlow](Data_in_TensorFlow.md)
