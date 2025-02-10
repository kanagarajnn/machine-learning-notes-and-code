# **Demand Prediction Example : Neural Networks**

## **Understanding Neural Networks with a Practical Example**
Imagine you are a retailer selling **T-shirts** and want to predict if a particular design will be a **top seller**. You have historical data on different T-shirts, their prices, and whether they became popular. Retailers like **Amazon and Walmart** use similar techniques for inventory management and marketing planning.

### **Step 1: Basic Logistic Regression**
- The **input feature (x)** is the price of the T-shirt.
- The **output (prediction)** is whether the T-shirt is a top seller (**1 for yes, 0 for no**).
- Logistic regression is applied using the **sigmoid function**:
  
  ```f(x) = 1 / (1 + e^(-wx + b))```

- This function helps predict the probability of a T-shirt becoming a bestseller.
- We redefine the output as **activation (a)** to align with neural network terminology. This concept originates from neuroscience, where a neuron "activates" to send signals.

## **Step 2: From Single Neuron to Neural Network**
A **single neuron** can model a basic logistic regression, but real-world problems are more complex. Let’s extend our example to include multiple factors affecting sales:
- **Price**
- **Shipping costs**
- **Marketing spend**
- **Material quality**

These factors influence three major aspects of customer perception:
1. **Affordability**: How cheap or expensive the product is.
2. **Awareness**: How well the product is marketed.
3. **Perceived Quality**: Customers often associate higher prices with higher quality.

We create **three neurons** to predict these values, and their outputs are fed into another neuron that predicts the overall probability of the T-shirt becoming a top seller.

## **Step 3: Layers in a Neural Network**
Neural networks are structured into **layers**:
- **Input Layer**: Takes in raw data (**price, shipping cost, etc.**).
- **Hidden Layer**: Processes and combines inputs to generate intermediate values (**affordability, awareness, quality**).
- **Output Layer**: Makes the final prediction (**probability of becoming a top seller**).

The **hidden layer** acts as a feature extractor, learning new representations automatically instead of relying on manually crafted features.

## **Step 4: Fully Connected Networks**
Instead of manually selecting which inputs go to which neurons, we **fully connect** all neurons, allowing the model to learn which features matter most. This means each neuron in a layer receives inputs from **all neurons in the previous layer**.

For example, in **Tesla’s Autopilot**, a neural network might receive inputs from:
- **Cameras** (visual data)
- **Radar** (distance measurement)
- **LIDAR** (3D environment mapping)

Each neuron processes this data to decide whether a car should **stop, accelerate, or turn**.

## **Step 5: Expanding Neural Networks with More Hidden Layers**
Neural networks can have **multiple hidden layers**. Each layer extracts increasingly complex features:
1. **First Hidden Layer**: Extracts basic features (e.g., edges in an image).
2. **Second Hidden Layer**: Detects more complex patterns (e.g., shapes like eyes, nose).
3. **Third Hidden Layer**: Recognizes entire objects (e.g., a face).

This is how **Face ID on iPhones** recognizes faces by breaking down images into layers of increasing complexity.

## **Step 6: Neural Networks Learn Their Own Features**
Traditional machine learning required **manual feature engineering**, such as:
- In **real estate pricing**, multiplying **lot width × lot depth** to calculate total area.
- In **spam detection**, counting the frequency of certain words.

Neural networks **automatically learn** the most relevant features from data. This is why deep learning has revolutionized fields like:
- **Speech recognition** (Siri, Google Assistant)
- **Medical imaging** (detecting cancer in X-rays)
- **Autonomous driving** (self-driving cars)

## **Step 7: Architecture Decisions in Neural Networks**
When designing a neural network, we must decide:
- **Number of hidden layers**
- **Number of neurons per layer**
- **Connections between layers**

This is called the **architecture of the neural network**. Choosing the right structure affects model performance.

For example:
- **Shallow networks** (1 hidden layer) are effective for simple tasks like predicting housing prices.
- **Deep networks** (multiple hidden layers) are better for complex tasks like **language translation** (**Google Translate**).

## **Step 8: Deep Learning and Multilayer Perceptrons**
When a neural network has **many hidden layers**, it is often called a **Multilayer Perceptron (MLP)**. These models are used in cutting-edge applications such as:
- **Chatbots** (ChatGPT, Alexa)
- **Image recognition** (Google Photos, Pinterest visual search)
- **Robotics** (Boston Dynamics robots)

## **Conclusion**
Neural networks have transformed machine learning by enabling models to **learn representations directly from data**. Unlike traditional methods that required manual feature selection, deep learning allows models to extract and refine features **automatically**, making them incredibly powerful for real-world applications.

In the next section, we will explore how these concepts apply to **computer vision and facial recognition**.

## Next Section
- ### [Recognizing Images](Recognizing_Images.md)
