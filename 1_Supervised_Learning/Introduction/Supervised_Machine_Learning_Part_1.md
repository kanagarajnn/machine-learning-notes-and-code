# Supervised Machine Learning Part 1

## 1. Economic Impact of Machine Learning
- 99% of economic value from machine learning today is driven by **supervised learning**.
- Supervised learning maps input **x** to output **y** using labeled data.

## 2. Definition of Supervised Learning
- Supervised learning algorithms learn from **input-output pairs**.
- The model predicts an output given a new input based on previous examples.

## 3. Real-World Applications of Supervised Learning
- **Spam Filtering**: Determines if an email is spam or not.
- **Speech Recognition**: Converts audio clips to text.
- **Machine Translation**: Translates text between languages.
- **Online Advertising**: Predicts if a user will click on an ad.
- **Self-Driving Cars**: Identifies other vehicles using sensor data.
- **Manufacturing**: Uses **visual inspection** to detect defects in products.

    | **Input (X)**            | **Output (Y)**                | **Application**          |
    |---------------------------|-------------------------------|--------------------------|
    | email                    | spam? (0/1)                  | spam filtering           |
    | audio                    | text transcripts             | speech recognition       |
    | English                  | Spanish                      | machine translation      |
    | ad, user info            | click? (0/1)                 | online advertising       |
    | image, radar info        | position of other cars       | self-driving car         |
    | image of phone           | defect? (0/1)                | visual inspection        |


## 4. Training the Model
- Supervised learning requires a dataset containing **input x** and the corresponding **correct output y**.
- After training, the model can make predictions on new, unseen data.

## 5. Example: Predicting House Prices
- A model can predict house prices based on square footage.
- A simple model might fit a **straight line**, while a more complex model could fit a **curve**.
- The model systematically chooses the most appropriate function for the data.

## 6. Regression vs. Classification
- **Regression**: Predicts continuous numerical values (e.g., house prices, temperatures).
- **Classification**: Categorizes data into distinct classes (e.g., spam vs. not spam).

## 7. Importance of Choosing the Right Model
- The choice between a simple model (e.g., linear regression) and a complex model (e.g., polynomial regression) depends on the dataset and the prediction accuracy required.
- Overfitting and underfitting are key considerations when selecting a model.

## 8. Next Steps in Learning
- The next topic explores **classification**, the second major type of supervised learning.
- Learners will understand how classification problems differ from regression problems.

---
## Next Section
- [Supervised Machine Learning Part 2](Supervised_Machine_Learning_Part_2.md)
