# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the Iris dataset, create a DataFrame, and split it into features (X) and target (y).

2.Data Splitting: Split the data into training and testing sets with an 80-20 ratio using train_test_split.

3.Model Training: Initialize an SGDClassifier and train it on the training data (X_train, y_train).

4.Prediction and Evaluation: Predict on the test set, calculate accuracy, and generate a confusion matrix for evaluation.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SURYAMALARV
RegisterNumber:  212223230224
*/
```
```
# Import libraries.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
```
# Load the Iris dataset
iris = load_iris()
```
```
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
```
```
# Display the first few rows of the dataset
print(df.head())
```
![image](https://github.com/user-attachments/assets/edf572f7-abe4-4b46-b6fb-bb698e231773)
```
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
```
```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
```
```
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/a126b0c9-28b6-4013-a4d5-43251c9733ed)
```
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
```
```
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![image](https://github.com/user-attachments/assets/6e984674-0a8f-436c-a5dc-6df52a742d54)
```
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
![image](https://github.com/user-attachments/assets/f0adf230-1b63-4a65-8102-5441bff15a98)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
