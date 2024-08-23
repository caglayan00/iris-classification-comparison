# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:14:23 2024

@author: caglayan00
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Import the dataset
dataset = pd.read_csv("iris.csv")

# Use only the first three features for the 3D plot
x = dataset.iloc[:, :3].values
y = dataset.iloc[:, -1].values

# Convert string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Splitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Feature Scaling
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# Classifiers to compare
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=0),
    "SVM (Linear)": SVC(kernel='linear', random_state=0),
    "SVM (RBF)": SVC(kernel='rbf', random_state=0),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0)
}

accuracy_results = {}


# Train and evaluate classifiers
for name, classifier in classifiers.items():
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[name] = accuracy
    
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "-"*50 + "\n")
    
# Plotting the comparison of accuracy
plt.figure(figsize=(10, 5))
plt.barh(list(accuracy_results.keys()), list(accuracy_results.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title("Comparison of Classification Algorithms")
plt.xlim(0, 1)
plt.show()
