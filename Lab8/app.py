# app.py
from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

app = Flask(__name__)

@app.route('/')
def index():
    # Load the Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Prepare data for rendering
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization
    }

    return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)