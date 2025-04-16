# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import visuals  # Import custom visualization module

# Load dataset
df = pd.read_csv("Diabetes Data/diabetes.csv")

# Add column names if not present
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Visualizations
visuals.plot_feature_distributions(df)
visuals.plot_correlation_heatmap(df)

# Split data into features and labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
visuals.plot_confusion_matrix(cm)


import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# save the trained model
joblib.dump(model, "diabetes_model.pkl")

