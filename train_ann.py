import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
df = pd.read_excel("dataset.xlsx")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show column names
print("\nColumns:")
print(df.columns)

# Show dataset shape
print("\nShape:")
print(df.shape)
# Define input features (X)
X = df[['attendance', 'assignment', 'quiz', 'mid', 'study_hours']]

# Define target (y)
y = df['result']

# Show sample
print("\nX (features):")
print(X.head())

print("\ny (target):")
print(y.head())
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData scaled successfully!")

# Save scaler
joblib.dump(scaler, "scaler.joblib")

print("\nScaler saved as scaler.joblib")
from sklearn.neural_network import MLPClassifier

# Build model
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)

print("\nTraining Complete")
print("Iterations:", model.n_iter_)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
import joblib
joblib.dump(model, "model.joblib")

print("Model saved as model.joblib")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nEvaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")