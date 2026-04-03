# ML Leakage & Pipeline Assignment

# Imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Task 1 — WRONG APPROACH (Leakage)

print("===== Task 1: Data Leakage Example =====")

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# ❌ WRONG: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Train Accuracy (Leakage):", train_acc)
print("Test Accuracy (Leakage):", test_acc)

print("\nProblem:")
print("Scaling was done on the entire dataset before splitting, causing data leakage.")

# Task 2 — FIX USING PIPELINE

print("\n===== Task 2: Pipeline + Cross Validation =====")

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Std Deviation:", np.std(cv_scores))

# Train final model
pipeline.fit(X_train, y_train)

# Evaluate
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)

print("Train Accuracy (Pipeline):", train_acc)
print("Test Accuracy (Pipeline):", test_acc)

# ==============================
# Task 3 — Decision Tree Depth Experiment
# ==============================

print("\n===== Task 3: Decision Tree Depth Comparison =====")

depths = [1, 5, 20]

results = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    results.append([depth, train_acc, test_acc])

# Create table
df = pd.DataFrame(results, columns=["Max Depth", "Train Accuracy", "Test Accuracy"])
print("\nDecision Tree Results:")
print(df)

print("\nConclusion:")
print("Depth=5 usually gives the best balance between underfitting (depth=1) and overfitting (depth=20).")
