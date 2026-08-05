"""
Iris Flower Classification using Artificial Intelligence
----------------------------------------------------------
CodTech Internship Project

This program uses a Machine Learning model (Support Vector Machine)
to classify Iris flowers into three species:
    1. Iris-setosa
    2. Iris-versicolor
    3. Iris-virginica

based on four features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

Author: <Your Name>
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    """Load the Iris dataset and return features, labels and target names."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def main():
    print("=" * 55)
    print(" IRIS FLOWER CLASSIFICATION USING AI (SVM MODEL)")
    print("=" * 55)

    # Step 1: Load Dataset
    X, y, feature_names, target_names = load_data()
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]

    print("\n[1] Dataset Loaded Successfully")
    print(f"    Total Samples : {df.shape[0]}")
    print(f"    Features      : {feature_names}")
    print(f"    Classes       : {[str(name) for name in target_names]}")

    print("\n[2] Sample Data (first 5 rows):")
    print(df.head().to_string(index=False))

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[3] Data Split -> Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

    # Step 3: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Model Training
    model = SVC(kernel="linear", C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("\n[4] Model Training Complete (SVM - Linear Kernel)")

    # Step 5: Prediction
    y_pred = model.predict(X_test_scaled)

    # Step 6: Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[5] Model Accuracy: {acc * 100:.2f}%")

    print("\n[6] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())

    print("\n[7] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Step 7: Predict a New Sample
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example flower measurements
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    print("[8] Prediction for New Sample", sample.tolist(), "->",
          target_names[prediction[0]].upper())

    print("\n" + "=" * 55)
    print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
