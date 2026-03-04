# src/validate.py

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)


def main():

    os.makedirs("reports", exist_ok=True)

    # Load model
    model = joblib.load("models/model.pkl")

    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Save metrics
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("reports/roc_curve.png")
    plt.close()

    print("Validation complete. Metrics and plots saved.")


if __name__ == "__main__":
    main()