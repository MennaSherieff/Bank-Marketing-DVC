# src/train.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():

    os.makedirs("models", exist_ok=True)

    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    # ===== CHOOSE MODEL HERE =====
    MODEL_TYPE = "logreg"  # change to "rf" in second branch

    if MODEL_TYPE == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif MODEL_TYPE == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    else:
        raise ValueError("Invalid model type")

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/model.pkl")

    print(f"Model trained and saved ({MODEL_TYPE}).")


if __name__ == "__main__":
    main()