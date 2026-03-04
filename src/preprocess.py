# src/preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/bank.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(RAW_PATH)

    # Encode target
    df["target"] = df["target"].map({"yes": 1, "no": 0})

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        if col != "target":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Split
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save train/test
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()