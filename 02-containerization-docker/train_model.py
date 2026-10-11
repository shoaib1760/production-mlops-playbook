# =============================================================================
# train_model.py
# PURPOSE: Train a tabular Scikit-Learn classification model and save it to disk.
# =============================================================================

import os
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.joblib"

def train_and_save_model(output_path: str = MODEL_PATH):
    """
    Trains a 5-feature binary classifier and serializes it via joblib.
    """
    print("⏳ Generating synthetic training data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        random_state=42
    )

    print("⏳ Building pipeline (StandardScaler + RandomForestClassifier)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
    ])

    print("⏳ Fitting model...")
    pipeline.fit(X, y)

    joblib.dump(pipeline, output_path)
    print(f"✅ Model successfully saved to '{output_path}'.")
    return pipeline

if __name__ == "__main__":
    train_and_save_model()
