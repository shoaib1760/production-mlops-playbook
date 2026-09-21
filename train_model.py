# =============================================================================
# train_model.py
# PURPOSE: Train a simple tabular ML model and save it to disk.
#
# WHY THIS STEP?
#   In real MLOps, models are trained offline (in a pipeline / notebook),
#   serialized (saved to disk / model registry), then loaded by the serving
#   API at startup.  We simulate that here with a simple Scikit-Learn model.
# =============================================================================

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 1. Create a synthetic tabular dataset ────────────────────────────────────
#   make_classification() generates a random binary-classification dataset.
#   n_features=5 means each "row" has 5 numeric feature columns.
X, y = make_classification(
    n_samples=1000,   # 1 000 training rows
    n_features=5,     # 5 input features (e.g. age, income, score, etc.)
    random_state=42   # fixed seed → reproducible data every time
)

# ── 2. Build a Scikit-Learn Pipeline ─────────────────────────────────────────
#   Pipeline chains steps in order.  During .predict() each step transforms
#   the data before passing it to the next step.
#
#   Step 1 - StandardScaler:
#     Normalises features to zero-mean / unit-variance.
#     Helps RandomForest converge slightly faster (less critical for trees,
#     but good practice to include in the pipeline).
#
#   Step 2 - RandomForestClassifier:
#     An ensemble of decision trees.
#     n_estimators=10 → 10 trees (small for speed; real models use 100+).
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
])

# ── 3. Train the model ───────────────────────────────────────────────────────
#   .fit() trains on the synthetic data.  In production this would happen
#   in a separate training job (Vertex AI, SageMaker, Kubeflow, etc.).
model_pipeline.fit(X, y)
print("✅ Model trained successfully.")

# ── 4. Save the trained model to disk ────────────────────────────────────────
#   joblib is the recommended serializer for Scikit-Learn objects because it
#   handles large NumPy arrays (model weights) more efficiently than pickle.
#   The saved file is the "model artifact" that the API will load at startup.
MODEL_PATH = "model.joblib"
joblib.dump(model_pipeline, MODEL_PATH)
print(f"✅ Model saved to '{MODEL_PATH}'.")
print("   Run 'python app.py' to start the API server.")
