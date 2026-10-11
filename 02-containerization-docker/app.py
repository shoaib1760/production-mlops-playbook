# =============================================================================
# app.py
# PURPOSE: Serve a pre-trained tabular model via FastAPI inside Docker.
#          Uses ThreadPoolExecutor to prevent CPU-bound inference from
#          blocking the async ASGI event loop.
# =============================================================================

import os
import asyncio
import joblib
import numpy as np
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =============================================================================
# SECTION 1 – GLOBAL RESOURCES
# =============================================================================

THREAD_POOL_SIZE = os.cpu_count() or 4
MODEL_PATH = "model.joblib"

model = None
thread_pool = None


# =============================================================================
# SECTION 2 – APP LIFESPAN (Startup & Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, thread_pool

    # 1. Ensure model exists; train fallback if missing
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ '{MODEL_PATH}' not found on startup! Generating model artifact...")
        from train_model import train_and_save_model
        train_and_save_model(MODEL_PATH)

    print("🚀 Loading model into memory...")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully: {type(model).__name__}")

    # 2. Initialize bounded worker thread pool
    thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
    print(f"✅ Thread pool initialized with {THREAD_POOL_SIZE} worker threads.")

    yield  # Server handles incoming requests

    # 3. Clean shutdown
    print("🛑 Shutting down thread pool...")
    thread_pool.shutdown(wait=True)
    print("✅ Clean shutdown completed.")


# =============================================================================
# SECTION 3 – FASTAPI APP INSTANCE
# =============================================================================

app = FastAPI(
    title="Containerized ML Inference API",
    description="Production-grade ML serving container with thread pool concurrency.",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# SECTION 4 – REQUEST & RESPONSE SCHEMAS
# =============================================================================

class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 numeric feature values for classification.",
        examples=[[1.2, -0.5, 3.1, 0.0, 2.7]]
    )


class PredictResponse(BaseModel):
    predicted_class: int = Field(..., description="0 or 1")
    probability: float = Field(..., description="Model confidence score")


# =============================================================================
# SECTION 5 – SYNCHRONOUS INFERENCE WORKER
# =============================================================================

def run_prediction(features: list[float]) -> dict:
    """
    Synchronous CPU-bound inference execution executed inside worker threads.
    """
    X = np.array(features).reshape(1, -1)
    probabilities = model.predict_proba(X)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    return {"predicted_class": predicted_class, "probability": confidence}


# =============================================================================
# SECTION 6 – ENDPOINTS: /health and /predict
# =============================================================================

@app.get("/health")
async def health():
    """
    Liveness & readiness probe endpoint.
    Used by Docker HEALTHCHECK and Kubernetes probes.
    """
    return {
        "status": "healthy",
        "containerized": True,
        "model_loaded": model is not None,
        "thread_pool_size": THREAD_POOL_SIZE
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Accepts 5 features and offloads CPU computation to ThreadPoolExecutor.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        thread_pool,
        run_prediction,
        request.features
    )
    return result


if __name__ == "__main__":
    import uvicorn
    # When running directly: bind to 0.0.0.0 so external/container traffic is reachable
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
