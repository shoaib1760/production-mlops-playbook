# =============================================================================
# app.py
# PURPOSE: Serve a pre-trained tabular model via FastAPI.
#          Handles 100 concurrent requests safely using a ThreadPoolExecutor.
#
# KEY CONCEPTS IN THIS FILE:
#   1. lifespan  → load model once at startup, clean up at shutdown
#   2. Pydantic  → validate every incoming JSON request automatically
#   3. ThreadPoolExecutor → run model.predict() off the event loop thread
#   4. run_in_executor    → the bridge between async world and thread world
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
# These are created ONCE when the server starts and shared across ALL requests.
# =============================================================================

# Number of worker threads in the pool.
# Rule of thumb for CPU-bound ML tasks: match your CPU core count.
# os.cpu_count() returns 4 on a quad-core machine, for example.
THREAD_POOL_SIZE = os.cpu_count() or 4

# These will be populated inside the lifespan function below.
model = None           # The single loaded model instance (shared, read-only)
thread_pool = None     # The single ThreadPoolExecutor (shared, bounded)


# =============================================================================
# SECTION 2 – APP LIFESPAN (Startup & Shutdown)
#
# WHY lifespan instead of @app.on_event("startup")?
#   It's the modern FastAPI approach (v0.93+).  It uses a single async
#   generator that yields once: code BEFORE yield = startup, AFTER = shutdown.
#   This keeps setup/teardown logic together and is cleaner.
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────
    global model, thread_pool

    # Load the serialized model from disk into memory.
    # This happens ONCE at startup — not on every request!
    # After this, 'model' lives in RAM and all threads share the same object.
    print("🚀 Loading model into memory...")
    model = joblib.load("model.joblib")
    print(f"✅ Model loaded: {type(model)}")

    # Create the thread pool with a fixed number of worker threads.
    # max_workers=4 (or cpu_count) means at most 4 predictions happen in
    # parallel at any given moment; others wait briefly in an internal queue.
    thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
    print(f"✅ Thread pool ready with {THREAD_POOL_SIZE} worker threads.")

    yield  # ← Server is now running and accepting requests

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # Gracefully signal threads to finish current work and then exit.
    # shutdown(wait=True) blocks until all in-progress predictions complete.
    thread_pool.shutdown(wait=True)
    print("🛑 Thread pool shut down cleanly.")


# =============================================================================
# SECTION 3 – FASTAPI APP INSTANCE
# =============================================================================

app = FastAPI(
    title="Tabular Model Inference API",
    description="Serves a pre-trained Scikit-Learn model with async thread pooling.",
    version="1.0.0",
    lifespan=lifespan   # wire up our startup/shutdown logic
)


# =============================================================================
# SECTION 4 – REQUEST / RESPONSE SCHEMAS (Pydantic)
#
# WHY Pydantic?
#   FastAPI uses Pydantic to automatically:
#     • Parse incoming JSON into a Python object
#     • Validate data types (e.g. features must be floats, not strings)
#     • Return a clear 422 Unprocessable Entity error if validation fails
#   This means model.predict() never receives malformed data.
# =============================================================================

class PredictRequest(BaseModel):
    """Schema for a single prediction request."""
    features: list[float] = Field(
        ...,                              # required field (no default)
        min_length=5,
        max_length=5,
        description="Exactly 5 numeric feature values for the tabular model.",
        examples=[[1.2, -0.5, 3.1, 0.0, 2.7]]
    )


class PredictResponse(BaseModel):
    """Schema for the prediction response."""
    predicted_class: int   = Field(..., description="0 or 1 (binary classification)")
    probability:     float = Field(..., description="Confidence score for predicted class")


# =============================================================================
# SECTION 5 – HELPER FUNCTION: run_prediction_in_thread()
#
# WHY wrap model.predict() in a plain function (not async)?
#   ThreadPoolExecutor.submit() / run_in_executor() expects a REGULAR
#   (synchronous) callable — not a coroutine.  The thread pool will call
#   this function inside one of its worker threads.
# =============================================================================

def run_prediction(features: list[float]) -> dict:
    """
    Runs model inference synchronously.
    This function executes inside a worker thread, NOT on the event loop.

    Steps:
      1. Convert the list of floats into a 2-D NumPy array.
         model.predict() expects shape (n_samples, n_features).
         For a single sample that is (1, 5).
      2. Call model.predict_proba() to get class probabilities.
      3. Pick the highest-probability class as the prediction.
    """
    # Step 1: reshape [1.2, -0.5, 3.1, 0.0, 2.7] → [[1.2, -0.5, 3.1, 0.0, 2.7]]
    X = np.array(features).reshape(1, -1)   # shape: (1, 5)

    # Step 2: predict_proba returns [[prob_class_0, prob_class_1]]
    probabilities = model.predict_proba(X)[0]  # shape: (2,)

    # Step 3: argmax picks index of the highest probability → predicted class
    predicted_class = int(np.argmax(probabilities))
    confidence      = float(probabilities[predicted_class])

    return {"predicted_class": predicted_class, "probability": confidence}


# =============================================================================
# SECTION 6 – API ENDPOINT: /predict
#
# THE CONCURRENCY FLOW for 100 simultaneous requests:
#
#  Client 1..100 ──► Event Loop (async, non-blocking)
#                         │
#                         │  asyncio.get_running_loop().run_in_executor(
#                         │      thread_pool, run_prediction, features
#                         │  )
#                         │
#                    Thread Pool Queue
#                   ┌──────────────────┐
#                   │ [req1][req2]...  │  ← all 100 tasks queued instantly
#                   └────────┬─────────┘
#                   Workers execute:
#                   [Thread1] [Thread2] [Thread3] [Thread4]
#                         │
#                    Each thread calls run_prediction() independently.
#                    As each finishes, the event loop gets the result and
#                    immediately sends the HTTP response to THAT client.
#                    No client waits for other clients!
# =============================================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Accepts a JSON body with 5 feature values and returns a class prediction.

    This endpoint is 'async def' so FastAPI runs it on the event loop.
    We use run_in_executor() to offload the CPU-bound model.predict()
    call to a worker thread — keeping the event loop free to accept more
    incoming requests while prediction is happening in the background.
    """
    # Get a reference to the currently running asyncio event loop.
    loop = asyncio.get_running_loop()

    # Offload run_prediction() to a worker thread in our thread pool.
    # 'await' suspends THIS coroutine (freeing the loop for other requests)
    # until the worker thread finishes and returns the result dict.
    result = await loop.run_in_executor(
        thread_pool,               # which pool to use (our bounded 4-thread pool)
        run_prediction,            # the synchronous function to run
        request.features           # argument passed to run_prediction()
    )

    # By the time we reach here, this specific request's prediction is done.
    # FastAPI immediately serializes 'result' to JSON and sends it back
    # to THIS client — without waiting for any other request to finish.
    return result


# =============================================================================
# SECTION 7 – HEALTH CHECK ENDPOINT
#
# Standard practice in MLOps: a /health endpoint that load balancers and
# Kubernetes probes can call to verify the service is alive and the model
# is loaded.  Returns instantly without touching the model or thread pool.
# =============================================================================

@app.get("/health")
async def health():
    """Quick liveness check. Returns 200 if the server is running."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "thread_pool_size": THREAD_POOL_SIZE
    }
