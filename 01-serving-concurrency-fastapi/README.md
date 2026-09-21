# Lab 01: High-Throughput ML Serving & Thread Pool Concurrency

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Async%20%2B%20ThreadPoolExecutor-orange.svg)]()
[![MLOps](https://img.shields.io/badge/MLOps-Level%201%20Serving-purple.svg)]()

> **Audience**: MLOps beginners & AI Engineering students  
> **Core Focus**: **Infrastructure & Concurrency Architecture** (How to prevent CPU-heavy ML predictions from freezing your API web server).

---

## 📑 Table of Contents
1. [Why This Lab Matters (The Infra Perspective)](#1-why-this-lab-matters-the-infra-perspective)
2. [The Core Concept: Deep Dive into Lines 173–198](#2-the-core-concept-deep-dive-into-lines-173198)
3. [Complete Beginner Setup Guide (Using VS Code)](#3-complete-beginner-setup-guide-using-vs-code)
4. [Running the Lab Step-by-Step](#4-running-the-lab-step-by-step)
5. [Understanding Production Latency Metrics (P50, P95, P99)](#5-understanding-production-latency-metrics-p50-p95-p99)
6. [Key MLOps Takeaways & Interview Questions](#6-key-mlops-takeaways--interview-questions)

---

## 1. Why This Lab Matters (The Infra Perspective)

In traditional Machine Learning courses, you are taught how to train a model (`model.fit()`) and evaluate accuracy. 

In **MLOps and Production Engineering**, our primary focus shifts:
- How does the server behave when **100 users hit `/predict` simultaneously**?
- Does one user's heavy prediction block another user's request?
- Can Kubernetes health probes (`/health`) still respond while the CPU is busy computing probabilities?

### The Core Problem: CPU-Bound ML vs. Async Web Servers
- **FastAPI** runs on an **asynchronous event loop** on a single main thread. It is designed to handle thousands of I/O requests (waiting for databases or network calls) without breaking a sweat.
- **ML Inference (`model.predict()`)** is **CPU-bound**. It runs pure math (NumPy array operations, decision tree traversals, matrix multiplications) that consume 100% of a CPU core until it finishes.
- **The Disaster**: If you run `model.predict()` directly inside an async endpoint, it will **block the entire event loop**. Every other client, ping, and `/health` check will freeze until that prediction finishes!

---

## 2. The Core Concept: Deep Dive into Lines 173–198

> 💡 **Notice**: Everything else in this project (training the RandomForest, Pydantic schemas, routes) is just **scaffolding**. The code below in `app.py` is the **heart of the infrastructure architecture**.

```python
# =============================================================================
# SECTION 6 – API ENDPOINT: /predict (app.py: Lines 173–198)
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
    # 1. Grab reference to the active asyncio event loop
    loop = asyncio.get_running_loop()

    # 2. OFFLOAD CPU WORK: Bridge async world to thread pool world
    result = await loop.run_in_executor(
        thread_pool,               # Bounded pool of worker threads (e.g. 4 threads)
        run_prediction,            # Synchronous CPU-bound function to execute
        request.features           # Input argument passed to run_prediction
    )

    # 3. Return response immediately to THIS client as soon as thread finishes
    return result
```

### How the Request Concurrency Flow Works

```mermaid
sequenceDiagram
    autonumber
    actor Client1 as Client 1 (Req #1)
    actor Client2 as Client 2 (Req #2)
    participant Loop as Async Event Loop (Single Thread)
    participant Pool as ThreadPoolExecutor (Workers: 1-4)
    participant Model as model.predict() (CPU Math)

    Client1->>Loop: POST /predict
    Note over Loop: Event loop stays UNBLOCKED!
    Loop->>Pool: loop.run_in_executor(thread_pool, run_prediction)
    Pool->>Model: Worker Thread 1 runs inference
    
    Client2->>Loop: POST /predict (Arrives immediately after)
    Note over Loop: Event loop receives it without delay!
    Loop->>Pool: loop.run_in_executor(thread_pool, run_prediction)
    Pool->>Model: Worker Thread 2 runs inference in parallel
    
    Model-->>Pool: Worker 1 finishes
    Pool-->>Loop: Notify completion
    Loop-->>Client1: Return 200 OK + JSON Response

    Model-->>Pool: Worker 2 finishes
    Pool-->>Loop: Notify completion
    Loop-->>Client2: Return 200 OK + JSON Response
```

### Why Each Line Matters:
1. **`async def predict(...)`**: Tells FastAPI that this route is a coroutine managed by the event loop.
2. **`loop = asyncio.get_running_loop()`**: Obtains a reference to the single-threaded scheduler currently executing.
3. **`thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())`**: Allocates a fixed, bounded set of OS worker threads dedicated to crunching numbers.
4. **`await loop.run_in_executor(...)`**: 
   - **Hands off** the heavy `run_prediction` computation to an idle thread in the thread pool.
   - **Suspends** only this specific client's coroutine without halting the loop.
   - While the thread is calculating the prediction, the event loop is 100% free to accept requests from Client 2, Client 3, or answer `/health`.
5. **`return result`**: When the thread pool finishes calculating, the coroutine wakes up and immediately streams the JSON response back to the client.

---

## 3. Complete Beginner Setup Guide (Using VS Code)

Follow these exact steps to set up the environment from scratch.

### Step 1: Install Python
1. Download Python 3.10, 3.11, or 3.12 from [python.org](https://www.python.org/downloads/).
2. **CRITICAL ON WINDOWS**: On the first installer screen, **CHECK THE BOX**:
   > ☑ **"Add python.exe to PATH"**
3. Complete the installation.
4. Verify by opening a terminal / command prompt and typing:
   ```bash
   python --version
   ```

### Step 2: Open Project in VS Code
1. Open **Visual Studio Code**.
2. Go to **File > Open Folder...** and select this directory (`hands_on`).
3. If prompted, click **"Yes, I trust the authors"**.
4. Open the Extensions tab (`Ctrl + Shift + X` on Windows / `Cmd + Shift + X` on Mac).
5. Search for **Python** (by Microsoft) and click **Install**.

### Step 3: Open the Integrated Terminal
- Press `` Ctrl + ` `` (backtick) or go to **Terminal > New Terminal** in the top menu.
- Make sure your terminal is opened in the project folder.

### Step 4: Create a Virtual Environment (`venv`)
A virtual environment isolates this project's dependencies so they do not conflict with your global Python installation.

Run in your terminal:
```bash
python -m venv venv
```
*(You will see a new folder named `venv` appear in your file explorer).*

### Step 5: Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
> ⚠️ **If you get an execution policy error on Windows PowerShell**, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run `.\venv\Scripts\Activate.ps1` again.

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
source venv/bin/activate
```

*(When activated, you will see `(venv)` at the beginning of your terminal prompt).*

### Step 6: Install Project Dependencies
Run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Running the Lab Step-by-Step

### Step 1: Train and Save the Model Artifact
Before running the server, we need a trained model artifact (`model.joblib`).

```bash
python train_model.py
```
**Expected Output:**
```text
✅ Model trained successfully.
✅ Model saved to 'model.joblib'.
   Run 'python app.py' to start the API server.
```

---

### Step 2: Start the FastAPI Inference Server
Launch the application using `uvicorn`:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Expected Output:**
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
🚀 Loading model into memory...
✅ Model loaded: <class 'sklearn.pipeline.Pipeline'>
✅ Thread pool ready with 4 worker threads.
INFO:     Application startup complete.
```

---

### Step 3: Verify in Browser & Test `/health`
Open your web browser and check:
1. **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
   - You should see: `{"status":"ok","model_loaded":true,"thread_pool_size":4}`
2. **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - You can test predictions manually by clicking `POST /predict` > **Try it out** > **Execute**.

---

### Step 4: Run the 100-Request Concurrency Load Test
Keep the server terminal running! Open a **second terminal** in VS Code (click the `+` icon in the terminal window), activate your venv, and run:

```bash
python load_test.py
```

`load_test.py` fires **100 asynchronous HTTP requests simultaneously** to stress-test your thread pool and measures response latency.

**Sample Benchmark Output:**
```text
============================================================
 LOAD TEST RESULTS: 100 concurrent requests
============================================================
 Total time taken : 0.42 seconds
 Successful       : 100 / 100
 Failed           : 0 / 100
 Requests/second  : 238.1 req/s
------------------------------------------------------------
 LATENCY PERCENTILES:
  P50 (Median)   :  12.4 ms  <-- 50% of requests finished faster than this
  P95            :  24.1 ms  <-- 95% of requests finished faster than this
  P99            :  31.8 ms  <-- The slowest 1% of requests
  Max Latency    :  35.2 ms
============================================================
```

---

## 5. Understanding Production Latency Metrics (P50, P95, P99)

In MLOps and infrastructure monitoring, **we never rely on "average latency"**. Averages hide severe bottlenecks. Instead, we look at **percentiles**:

| Metric | What It Means | Why Infra Engineers Care |
| :--- | :--- | :--- |
| **P50 (Median)** | 50% of your requests were faster than this value. | Represents the typical experience of a normal user. |
| **P95** | 95% of requests were faster than this value; only 5% were slower. | Standard metric for **Service Level Agreements (SLAs)** in enterprise APIs. |
| **P99 (Tail Latency)** | The worst 1% of user requests. | Shows whether requests are getting stuck in the thread pool queue when high traffic spikes occur. |

---

## 6. Key MLOps Takeaways & Interview Questions

### Q1: Why not just use `def predict(...)` without `async`?
> When you declare a route as `def predict(...)` (synchronous), FastAPI automatically sends the whole request to its internal default anyio worker thread pool. However:
> 1. You lose direct control over thread pool boundaries and sizing (`max_workers`).
> 2. Using explicit `ThreadPoolExecutor` bounded to CPU cores gives you predictable memory usage and prevents CPU thrashing.

### Q2: What happens if 1,000 requests hit this endpoint at once?
> The event loop accepts all 1,000 connections instantly without dropping them. The first 4 are executed immediately on worker threads. The remaining 996 wait in the `ThreadPoolExecutor` internal queue in memory and are processed as threads free up. Tail latency (P99) increases, but the server does not crash or reject requests!

### Q3: How do we scale this further in production?
1. **Scale Workers**: Increase Uvicorn process workers (`uvicorn app:app --workers 4`).
2. **Containerize**: Package into Docker and run behind Nginx or an API Gateway.
3. **Horizontal Pod Autoscaling (HPA)**: Deploy to Kubernetes and scale replicas when CPU utilization exceeds 70%.
