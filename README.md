# Production MLOps Playbook 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Target-326CE5.svg)](https://kubernetes.io/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

> **A hands-on, progressive curriculum of battle-tested MLOps engineering blueprints.**  
> Moving beyond toy notebooks to solve real-world problems: **concurrency, CPU starvation, containerization, experiment tracking, observability, and automated scaling.**

---

## 🧭 Curriculum & Hands-on Labs

Each lab in this playbook is self-contained with its own code, architecture diagrams, benchmark reports, and step-by-step student guides.

| Lab | Topic | Focus Area | Status |
| :---: | :--- | :--- | :---: |
| **[01](./01-serving-concurrency-fastapi)** | **High-Throughput ML Serving & Concurrency** | Thread pool CPU isolation, async event loop starvation, P50/P95/P99 latency benchmarks under 100 concurrent requests |  **Completed** |
| **02** | **Production Containerization with Docker** | Multi-stage Docker builds, non-root security, resource limits (CPU/Memory), and health checks | 🔨 *In Progress* |
| **03** | **Experiment Tracking & Model Registry** | MLflow tracking server, parameter/metric logging, model staging, and lineage | 📅 *Planned* |
| **04** | **Automated Pipeline Orchestration** | Data ingestion, validation, and reproducible training pipelines with Prefect | 📅 *Planned* |
| **05** | **Production Monitoring & Data Drift** | Evidently AI, Prometheus metrics exporter, detecting feature distribution shift | 📅 *Planned* |
| **06** | **CI/CD for Machine Learning** | GitHub Actions automated model evaluation, smoke testing, and regression gates | 📅 *Planned* |
| **07** | **Kubernetes Deployment & Autoscaling** | K8s Deployments, Services, and Horizontal Pod Autoscaling (HPA) under load | 📅 *Planned* |

---

## 📂 Repository Structure

```text
production-mlops-playbook/
│
├── README.md                                  # You are here (Playbook overview & roadmap)
├── .gitignore                                 # Global Python / OS ignore rules
│
├── 01-serving-concurrency-fastapi/            # 📍 Lab 01: Serving & Concurrency
│   ├── README.md                              # Detailed lab guide & latency analysis
│   ├── app.py                                 # FastAPI app with ThreadPoolExecutor
│   ├── train_model.py                         # Model training & joblib serialization
│   ├── load_test.py                           # Async stress test (100 simultaneous requests)
│   ├── model.joblib                           # Serialized model artifact
│   └── requirements.txt                       # Minimal lab dependencies
│
├── 02-containerization-docker/                # 📍 Lab 02: Packaging & Docker
│   └── ...
│
└── ...
```

---

## 🛠️ General Prerequisites for Students

To work through any of the labs in this repository, you should have the following installed on your machine:

1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/) (*Ensure "Add Python to PATH" is checked on Windows*).
2. **Visual Studio Code**: Download from [code.visualstudio.com](https://code.visualstudio.com/).
   - Install the **Python** extension by Microsoft.
3. **Git**: Download from [git-scm.com](https://git-scm.com/).
4. **Docker Desktop** *(Needed starting from Lab 02)*: Download from [docker.com](https://www.docker.com/products/docker-desktop/).

---

## 🚀 Quickstart: Running Your First Lab

To dive right into **Lab 01 (High-Throughput ML Serving & Concurrency)**:

```bash
# 1. Clone this repository
git clone https://github.com/shoaib1760/production-mlops-playbook.git
cd production-mlops-playbook/01-serving-concurrency-fastapi

# 2. Follow the detailed step-by-step instructions in the Lab 01 README
```
👉 **Go directly to the [Lab 01 Guide](./01-serving-concurrency-fastapi/README.md)**.

---

## 👨‍💻 Author

- **Shoaib Abid(Senior AI/ML Engineer at Ibex)** - [@shoaib1760](https://github.com/shoaib1760)

---

## 📜 License

This repository is distributed under the MIT License. Feel free to use, adapt, and share it for your own learning and teaching.
