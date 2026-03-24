# ComputeGuard — GPU Node Failure Prediction

> Predicts GPU node failures before they happen using machine learning on
> drive telemetry data, with a full MLOps pipeline and real-time dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.18-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![DVC](https://img.shields.io/badge/DVC-3.56-orange)

---

## The problem

GPU infrastructure companies wake up every morning asking:
**which nodes will fail today — and how soon?**

A single unplanned GPU failure in a compute cluster can take hours
to diagnose and costs thousands of dollars in lost compute time.
ComputeGuard answers this question before it becomes an incident.

---

## Architecture
```
Raw telemetry (Backblaze)
        │
        ▼
DuckDB SQL pipeline ──► 5 data quality checks ──► features.csv (DVC)
        │
        ▼
XGBoost classifier ──► risk_score per node (0–1)
        │
Cox PH survival model ──► estimated_days_remaining per node
        │
        ▼
node_health.csv ──► FastAPI + WebSocket ──► Live dashboard
        │
        ▼
Evidently AI drift detection ──► auto-retrain ──► MLflow registry
        │
        ▼
GitHub Actions CI/CD ──► tests on every push
```

---

## Models

| Model | Purpose | Metric |
|---|---|---|
| XGBoost classifier | Failure probability in 30d | ROC AUC ~0.92 |
| Cox Proportional Hazards | Time-to-failure estimate | C-index ~0.78 |

---

## MLOps stack

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking + model registry |
| DVC | Data version control |
| Evidently AI | Data drift + model performance monitoring |
| GitHub Actions | CI/CD — tests on push, drift check daily |

---

## Project structure
```
computeguard/
├── ingestion/          Data download + DuckDB pipeline
├── models/             XGBoost + Cox PH + auto-promote
├── monitoring/         Evidently drift detection + alerts
├── dashboard/          FastAPI + WebSocket real-time UI
├── tests/              13 pytest unit tests
└── .github/workflows/  CI on push + drift check daily
```

---

## Running locally
```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data
python ingestion/download_data.py

# 3. Run pipeline
python ingestion/pipeline.py

# 4. Train models
python models/train.py
python models/survival.py

# 5. Run drift detection
python monitoring/drift_detection.py

# 6. Start dashboard
uvicorn dashboard.main:app --port 8000
```

Open `http://127.0.0.1:8000`

---

## Data

Uses the public [Backblaze Hard Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
dataset as a proxy for GPU telemetry. In production, this pipeline
would ingest DCGM or IPMI metrics from live GPU nodes.

---

## Key features

- **12 engineered features** — 7-day rolling averages, 30-day trends,
  day-over-day deltas, drive age
- **Class imbalance handling** — `scale_pos_weight` in XGBoost
- **Time-based train/test split** — more realistic than random split
  for time series data
- **Survival analysis** — not just "will it fail" but "when will it fail"
- **Full MLOps loop** — drift → retrain → compare → promote, zero manual steps
- **Real-time WebSocket dashboard** — live node health grid, alert feed,
  model metrics panel