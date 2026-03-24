import pandas as pd
import numpy as np
import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# ── CONFIG ────────────────────────────────────────────────────────────────────

NODE_HEALTH_PATH    = "data/processed/node_health.csv"
RISK_SCORES_PATH    = "data/processed/risk_scores.csv"
COX_REPORT_PATH     = "models/artifacts/cox_report.json"
DRIFT_SUMMARY_PATH  = "monitoring/reports/drift_summary.json"
MLFLOW_UI_URL       = "http://127.0.0.1:5000"

app = FastAPI(
    title       = "ComputeGuard Dashboard",
    description = "Real-time GPU node health monitoring",
    version     = "1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory="dashboard/static"),
    name="static",
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_node_health():
    df = pd.read_csv(NODE_HEALTH_PATH)
    df["risk_score"] = df["risk_score"].fillna(0)
    df["estimated_days_remaining"] = df[
        "estimated_days_remaining"
    ].fillna(365)
    return df

def load_model_metrics():
    metrics = {
        "xgboost": {
            "name"     : "computeguard-failure-predictor",
            "stage"    : "Production",
            "roc_auc"  : 0.0,
            "f1_score" : 0.0,
            "precision": 0.0,
            "recall"   : 0.0,
        },
        "survival": {
            "name"   : "computeguard-survival-model",
            "stage"  : "Registered",
            "c_index": 0.0,
        }
    }

    if os.path.exists(COX_REPORT_PATH):
        with open(COX_REPORT_PATH) as f:
            cox = json.load(f)
        metrics["survival"]["c_index"] = cox.get("c_index", 0.0)

    return metrics

def load_drift_summary():
    if not os.path.exists(DRIFT_SUMMARY_PATH):
        return {"drift_share": 0.0, "retrain_needed": False}
    with open(DRIFT_SUMMARY_PATH) as f:
        return json.load(f)

# ── REST ENDPOINTS ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("dashboard/static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/api/summary")
async def get_summary():
    df      = load_node_health()
    counts  = df["health_status"].value_counts().to_dict()
    total   = len(df)

    return JSONResponse({
        "total"   : total,
        "critical": int(counts.get("critical", 0)),
        "warning" : int(counts.get("warning",  0)),
        "healthy" : int(counts.get("healthy",  0)),
        "timestamp": datetime.now().isoformat(),
    })

@app.get("/api/nodes")
async def get_nodes(limit: int = 50, status: str = None):
    df = load_node_health()

    if status:
        df = df[df["health_status"] == status]

    df = df.head(limit)

    records = []
    for _, row in df.iterrows():
        records.append({
            "serial_number"           : row["serial_number"],
            "model"                   : row["model"],
            "risk_score"              : round(float(row["risk_score"]), 4),
            "estimated_days_remaining": round(float(row["estimated_days_remaining"]), 1),
            "health_status"           : row["health_status"],
            "risk_tier"               : row.get("risk_tier", "low"),
        })

    return JSONResponse({"nodes": records, "count": len(records)})

@app.get("/api/metrics")
async def get_metrics():
    return JSONResponse({
        "models"       : load_model_metrics(),
        "drift"        : load_drift_summary(),
        "mlflow_url"   : MLFLOW_UI_URL,
        "timestamp"    : datetime.now().isoformat(),
    })

@app.get("/api/risk-distribution")
async def get_risk_distribution():
    df     = load_node_health()
    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["bin"] = pd.cut(
        df["risk_score"], bins=bins, labels=labels, include_lowest=True
    )
    dist = df["bin"].value_counts().sort_index().to_dict()
    return JSONResponse({
        "labels": labels,
        "values": [int(dist.get(l, 0)) for l in labels],
    })

# ── WEBSOCKET — LIVE TELEMETRY STREAM ─────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"  WS connected  — total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        print(f"  WS disconnected — total: {len(self.active)}")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

def simulate_telemetry(df: pd.DataFrame) -> dict:
    """
    Simulates a real-time telemetry update.
    In production this would read from a live metrics endpoint.
    Picks a random sample of nodes and slightly mutates their scores
    to simulate changing health states over time.
    """
    sample = df.sample(min(20, len(df))).copy()

    # Add small random noise to risk scores to simulate live changes
    sample["risk_score"] = (
        sample["risk_score"] + np.random.normal(0, 0.02, len(sample))
    ).clip(0, 1)

    # Randomly escalate a node occasionally
    if random.random() < 0.15:
        idx = sample.sample(1).index[0]
        sample.loc[idx, "risk_score"] = round(
            random.uniform(0.75, 0.99), 4
        )
        sample.loc[idx, "health_status"] = "critical"

    nodes = []
    for _, row in sample.iterrows():
        risk = float(row["risk_score"])
        if risk >= 0.85:
            status = "critical"
        elif risk >= 0.60:
            status = "warning"
        else:
            status = "healthy"

        nodes.append({
            "serial_number"           : row["serial_number"],
            "model"                   : str(row["model"])[:20],
            "risk_score"              : round(risk, 4),
            "estimated_days_remaining": round(
                float(row["estimated_days_remaining"]), 1
            ),
            "health_status"           : status,
        })

    counts = {"critical": 0, "warning": 0, "healthy": 0}
    for n in nodes:
        counts[n["health_status"]] += 1

    return {
        "type"     : "telemetry_update",
        "timestamp": datetime.now().isoformat(),
        "nodes"    : nodes,
        "summary"  : counts,
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    df = load_node_health()
    try:
        while True:
            payload = simulate_telemetry(df)
            await manager.broadcast(payload)
            await asyncio.sleep(2)   # update every 2 seconds
    except WebSocketDisconnect:
        manager.disconnect(ws)