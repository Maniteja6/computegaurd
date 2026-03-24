import pytest
import pandas as pd
import os
import json
import pickle

RISK_SCORES_PATH = "data/processed/risk_scores.csv"
NODE_HEALTH_PATH = "data/processed/node_health.csv"
COX_MODEL_PATH   = "models/artifacts/cox_model.pkl"
COX_REPORT_PATH  = "models/artifacts/cox_report.json"

def test_risk_scores_exist():
    assert os.path.exists(RISK_SCORES_PATH), \
        "risk_scores.csv not found"

def test_risk_scores_range():
    df = pd.read_csv(RISK_SCORES_PATH)
    assert "risk_score" in df.columns, "risk_score column missing"
    assert df["risk_score"].between(0, 1).all(), \
        "risk_score values outside 0-1 range"

def test_risk_tiers_valid():
    df  = pd.read_csv(RISK_SCORES_PATH)
    valid_tiers = {"low", "medium", "high"}
    actual      = set(df["risk_tier"].dropna().unique())
    assert actual.issubset(valid_tiers), \
        f"Invalid risk tiers found: {actual - valid_tiers}"

def test_node_health_exists():
    assert os.path.exists(NODE_HEALTH_PATH), \
        "node_health.csv not found"

def test_node_health_status_values():
    df     = pd.read_csv(NODE_HEALTH_PATH)
    valid  = {"healthy", "warning", "critical"}
    actual = set(df["health_status"].dropna().unique())
    assert actual.issubset(valid), \
        f"Invalid health_status values: {actual - valid}"

def test_cox_model_loadable():
    assert os.path.exists(COX_MODEL_PATH), \
        "cox_model.pkl not found"
    with open(COX_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Cox model failed to load"

def test_cox_cindex_acceptable():
    assert os.path.exists(COX_REPORT_PATH), \
        "cox_report.json not found"
    with open(COX_REPORT_PATH) as f:
        report = json.load(f)
    c_index = report["c_index"]
    assert c_index > 0.55, \
        f"C-index too low: {c_index} (must be > 0.55)"