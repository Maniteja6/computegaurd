import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

FEATURES_PATH    = "data/processed/features.csv"
DRIFT_REPORT_DIR = "monitoring/reports"
DRIFT_JSON_PATH  = "monitoring/reports/drift_summary.json"
DRIFT_HTML_PATH  = "monitoring/reports/drift_report.html"

DRIFT_THRESHOLD  = 0.15   # if dataset drift share > 15% → trigger retrain
MISSING_THRESHOLD= 0.05   # if missing value rate > 5%  → alert

MONITOR_FEATURES = [
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_5_7d_avg",
    "smart_187_7d_avg",
    "drive_age_days",
]

# ── LOAD + SPLIT DATA ─────────────────────────────────────────────────────────

def load_reference_and_current(features_path):
    """
    Reference = first 70% of data (what the model was trained on)
    Current   = last 30% of data  (simulates new incoming data)
    In production this would be today's telemetry vs training window.
    """
    print("\n[1/3] Loading reference and current datasets...")

    df = pd.read_csv(features_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.70)
    reference = df.iloc[:split_idx][MONITOR_FEATURES].copy()
    current   = df.iloc[split_idx:][MONITOR_FEATURES].copy()

    print(f"      Reference rows : {len(reference):,}")
    print(f"      Current rows   : {len(current):,}")
    print(f"      Features       : {len(MONITOR_FEATURES)}")

    return reference, current

# ── RUN EVIDENTLY REPORT ──────────────────────────────────────────────────────

def run_drift_report(reference, current):
    print("\n[2/3] Running Evidently drift analysis...")

    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetMissingValuesMetric(),
    ])

    report.run(
        reference_data = reference,
        current_data   = current,
    )

    # Save HTML report — open this in browser
    report.save_html(DRIFT_HTML_PATH)
    print(f"      HTML report saved : {DRIFT_HTML_PATH}")

    # Extract drift summary as JSON
    report_dict   = report.as_dict()
    drift_metrics = report_dict["metrics"]

    # Pull dataset-level drift
    dataset_drift = next(
        (m for m in drift_metrics
         if m["metric"] == "DatasetDriftMetric"), None
    )

    drift_share   = 0.0
    n_drifted     = 0
    n_features    = len(MONITOR_FEATURES)

    if dataset_drift:
        result      = dataset_drift.get("result", {})
        drift_share = result.get("drift_share", 0.0)
        n_drifted   = result.get("number_of_drifted_columns", 0)

    # Per-feature drift
    feature_drift = {}
    for m in drift_metrics:
        if m["metric"] == "ColumnDriftMetric":
            col    = m.get("parameters", {}).get("column_name", "unknown")
            result = m.get("result", {})
            feature_drift[col] = {
                "drift_detected": result.get("drift_detected", False),
                "drift_score"   : round(result.get("drift_score", 0.0), 4),
                "stattest"      : result.get("stattest_name", "unknown"),
            }

    return drift_share, n_drifted, n_features, feature_drift

# ── EVALUATE + SAVE SUMMARY ───────────────────────────────────────────────────

def evaluate_drift(drift_share, n_drifted, n_features, feature_drift):
    print("\n[3/3] Evaluating drift thresholds...")

    retrain_needed = drift_share > DRIFT_THRESHOLD

    summary = {
        "timestamp"      : datetime.now().isoformat(),
        "drift_share"    : round(drift_share, 4),
        "n_drifted"      : n_drifted,
        "n_features"     : n_features,
        "threshold"      : DRIFT_THRESHOLD,
        "retrain_needed" : retrain_needed,
        "feature_drift"  : feature_drift,
    }

    with open(DRIFT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n      Drift share     : {drift_share:.2%}")
    print(f"      Drifted features: {n_drifted}/{n_features}")
    print(f"      Threshold       : {DRIFT_THRESHOLD:.0%}")
    print(f"      Retrain needed  : {retrain_needed}")

    if feature_drift:
        print("\n      Per-feature drift:")
        for feat, info in feature_drift.items():
            status = "DRIFT" if info["drift_detected"] else "OK   "
            print(
                f"        [{status}] {feat:<28}"
                f"  score={info['drift_score']:.4f}"
            )

    print(f"\n      Summary saved   : {DRIFT_JSON_PATH}")
    print(f"      Open report at  : {DRIFT_HTML_PATH}")

    return retrain_needed, summary

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_drift_detection():
    print("=" * 55)
    print("  ComputeGuard — Drift Detection")
    print("=" * 55)

    reference, current = load_reference_and_current(FEATURES_PATH)

    drift_share, n_drifted, n_features, feature_drift = run_drift_report(reference, current
    )

    retrain_needed, summary = evaluate_drift(
        drift_share, n_drifted, n_features, feature_drift
    )

    print("\n" + "=" * 55)
    if retrain_needed:
        print("  DRIFT DETECTED — retraining will be triggered.")
    else:
        print("  No significant drift — model is still valid.")
    print("=" * 55)

    return retrain_needed, summary

if __name__ == "__main__":
    run_drift_detection()