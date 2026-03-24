import pytest
import pandas as pd
import os
import json

FEATURES_PATH    = "data/processed/features.csv"
QUALITY_PATH     = "data/processed/quality_report.json"

def test_features_file_exists():
    assert os.path.exists(FEATURES_PATH), \
        f"features.csv not found at {FEATURES_PATH}"

def test_features_has_rows():
    df = pd.read_csv(FEATURES_PATH, nrows=1000)
    assert len(df) > 100, "features.csv has too few rows"

def test_required_columns_exist():
    df = pd.read_csv(FEATURES_PATH, nrows=10)
    required = [
        "date", "serial_number", "model", "failure",
        "smart_5_raw", "smart_9_raw", "smart_187_raw",
        "smart_197_raw", "smart_198_raw",
        "drive_age_days", "failure_in_30d",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_no_negative_drive_age():
    df = pd.read_csv(FEATURES_PATH, usecols=["drive_age_days"])
    assert (df["drive_age_days"] >= 0).all(), \
        "Negative drive_age_days found"

def test_failure_label_is_binary():
    df = pd.read_csv(FEATURES_PATH, usecols=["failure_in_30d"])
    unique_vals = df["failure_in_30d"].unique()
    assert set(unique_vals).issubset({0, 1, 0.0, 1.0}), \
        f"failure_in_30d has non-binary values: {unique_vals}"

def test_quality_report_all_passed():
    assert os.path.exists(QUALITY_PATH), \
        f"quality_report.json not found"
    with open(QUALITY_PATH) as f:
        report = json.load(f)
    failed = report["summary"]["failed"]
    assert failed == 0, \
        f"Quality report has {failed} failed checks"