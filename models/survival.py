import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import os
import json
import pickle
import warnings

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

FEATURES_PATH       = "data/processed/features.csv"
RISK_SCORES_PATH    = "data/processed/risk_scores.csv"
NODE_HEALTH_PATH    = "data/processed/node_health.csv"
KM_REPORT_PATH      = "models/artifacts/km_report.json"
COX_REPORT_PATH     = "models/artifacts/cox_report.json"
COX_MODEL_PATH      = "models/artifacts/cox_model.pkl"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME     = "COMPUTEGAURD"

COX_FEATURES = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_5_7d_avg",
    "smart_187_7d_avg",
    "smart_5_delta",
    "smart_197_delta",
    "drive_age_days",
]

# ── 1. PREPARE SURVIVAL DATA ──────────────────────────────────────────────────

def prepare_survival_data(df):
    print("\n[1/5] Preparing survival dataset...")

    # One row per drive — duration + did it fail?
    drive_summary = df.groupby("serial_number").agg(
        first_date  = ("date", "min"),
        last_date   = ("date", "max"),
        ever_failed = ("failure", "max"),
        model       = ("model",  "first"),
    ).reset_index()

    drive_summary["first_date"] = pd.to_datetime(drive_summary["first_date"])
    drive_summary["last_date"]  = pd.to_datetime(drive_summary["last_date"])
    drive_summary["duration"]   = (
        drive_summary["last_date"] - drive_summary["first_date"]
    ).dt.days + 1

    # Remove drives with zero duration
    drive_summary = drive_summary[drive_summary["duration"] > 0].copy()

    # Merge latest SMART readings per drive for Cox model
    latest_smart = (
        df.sort_values("date")
          .groupby("serial_number")[COX_FEATURES]
          .last()
          .reset_index()
    )

    survival_df = drive_summary.merge(
        latest_smart, on="serial_number", how="left"
    )
    survival_df[COX_FEATURES] = survival_df[COX_FEATURES].fillna(0)

    total    = len(survival_df)
    failures = int(survival_df["ever_failed"].sum())
    censored = total - failures

    print(f"      Total drives   : {total:,}")
    print(f"      Failures       : {failures:,} ({failures/total*100:.2f}%)")
    print(f"      Censored       : {censored:,} — still alive, not failed")
    print(f"      Median duration: {survival_df['duration'].median():.0f} days")

    return survival_df

# ── 2. KAPLAN-MEIER ───────────────────────────────────────────────────────────

def fit_kaplan_meier(survival_df):
    print("\n[2/5] Fitting Kaplan-Meier curves...")

    kmf      = KaplanMeierFitter()
    km_stats = {}

    # Overall curve
    kmf.fit(
        durations      = survival_df["duration"],
        event_observed = survival_df["ever_failed"],
    )
    overall_median = kmf.median_survival_time_
    print(f"      Overall median survival : {overall_median:.0f} days")

    # Per top-5 hardware models
    top_models = (
        survival_df["model"]
        .value_counts()
        .head(5)
        .index.tolist()
    )

    print(f"      Survival curves for top {len(top_models)} drive models:")

    for hw_model in top_models:
        subset = survival_df[survival_df["model"] == hw_model]
        if len(subset) < 10:
            continue

        kmf.fit(
            durations      = subset["duration"],
            event_observed = subset["ever_failed"],
            label          = hw_model,
        )

        median_days  = kmf.median_survival_time_
        survival_30  = float(kmf.survival_function_at_times([30]).values[0])
        survival_60  = float(kmf.survival_function_at_times([60]).values[0])
        survival_90  = float(kmf.survival_function_at_times([90]).values[0])

        km_stats[hw_model] = {
            "count"       : int(len(subset)),
            "failures"    : int(subset["ever_failed"].sum()),
            "median_days" : float(median_days) if not np.isinf(median_days) else 9999,
            "survival_30d": round(survival_30, 4),
            "survival_60d": round(survival_60, 4),
            "survival_90d": round(survival_90, 4),
        }

        print(
            f"        {hw_model[:35]:<35}"
            f"  30d={survival_30:.3f}"
            f"  60d={survival_60:.3f}"
            f"  90d={survival_90:.3f}"
        )

    os.makedirs("models/artifacts", exist_ok=True)
    with open(KM_REPORT_PATH, "w") as f:
        json.dump(km_stats, f, indent=2)
    print(f"      Saved: {KM_REPORT_PATH}")

    return kmf, km_stats

# ── 3. COX PROPORTIONAL HAZARDS ───────────────────────────────────────────────

def fit_cox_model(survival_df):
    print("\n[3/5] Fitting Cox Proportional Hazards model...")

    cox_df = survival_df[
        ["duration", "ever_failed"] + COX_FEATURES
    ].copy().dropna()

    # Clip outliers — Cox PH sensitive to extreme values
    for col in COX_FEATURES:
        upper = cox_df[col].quantile(0.99)
        if upper > 0:
            cox_df[col] = cox_df[col].clip(upper=upper)

    # Normalize 0-1
    for col in COX_FEATURES:
        col_max = cox_df[col].max()
        if col_max > 0:
            cox_df[col] = cox_df[col] / col_max

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        cox_df,
        duration_col  = "duration",
        event_col     = "ever_failed",
        show_progress = False,
    )

    # C-index — survival model equivalent of ROC AUC
    c_index = concordance_index(
        cox_df["duration"],
        -cph.predict_partial_hazard(cox_df),
        cox_df["ever_failed"],
    )
    print(f"      C-index : {c_index:.4f}  (0.5=random, 1.0=perfect)")

    # Hazard ratios — what drives failure risk
    print("\n      Feature hazard ratios (>1 = increases failure risk):")
    coef_values = cph.params_.values
    for feat, coef in sorted(
        zip(COX_FEATURES, coef_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        hr        = np.exp(coef)
        direction = "increases risk" if coef > 0 else "reduces risk"
        print(f"        {feat:<28}  HR={hr:.3f}  ({direction})")

    # Save
    with open(COX_MODEL_PATH, "wb") as f:
        pickle.dump(cph, f)

    cox_report = {
        "c_index"      : round(c_index, 4),
        "n_subjects"   : len(cox_df),
        "n_events"     : int(cox_df["ever_failed"].sum()),
        "feature_coefs": {
            feat: round(float(coef), 4)
            for feat, coef in zip(COX_FEATURES, coef_values)
        },
    }
    with open(COX_REPORT_PATH, "w") as f:
        json.dump(cox_report, f, indent=2)

    print(f"\n      Model saved : {COX_MODEL_PATH}")
    print(f"      Report saved: {COX_REPORT_PATH}")

    return cph, cox_report, c_index

# ── 4. BUILD NODE HEALTH TABLE ────────────────────────────────────────────────

def build_node_health(survival_df, cph, risk_scores_df):
    print("\n[4/5] Building node_health.csv...")

    # Prepare Cox input — same clipping + normalization as training
    cox_input = survival_df[COX_FEATURES].copy()
    for col in COX_FEATURES:
        upper = cox_input[col].quantile(0.99)
        if upper > 0:
            cox_input[col] = cox_input[col].clip(upper=upper)
        col_max = cox_input[col].max()
        if col_max > 0:
            cox_input[col] = cox_input[col] / col_max

    # Partial hazard — higher = more dangerous
    survival_df = survival_df.copy()
    survival_df["partial_hazard"] = cph.predict_partial_hazard(
        cox_input
    ).values

    # Estimated days remaining
    median_dur = survival_df["duration"].median()
    survival_df["estimated_days_remaining"] = (
        median_dur / survival_df["partial_hazard"].clip(lower=0.01)
    ).clip(upper=365).round(1)

    # Merge latest XGBoost risk score from Day 2
    risk_latest = (
        risk_scores_df
        .sort_values("date")
        .groupby("serial_number")[["risk_score", "risk_tier"]]
        .last()
        .reset_index()
    )

    node_health = survival_df.merge(
        risk_latest, on="serial_number", how="left"
    )

    node_health["risk_score"] = node_health["risk_score"].fillna(
        node_health["risk_score"].median()
    )
    node_health["risk_tier"] = node_health["risk_tier"].fillna("low")

    # Health status — combines both models
    def get_health_status(row):
        if row["risk_score"] >= 0.85 or row["estimated_days_remaining"] <= 7:
            return "critical"
        elif row["risk_score"] >= 0.60 or row["estimated_days_remaining"] <= 30:
            return "warning"
        else:
            return "healthy"

    node_health["health_status"] = node_health.apply(
        get_health_status, axis=1
    )

    # Final columns
    node_health = node_health[[
        "serial_number",
        "model",
        "duration",
        "ever_failed",
        "partial_hazard",
        "estimated_days_remaining",
        "risk_score",
        "risk_tier",
        "health_status",
    ]].sort_values("risk_score", ascending=False).reset_index(drop=True)

    node_health.to_csv(NODE_HEALTH_PATH, index=False)

    status = node_health["health_status"].value_counts()
    print(f"      Total nodes : {len(node_health):,}")
    print(f"      Critical    : {status.get('critical', 0):,}")
    print(f"      Warning     : {status.get('warning',  0):,}")
    print(f"      Healthy     : {status.get('healthy',  0):,}")
    print(f"      Saved       : {NODE_HEALTH_PATH}")

    return node_health

# ── 5. LOG TO MLFLOW ──────────────────────────────────────────────────────────

def log_to_mlflow(cox_report, km_stats):
    print("\n[5/5] Logging to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="survival-analysis-cox-ph") as run:

        mlflow.log_metrics({
            "c_index"   : cox_report["c_index"],
            "n_subjects": float(cox_report["n_subjects"]),
            "n_events"  : float(cox_report["n_events"]),
        })

        mlflow.log_params({
            "model_type"  : "CoxPH",
            "penalizer"   : 0.1,
            "n_features"  : len(COX_FEATURES),
        })

        mlflow.log_artifact(COX_MODEL_PATH)
        mlflow.log_artifact(COX_REPORT_PATH)
        mlflow.log_artifact(KM_REPORT_PATH)
        mlflow.log_artifact(NODE_HEALTH_PATH)

        # Register model
        mlflow.register_model(
            model_uri = f"runs:/{run.info.run_id}/survival-model",
            name      = "computeguard-survival-model",
        )

        run_id = run.info.run_id
        print(f"      Run ID  : {run_id}")
        print(f"      C-index : {cox_report['c_index']:.4f}")
        print(f"      View at : http://127.0.0.1:5000")

    return run_id

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_survival_analysis():
    print("=" * 55)
    print("  ComputeGuard — Survival Analysis")
    print("=" * 55)

    print("\n[0/5] Loading data...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    print(f"      features.csv     : {df.shape}")

    risk_df = pd.read_csv(RISK_SCORES_PATH)
    print(f"      risk_scores.csv  : {risk_df.shape}")

    survival_df        = prepare_survival_data(df)
    kmf, km_stats      = fit_kaplan_meier(survival_df)
    cph, cox_report, _ = fit_cox_model(survival_df)
    node_health        = build_node_health(survival_df, cph, risk_df)
    run_id             = log_to_mlflow(cox_report, km_stats)

    print("\n" + "=" * 55)
    print("  Survival analysis complete.")
    print(f"  node_health.csv : {NODE_HEALTH_PATH}")
    print(f"  MLflow run ID   : {run_id}")
    print("=" * 55)

    return node_health, cph, run_id

if __name__ == "__main__":
    run_survival_analysis()