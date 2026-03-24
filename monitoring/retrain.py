import mlflow
import json
import os
import sys
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME     = "computeguard"
MODEL_NAME          = "computeguard-failure-predictor"
DRIFT_JSON_PATH     = "monitoring/reports/drift_summary.json"

def get_production_metrics(client):
    """Get current production model metrics from MLflow."""
    try:
        prod_versions = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )
        if not prod_versions:
            print("      No production model found — will promote new model.")
            return None

        prod_run_id = prod_versions[0].run_id
        prod_run    = client.get_run(prod_run_id)
        metrics     = prod_run.data.metrics

        print(f"      Production model run  : {prod_run_id[:8]}...")
        print(f"      Production ROC AUC    : {metrics.get('roc_auc', 0):.4f}")
        print(f"      Production F1         : {metrics.get('f1_score', 0):.4f}")

        return metrics

    except Exception as e:
        print(f"      Could not load production metrics: {e}")
        return None

def should_retrain():
    """Check drift summary to decide if retraining is needed."""
    if not os.path.exists(DRIFT_JSON_PATH):
        print("No drift summary found — running retrain anyway.")
        return True

    with open(DRIFT_JSON_PATH) as f:
        summary = json.load(f)

    retrain = summary.get("retrain_needed", False)
    print(f"      Drift share    : {summary.get('drift_share', 0):.2%}")
    print(f"      Retrain needed : {retrain}")
    return retrain

def promote_if_better(client, new_run_id, prod_metrics):
    """Compare new model vs production — promote only if better."""
    new_run     = client.get_run(new_run_id)
    new_metrics = new_run.data.metrics

    new_auc  = new_metrics.get("roc_auc",  0)
    new_f1   = new_metrics.get("f1_score", 0)

    print(f"\n      New model ROC AUC : {new_auc:.4f}")
    print(f"      New model F1      : {new_f1:.4f}")

    if prod_metrics is None:
        print("      No existing production model — promoting new model.")
        should_promote = True
    else:
        prod_auc = prod_metrics.get("roc_auc",  0)
        prod_f1  = prod_metrics.get("f1_score", 0)
        # Promote only if new model beats current on both metrics
        should_promote = (new_auc >= prod_auc) and (new_f1 >= prod_f1)
        print(
            f"      Decision: {'PROMOTE' if should_promote else 'KEEP CURRENT'} "
            f"(new AUC={new_auc:.4f} vs prod AUC={prod_auc:.4f})"
        )

    if should_promote:
        # Get latest model version
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest   = max(versions, key=lambda v: int(v.version))

        client.transition_model_version_stage(
            name                  = MODEL_NAME,
            version               = latest.version,
            stage                 = "Production",
            archive_existing_versions = True,
        )
        print(f"      Version {latest.version} promoted to Production.")
    else:
        print("      New model did not improve — keeping current production model.")

    return should_promote

def run_retraining():
    print("=" * 55)
    print("  ComputeGuard — Automated Retraining")
    print("=" * 55)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("\n[1/3] Checking drift summary...")
    retrain = should_retrain()

    if not retrain:
        print("\n  No drift detected — skipping retraining.")
        print("=" * 55)
        return False

    print("\n[2/3] Getting current production metrics...")
    prod_metrics = get_production_metrics(client)

    print("\n[3/3] Running retraining pipeline...")

    # Import and run training
    sys.path.insert(0, ".")
    from models.train import run_training
    _, new_metrics, new_run_id = run_training()

    print("\n[4/3] Comparing and promoting...")
    promoted = promote_if_better(client, new_run_id, prod_metrics)

    print("\n" + "=" * 55)
    print(f"  Retraining complete. Promoted: {promoted}")
    print("=" * 55)

    return promoted

if __name__ == "__main__":
    run_retraining()