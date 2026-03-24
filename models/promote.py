import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "ComputeGuard_XGB_Failure_Predictor"

def promote_model_to_production():
    print("[Promoting Model to Production]")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get the latest version of the model in the "Staging" stage
    versions = client.get_latest_versions(name=MODEL_NAME)

    print(f"\nAll versions of '{MODEL_NAME}':")
    best_version = None
    best_auc     = 0

    for v in versions:
        run = client.get_run(v.run_id)
        auc = run.data.metrics.get("roc_auc", 0)
        print(f"  Version {v.version} | ROC AUC: {auc:.4f} | Stage: {v.current_stage}")

        if auc > best_auc:
            best_auc     = auc
            best_version = v.version

    print(f"\nBest version: {best_version} (ROC AUC: {best_auc:.4f})")


    if best_version is None:
    # Fallback: promote the latest version
        latest_version = max(int(v.version) for v in versions)
        best_version = str(latest_version)
        print(f"No good version found; promoting latest version {best_version}")

    # Transition the model to "Production" stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model version {best_version} promoted to Production stage.")



if __name__ == "__main__":
    promote_model_to_production()
