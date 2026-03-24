import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (auc, mean_squared_error, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report, accuracy_score)
import xgboost as xgb
import os
import json

Feature_Path = "data/processed/features.csv"
Model_Output_DIR = "models/artifacts"
Risk_Score_Path = "data/processed/risk_scores.csv"
MLFOW_TRACKING_URI = "http://127.0.0.1:5000"
Experiment_Name = "COMPUTEGAURD"

FEATURE_COLS = [
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_5_7d_avg",
    "smart_187_7d_avg",
    "smart_197_7d_avg",
    "smart_198_7d_avg",
    "smart_5_30d_avg",
    "smart_187_30d_avg",
    "smart_5_delta",
    "smart_197_delta",
    "smart_198_delta",
    "drive_age_days",
]

LABEL_COL = "failure_in_30d"

PARAMS = {
    "n_estimators": 500,#300
    "max_depth": 8,#6
    "learning_rate": 0.01, #0.05
    "subsample": 0.7,#0.8
    "colsample_bytree": 0.7,#0.8
    "random_state": 42,
    "tree_method": "hist",
    "device" : "cpu"
}

#------ 1. DATA LOADING FUNCTION--------------------
def load_data():
    print("[1/5] Loading Features....")
    df = pd.read_csv(Feature_Path)

    print(f"     Shape         : {df.shape}")
    print(f"     Failure Rate  : {df[LABEL_COL].mean()*100:.4f}%")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the dataset: {missing}")
    
    return df

#-------- 2. PREPARING DATA ----------------

def prepare_data(df):
    print("[2/5] Preparing Data...")

    # Convert date to datetime for efficient sorting
    df["date"] = pd.to_datetime(df["date"])
    
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    
    # Sample to last 500K rows to avoid memory issues (optional)
    if len(df) > 500000:
        print(f"     Sampling last 500K rows from {len(df):,} total rows")
        df = df.tail(500000).reset_index(drop=True)

    X = df[FEATURE_COLS].astype(np.float32)
    y = df[LABEL_COL].astype(np.int8)

    df_sorted = df.sort_values(by="date")
    split_idx = int(len(df_sorted) * 0.8)
    df_train_idx = df_sorted.index[:split_idx]
    df_test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[df_train_idx], X.loc[df_test_idx]
    y_train, y_test = y.loc[df_train_idx], y.loc[df_test_idx]

    print(f"     Training Samples : {len(X_train)}")
    print(f"     Testing Samples  : {len(X_test)}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    print(f"     Positives     : {pos:,} ({pos/len(y_train)*100:.4f}%)")
    print(f"     Scale Pos Weight : {scale_pos_weight:.4f}")

    return X_train, X_test, y_train, y_test, scale_pos_weight, df.loc[df_test_idx]


#-------- 3. TRAINING MODEL ----------------

def train_model(X_train, y_train, scale_pos_weight):
    print("[3/5] Training Model...")

    model = xgb.XGBClassifier(**PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=50,
    )

    print("     Model training completed.")
    return model


#------- 4. EVALUATING MODEL ----------------

def evaluate_model(model, X_test, y_test):
    print("[4/5] Evaluating Model...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    print(f"     Precision : {metrics['precision']:.4f}")
    print(f"     Recall    : {metrics['recall']:.4f}")
    print(f"     F1-Score  : {metrics['f1']:.4f}")
    print(f"     AUC-ROC   : {metrics['auc']:.4f}")

    print("     Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("     Confusion Matrix:")
    print(cm)

    return metrics, y_proba

#------- 5. LOGGING WITH MLFLOW ----------------

def log_with_mlflow(model, metrics, X_train):
    print("[5/5] Logging with MLflow...")

    mlflow.set_tracking_uri(MLFOW_TRACKING_URI)
    mlflow.set_experiment(Experiment_Name)

    with mlflow.start_run(run_name="xgboost-failure-prediction") as run:

        #Log Parameters
        mlflow.log_params(PARAMS)

        #Log Metrics
        mlflow.log_metrics(metrics)

        #Log Feature Importance
        importance = model.get_booster().get_score(importance_type="gain")
        importamce_path = "models/artifacts/feature_importance.json"
        with open(importamce_path, "w") as f:
            json.dump(importance, f, indent=2)
        
        mlflow.log_artifact(importamce_path)

        #Log Model to registry
        mlflow.xgboost.log_model(
            model, 
            artifact_path = "xgb_model",
            registered_model_name = "ComputeGuard_XGB_Failure_Predictor"
        )
       
        run_id = run.info.run_id

        print(f"     Run ID : {run_id}")
        print(f"     Experiment Name : {Experiment_Name}")
        print(f"     View at : http://localhost:5000")

    return run_id


#------ 6. SAVE RISK SCORES ----------------

def save_risk_scores(df_test, y_proba, run_id):
    print("Saving Risk Scores...")

    risk_df = df_test[["date", "serial_number", "model", "failure_in_30d"]].copy()
    risk_df["risk_score"] = y_proba

    # Risk tier — used by dashboard on Day 7
    risk_df["risk_tier"] = pd.cut(
        risk_df["risk_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"]
    )

    risk_df = risk_df.sort_values("risk_score", ascending=False)
    risk_df.to_csv(Risk_Score_Path, index=False)

    print(f"      Saved      : {Risk_Score_Path}")
    print(f"      High risk  : {(risk_df['risk_tier']=='high').sum():,} drives")
    print(f"      Medium risk: {(risk_df['risk_tier']=='medium').sum():,} drives")
    print(f"      Low risk   : {(risk_df['risk_tier']=='low').sum():,} drives")


#------ MAIN FUNCTION ----------------

def run_training():
    print(" ComputeGuard -- XGBoost Failure Prediction")

    os.makedirs(Model_Output_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, scale_pos_weight, df_test = prepare_data(df)

    model = train_model(X_train, y_train, scale_pos_weight)

    metrics, y_proba = evaluate_model(model, X_test, y_test)
    run_id = log_with_mlflow(model, metrics, X_train)

    save_risk_scores(df_test, y_proba, run_id)

    print("Training pipeline completed successfully....")

    print(f"MLflow run ID: {run_id}")

    print(f" Risk scores saved to: {Risk_Score_Path}")

    return model, metrics, run_id


if __name__ == "__main__":

    run_training()