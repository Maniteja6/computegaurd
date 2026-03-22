import mlflow
import os
MLFlow_Tracking_URI = "sqlite:///mlruns.db"
Experiment_Name = "COMPUTEGAURD"
def init_mlflow():
    mlflow.set_tracking_uri(MLFlow_Tracking_URI)
    Experiment = mlflow.get_experiment_by_name(Experiment_Name)
    if Experiment is None:
        mlflow.create_experiment(Experiment_Name,
                                 tags = {"project": "COMPUTEGAURD",
                                         "version": "1.0",
                                         "description": "GPU Failure Prediction using ML models and Capacity Forecasting",
                                         "owner": "Maniteja Julakanti"})
        print(f"Experiment '{Experiment_Name}' created successfully.")
    else:
        print(f"Experiment '{Experiment_Name}' already exists.")

    
    mlflow.set_experiment(Experiment_Name) 
    print(f"Tracking URI: {MLFlow_Tracking_URI}")
    print("Mlflow initialized successfully.")
if __name__ == "__main__":
    init_mlflow()