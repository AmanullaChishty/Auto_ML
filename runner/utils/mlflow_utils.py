import mlflow, os
def setup_mlflow(experiment_name: str):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)
    if os.getenv("MLFLOW_S3_BUCKET","").startswith("s3://"):
        pass  # MLflow server already configured to S3
