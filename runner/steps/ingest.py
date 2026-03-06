import os
from runner.io import snowflake, redshift, athena, s3
from runner.utils.mlflow_utils import setup_mlflow
import mlflow, json

def run_ingest(cfg):
    setup_mlflow(cfg["experiment_name"])
    with mlflow.start_run(run_name="ingest"):
        df = (snowflake if os.getenv("WAREHOUSE")=="snowflake" else athena).fetch_df()
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_cols", len(df.columns))
        path = s3.write_parquet_df(df, cfg["s3"]["prefix"])
        mlflow.log_artifact_local = False
        mlflow.log_dict({"dataset_uri": path, "columns": list(df.columns)}, "ingest_manifest.json")
        return {"dataset_uri": path, "columns": list(df.columns)}
