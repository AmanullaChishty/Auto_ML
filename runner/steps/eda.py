import os, io, json, re, boto3, pandas as pd, pyarrow.parquet as pq
from ydata_profiling import ProfileReport
from urllib.parse import urlparse
from runner.utils.mlflow_utils import setup_mlflow
from runner.utils.schema import make_data_card 
import mlflow
from datetime import datetime

def _read_parquet_s3(uri):
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _put_s3_bytes(bucket, key, data: bytes, content_type: str = "application/octet-stream"):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"

def run_eda(cfg, dataset_uri):
    setup_mlflow(cfg["experiment_name"])
    with mlflow.start_run(run_name="eda"):
        df = _read_parquet_s3(dataset_uri)

        # Handle timestamp parsing (keep your legacy behavior too)
        ts_col = cfg.get("timestamp_column") or ("INDEX_MONTH" if "INDEX_MONTH" in df.columns else None)
        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        target = cfg.get("target")
        prof = ProfileReport(df, title="EDA Profile", minimal=True, infer_dtypes=True)
        html = prof.to_html()

        # S3 base
        bucket = cfg["s3"]["bucket"].split("s3://", 1)[1]
        key_base = f'{cfg["s3"]["prefix"]}/eda'

        # 1) HTML profile
        profile_key = f"{key_base}/profile.html"
        profile_s3 = _put_s3_bytes(bucket, profile_key, html.encode("utf-8"), content_type="text/html")

        # 2) Data Card (machine-readable)
        data_card = make_data_card(
            df=df,
            dataset_uri=dataset_uri,
            target=target,
            task_type=cfg.get("task_type"),
            id_column=cfg.get("id_column"),
            eda_cfg=cfg.get("eda", {})  # new knobs (defaults handled inside)
        )
        mlflow.log_dict(data_card, "eda/data_card.json")  # MLflow artifact
        data_card_s3 = _put_s3_bytes(
            bucket,
            f"{key_base}/data_card.json",
            json.dumps(data_card, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"),
            content_type="application/json"
        )

        # 3) Preview CSV (head)
        sample_rows = int(cfg.get("eda", {}).get("sample_rows", 100))
        preview_csv = df.head(sample_rows).to_csv(index=False)
        mlflow.log_text(preview_csv, "eda/preview.csv")
        preview_s3 = _put_s3_bytes(
            bucket,
            f"{key_base}/preview.csv",
            preview_csv.encode("utf-8"),
            content_type="text/csv"
        )

        # 4) Quick label stats (kept from your version)
        eda_summary = {"rows": int(len(df)), "cols": int(len(df.columns))}
        if target in df.columns:
            vc = df[target].value_counts(dropna=False).to_dict()
            eda_summary["target_counts"] = {str(k): int(v) for k, v in vc.items()}
        mlflow.log_dict(eda_summary, "eda_summary.json")
        mlflow.log_text(html, "profile_preview.html")

        return {
            "eda_s3": profile_s3,
            "eda_summary": eda_summary,
            "data_card_s3": data_card_s3,
            "preview_s3": preview_s3
        }
