import io, os, json, math, boto3, pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from urllib.parse import urlparse
from datetime import datetime
import mlflow

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix
)

from runner.utils.mlflow_utils import setup_mlflow
from runner.utils import infer as infer_utils  # uses Step-8 helper

# ----------------------------
# S3 IO helpers
# ----------------------------
def _read_s3_bytes(uri: str) -> bytes:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read()

def _read_df(uri: str) -> pd.DataFrame:
    uri_l = uri.lower()
    raw = _read_s3_bytes(uri)
    if uri_l.endswith(".parquet") or uri_l.endswith(".pq"):
        return pd.read_parquet(io.BytesIO(raw))
    if uri_l.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    # try parquet then csv
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        return pd.read_csv(io.BytesIO(raw))

def _put_s3_bytes(s3, bucket, key, data: bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"

def _write_parquet_s3(df: pd.DataFrame, s3_uri: str):
    bucket = s3_uri.split("s3://",1)[1].split("/",1)[0]
    key = s3_uri.split(bucket+"/",1)[1]
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream")

# ----------------------------
# Model registry utilities
# ----------------------------
def _resolve_model_version(model_name: str | None) -> tuple[str, str]:
    """
    If version not given, pick the highest numeric version of the model in registry.
    Returns (model_name, version_str). Raises if none found.
    """
    client = mlflow.tracking.MlflowClient()
    if not model_name:
        raise ValueError("model_name must be provided or resolvable from cfg['registry']['name']")
    vers = client.search_model_versions(f"name = '{model_name}'")
    if not vers:
        raise RuntimeError(f"No versions found in MLflow registry for model '{model_name}'")
    best = max(vers, key=lambda v: int(v.version))
    return best.name, str(best.version)

# ----------------------------
# Main driver
# ----------------------------
def run_predict(cfg, input_uri: str, model_name: str | None = None, model_version: str | None = None, output_prefix: str | None = None):
    """
    Batch scoring:
      - Loads model from registry (latest by default)
      - Reads CSV/Parquet from S3
      - Produces {id, score, prediction, [target if present]} and writes Parquet to S3
      - Logs a small prediction report to MLflow
    """
    setup_mlflow(cfg["experiment_name"])
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(cfg["experiment_name"])

    # resolve model
    model_name = model_name or cfg["registry"]["name"]
    if model_version is None:
        model_name, model_version = _resolve_model_version(model_name)

    # load pipeline + decision threshold
    pipe, threshold = infer_utils.load_model(model_name, model_version)

    # read data
    df = _read_df(input_uri)
    target_col = cfg.get("target")
    id_col = cfg.get("id_column")

    y_true = None
    if target_col and target_col in df.columns:
        y_true = df[target_col].copy()

    # build feature frame (do not drop id; pipeline will ignore unknowns)
    X = df.copy()

    # predict proba
    proba = pipe.predict_proba(X)
    if proba.ndim == 1:
        # some models may output (n,) with positive probs
        pos = proba.astype(float)
        pred = (pos >= float(threshold)).astype(int)
        score = pos
    else:
        if proba.shape[1] == 2:
            pos = proba[:, 1].astype(float)
            pred = (pos >= float(threshold)).astype(int)
            score = pos
        else:
            # multiclass: choose argmax; score is max class prob
            argmax = np.argmax(proba, axis=1)
            pred = argmax
            score = np.max(proba, axis=1)

    # prepare output
    out = pd.DataFrame({
        "prediction": pred,
        "score": score,
    })
    out["threshold"] = float(threshold)

    # include id if present
    if id_col and id_col in df.columns:
        out.insert(0, id_col, df[id_col].values)
    else:
        out.insert(0, "row_id", np.arange(len(out)))

    # include target if present
    if y_true is not None:
        out[target_col] = y_true.values

    # choose output location
    s3_bucket = cfg["s3"]["bucket"].split("s3://",1)[1]
    base_prefix = (output_prefix or f'{cfg["s3"]["prefix"].strip("/")}/predictions/{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}')
    pred_uri = f"s3://{s3_bucket}/{base_prefix}/predictions.parquet"
    meta_uri = f"s3://{s3_bucket}/{base_prefix}/summary.json"

    # write predictions
    _write_parquet_s3(out, pred_uri)

    # compute optional quick metrics if y_true available (binary only)
    quick = {}
    cm_dict = None
    pr_png_uri = None
    cm_png_uri = None

    with mlflow.start_run(run_name=f"predict_{model_name}_v{model_version}") as run:
        mlflow.log_params({
            "input_uri": input_uri,
            "model_name": model_name,
            "model_version": model_version,
            "used_threshold": float(threshold),
        })

        if y_true is not None and (proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 2)):
            y = (pd.Series(y_true).astype(int)).values
            p = score  # positive class probability

            try:
                quick["auprc"] = float(average_precision_score(y, p))
            except Exception:
                quick["auprc"] = float("nan")
            try:
                quick["auroc"] = float(roc_auc_score(y, p))
            except Exception:
                quick["auroc"] = float("nan")

            yhat = (p >= float(threshold)).astype(int)
            quick["precision"] = float(precision_score(y, yhat, zero_division=0))
            quick["recall"]    = float(recall_score(y, yhat, zero_division=0))
            quick["f1"]        = float(f1_score(y, yhat, zero_division=0))

            # confusion matrix png
            cm = confusion_matrix(y, yhat, labels=[0,1])
            cm_dict = {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}

            # log PR curve & CM
            import matplotlib.pyplot as plt
            pr, rc, _ = precision_recall_curve(y, p)
            plt.figure(); plt.plot(rc, pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Prediction PR Curve")
            mlflow.log_figure(plt.gcf(), "predict_prc.png"); plt.close()

            plt.figure()
            plt.imshow(cm, interpolation='nearest')
            plt.title(f"Confusion Matrix @ thr={float(threshold):.3f}")
            plt.xlabel("Predicted"); plt.ylabel("Actual")
            plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
            for (i,j), val in np.ndenumerate(cm):
                plt.text(j, i, int(val), ha='center', va='center')
            mlflow.log_figure(plt.gcf(), "predict_confusion.png"); plt.close()

            # pull logged figures back and mirror to S3 near predictions
            client = mlflow.tracking.MlflowClient()
            s3 = boto3.client("s3")
            def _dl(artifact_path):
                try:
                    local = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=artifact_path)
                    with open(local, "rb") as f: return f.read()
                except Exception:
                    return None
            pr_png = _dl("predict_prc.png")
            cm_png = _dl("predict_confusion.png")
            if pr_png:
                pr_png_uri = _put_s3_bytes(s3, s3_bucket, f"{base_prefix}/predict_prc.png", pr_png, "image/png")
            if cm_png:
                cm_png_uri = _put_s3_bytes(s3, s3_bucket, f"{base_prefix}/predict_confusion.png", cm_png, "image/png")

            mlflow.log_metrics(quick)

        # write summary JSON alongside predictions
        summary = {
            "input_uri": input_uri,
            "predictions_uri": pred_uri,
            "model": {"name": model_name, "version": model_version},
            "used_threshold": float(threshold),
            "metrics": quick or None,
            "confusion": cm_dict,
            "artifacts": {"pr_curve_png": pr_png_uri, "confusion_png": cm_png_uri}
        }
        s3 = boto3.client("s3")
        bucket = s3_bucket
        key = f"{base_prefix}/summary.json"
        _put_s3_bytes(s3, bucket, key, json.dumps(summary, indent=2).encode("utf-8"), content_type="application/json")

    return {
        "predictions_s3": pred_uri,
        "summary_s3": meta_uri,
        "model_used": {"name": model_name, "version": model_version},
        "threshold": float(threshold),
    }
