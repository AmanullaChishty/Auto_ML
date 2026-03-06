import json, mlflow
from typing import Optional, Iterable
import numpy as np
import pandas as pd

def _get_model_threshold_from_registry(model_name: str, model_version: Optional[str] = None, default: float = 0.5) -> float:
    """
    Reads decision_threshold from model version tags (preferred) or from the originating run's
    artifact `serving/serving_threshold.json`. Falls back to `default` if not found.
    """
    client = mlflow.tracking.MlflowClient()
    if model_version is None:
        # Use latest version in any stage
        vers = client.search_model_versions(f"name = '{model_name}'")
        if not vers:
            return default
        # pick the most recent numerically
        model_version = max(vers, key=lambda v: int(v.version)).version

    mv = client.get_model_version(model_name, str(model_version))
    # Prefer model-version tag
    thr = (mv.tags or {}).get("decision_threshold")
    if thr is not None:
        try:
            return float(thr)
        except Exception:
            pass

    # Fall back to the originating run’s artifact
    run_id = mv.run_id
    try:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="serving/serving_threshold.json")
        with open(local, "r") as f:
            payload = json.load(f)
        t = payload.get("decision_threshold", default)
        return float(t)
    except Exception:
        return default

def load_model(model_name: str, model_version: Optional[str] = None):
    """
    Loads an MLflow model as a sklearn pipeline (we logged via mlflow.sklearn).
    Returns (pipeline, decision_threshold).
    """
    uri = f"models:/{model_name}/{model_version}" if model_version else f"models:/{model_name}/latest"
    pipe = mlflow.sklearn.load_model(uri)
    thr  = _get_model_threshold_from_registry(model_name, model_version)
    return pipe, thr

def predict_proba(pipe, df: pd.DataFrame) -> np.ndarray:
    """Returns positive-class probabilities for binary or class-wise probs for multiclass."""
    proba = pipe.predict_proba(df)
    return proba

def predict_labels(pipe, df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Binary labels using the provided threshold (if multiclass, chooses argmax)."""
    proba = predict_proba(pipe, df)
    if proba.ndim == 2 and proba.shape[1] > 2:
        return np.argmax(proba, axis=1)
    pos = proba[:, 1] if proba.ndim == 2 else proba
    return (pos >= float(threshold)).astype(int)
