import os, io, json, tempfile
from urllib.parse import urlparse

import boto3
import numpy as np
import mlflow
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

# ----------------------------
# Plan loading helpers
# ----------------------------
def _read_s3_text(uri: str) -> str:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _load_plan_from_state():
    try:
        state = json.load(open(".state.json"))
    except FileNotFoundError:
        return None
    s3_uri = state.get("ai_plan_s3")
    if not s3_uri:
        return None
    try:
        return yaml.load(_read_s3_text(s3_uri)) or {}
    except Exception:
        return None

def _primary_metric(cfg, plan):
    pm = (plan or {}).get("evaluation", {}).get("primary_metric")
    if pm: return pm.lower()
    pm = cfg.get("metrics", {}).get("primary")
    return (pm or "auprc").lower()

def _min_recall(cfg, plan, override):
    # Prefer plan constraint if present, else CLI arg, else cfg, else 0.6
    if (plan or {}).get("evaluation", {}) \
                  .get("threshold_tuning", {}) \
                  .get("constraint", {}) \
                  .get("min_recall") is not None:
        return float(plan["evaluation"]["threshold_tuning"]["constraint"]["min_recall"])
    if override is not None:
        return float(override)
    if cfg.get("metrics", {}).get("min_recall") is not None:
        return float(cfg["metrics"]["min_recall"])
    return 0.6

def _metric_or_nan(row, key):
    v = row.get(key)
    return v if (isinstance(v, (int,float)) and np.isfinite(v)) else float("nan")

# ----------------------------
# Selection
# ----------------------------
def select_model(cfg, leaderboard, min_recall=0.6):
    """
    Uses AI plan's primary metric (default AUPRC) and recall constraint.
    Falls back safely if plan missing.
    """
    plan = _load_plan_from_state() or {}
    pm = _primary_metric(cfg, plan)             # e.g., "auprc" / "auroc" / "f1"
    mr = _min_recall(cfg, plan, min_recall)     # recall floor

    # filter by recall >= mr (if none, keep all)
    filtered = [m for m in leaderboard if _metric_or_nan(m, "recall") >= mr]
    pool = filtered if filtered else leaderboard

    # sort by primary metric desc, then F1 as tie-breaker
    def sort_key(m):
        pm_val = _metric_or_nan(m, pm)
        f1_val = _metric_or_nan(m, "f1")
        # NaNs should sink
        pm_sort = -1e9 if not np.isfinite(pm_val) else pm_val
        f1_sort = -1e9 if not np.isfinite(f1_val) else f1_val
        return (pm_sort, f1_sort)

    pool = sorted(pool, key=sort_key, reverse=True)
    chosen = dict(pool[0])
    # annotate selection policy for traceability
    chosen["selection"] = {
        "primary_metric": pm,
        "min_recall": mr,
        "filtered_count": len(filtered),
        "total_count": len(leaderboard),
        "used_recall_filter": bool(filtered)
    }
    return chosen

# ----------------------------
# Registration (+ persist decision threshold)
# ----------------------------
def register_selected(cfg, selected):
    """
    Finds the matching 'final_{name}' run, tags it with the serving threshold & policy,
    registers the model, and mirrors the same tags to the model version.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()

    exp = mlflow.get_experiment_by_name(cfg["experiment_name"])
    run_name = f"final_{selected['name']}"
    # Narrow search to the expected runName
    runs = client.search_runs(
        [exp.experiment_id],
        f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=50
    )

    if not runs:
        raise RuntimeError(f"No MLflow runs found with name {run_name}")

    # pick the run whose test metrics are closest to 'selected'
    def dist(r):
        keys = ["auprc","auroc","precision","recall","f1"]
        s = 0.0
        for k in keys:
            sel = selected.get(k)
            rm  = r.data.metrics.get(f"test_{k}", None)
            if sel is None or rm is None or not (isinstance(sel, (int,float)) and isinstance(rm, (int,float))):
                continue
            s += float(sel - rm) ** 2
        return s

    runs_sorted = sorted(runs, key=dist)
    match = runs_sorted[0]
    rid = match.info.run_id

    # Pull policy context
    plan = _load_plan_from_state() or {}
    pm = _primary_metric(cfg, plan)
    mr = _min_recall(cfg, plan, None)
    thr = float(selected.get("threshold", 0.5))

    # Tag the run so the model card / UI can display serving threshold
    client.set_tag(rid, "decision_threshold", f"{thr:.6f}")
    client.set_tag(rid, "selection_primary_metric", pm)
    client.set_tag(rid, "selection_min_recall", f"{mr:.6f}")
    client.set_tag(rid, "selected_from_pool", json.dumps(selected.get("selection", {})))

    # Also attach a tiny artifact for downstream inference jobs
    try:
        payload = {
            "decision_threshold": thr,
            "primary_metric": pm,
            "min_recall": mr,
            "metrics_on_test": {k: selected.get(k) for k in ["auprc","auroc","precision","recall","f1"]},
            "model_name": selected["name"]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        client.log_artifact(rid, tmp_path, artifact_path="serving")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Register
    mv = mlflow.register_model(f"runs:/{rid}/model", cfg["registry"]["name"])

    # Mirror important info as model-version tags
    client.set_model_version_tag(mv.name, mv.version, "Name", f"{selected['name']}")
    client.set_model_version_tag(mv.name, mv.version, "decision_threshold", f"{thr:.6f}")
    client.set_model_version_tag(mv.name, mv.version, "selection_primary_metric", pm)
    client.set_model_version_tag(mv.name, mv.version, "selection_min_recall", f"{mr:.6f}")
    client.set_model_version_tag(mv.name, mv.version, "selected_run_id", rid)
    
    client.set_registered_model_alias(mv.name,f"{selected['name']}", mv.version)
    # convenience: copy headline metrics
    for k in ["auprc","auroc","precision","recall","f1"]:
        v = selected.get(k)
        if isinstance(v, (int,float)) and np.isfinite(v):
            client.set_model_version_tag(mv.name, mv.version, f"test_{k}", f"{float(v):.6f}")

    return {"model_version": mv.version, "model_name": mv.name, "run_id": rid, "decision_threshold": thr}
