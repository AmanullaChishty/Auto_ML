import io, json, boto3, numpy as np, pandas as pd, pyarrow.parquet as pq
from urllib.parse import urlparse
import mlflow, matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from ruamel.yaml import YAML

from runner.steps.hpo import _load_pre, _read_parquet_s3
from runner.utils.metrics import classification_metrics  # keep for fallback / logging

yaml = YAML(typ="safe")

# ----------------------------
# Small IO helpers
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

# ----------------------------
# Models
# ----------------------------
def _make_model(name, params, random_state=42):
    params = dict(params)

    if name == "xgb":
        return XGBClassifier(
            **params,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "lgbm":
        # ---- resolve is_unbalance vs scale_pos_weight conflict ----
        spw = params.get("scale_pos_weight", None)
        is_unb = params.get("is_unbalance", None)

        if spw is not None and spw != 1:
            # Prefer explicit scale_pos_weight, drop is_unbalance
            params.pop("is_unbalance", None)
        elif is_unb is not None:
            # scale_pos_weight is None or == 1 -> drop it, keep is_unbalance
            params.pop("scale_pos_weight", None)

        return LGBMClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "logreg":
        # keep a robust solver; params may include C, class_weight, etc.
        base = {"solver": "liblinear", "max_iter": 2000}
        base.update(params)
        return LogisticRegression(**base)

    raise ValueError(f"Unknown model: {name}")


# ----------------------------
# Threshold search on validation
# ----------------------------
def _best_threshold_on_val(y_val, proba_val, min_recall=0.6):
    """
    Grid-search thresholds ∈ unique(proba_val) ∪ linspace to maximize F1
    under recall >= min_recall. Returns (threshold, {prec, rec, f1} on val).
    If no threshold satisfies constraint, pick the one with highest recall and tag violation.
    """
    y_val = np.asarray(y_val).astype(int)
    p = np.asarray(proba_val).astype(float)

    # candidate thresholds from PR curve + a small linspace safety net
    pr, rc, thr_pr = precision_recall_curve(y_val, p)
    cand = list(thr_pr) + list(np.linspace(0.01, 0.99, 99))
    cand = np.unique(np.clip(cand, 1e-9, 1 - 1e-9))

    best = None
    best_f1 = -1.0
    feasible = False
    for t in cand:
        yhat = (p >= t).astype(int)
        r = recall_score(y_val, yhat, zero_division=0)
        if r + 1e-12 < float(min_recall):
            continue
        feasible = True
        prc = precision_score(y_val, yhat, zero_division=0)
        f1 = f1_score(y_val, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = (t, {"precision": float(prc), "recall": float(r), "f1": float(f1)})

    if feasible:
        return best[0], best[1], False  # (threshold, metrics_on_val, violated=False)

    # No feasible threshold → pick the one with max recall (break ties by F1)
    best_r = -1.0
    best_f1_any = -1.0
    best_any = None
    for t in cand:
        yhat = (p >= t).astype(int)
        r = recall_score(y_val, yhat, zero_division=0)
        f1 = f1_score(y_val, yhat, zero_division=0)
        if (r > best_r) or (r == best_r and f1 > best_f1_any):
            best_r = r; best_f1_any = f1
            prc = precision_score(y_val, yhat, zero_division=0)
            best_any = (t, {"precision": float(prc), "recall": float(r), "f1": float(f1)})
    return best_any[0], best_any[1], True  # violated=True

# ----------------------------
# Main
# ----------------------------
def run_evaluate(cfg, features_base, hpo):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(cfg["experiment_name"])

    # Load splits
    train = _read_parquet_s3(f'{features_base}/train.parquet')
    val   = _read_parquet_s3(f'{features_base}/val.parquet')
    test  = _read_parquet_s3(f'{features_base}/test.parquet')
    target = cfg["target"]

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val,   y_val   = val.drop(columns=[target]),   val[target]
    X_test,  y_test  = test.drop(columns=[target]),  test[target]
    
    pos_label = str(cfg.get("positive_label", "true"))

    def _to_binary(y):
        y = y.astype(str)
        return (y == pos_label).astype(int)

    y_train = _to_binary(y_train)
    y_val   = _to_binary(y_val)
    y_test  = _to_binary(y_test)

    # Preprocessor (frozen from features step, date-safe wrapper)
    pre = _load_pre()

    # AI plan: find threshold tuning settings
    plan = _load_plan_from_state() or {}
    tconf = ((plan.get("evaluation") or {}).get("threshold_tuning") or {})
    tt_enabled = bool(tconf.get("enabled", False))
    min_recall = float((tconf.get("constraint") or {}).get("min_recall", cfg.get("metrics", {}).get("min_recall", 0.6)))
    optimize_for = (tconf.get("optimize_for") or "max_f1_at_recall_constraint").lower()

    # For ranking we’ll keep primary metric (auprc by default)
    primary_metric = ((plan.get("evaluation") or {}).get("primary_metric") or cfg.get("metrics", {}).get("primary", "auprc")).lower()

    rs = int(cfg.get("split", {}).get("random_state", 42))

    leaderboard=[]
    for name, params in hpo.items():
        params = {k: v for k, v in params.items() if k != "best_value"}
        base_clf = _make_model(name, params, random_state=rs)

        # 1) Fit on TRAIN only for threshold selection (if enabled)
        pipe_train = Pipeline([("pre", pre), ("clf", base_clf)])
        with mlflow.start_run(run_name=f"final_{name}"):
            # ---- threshold tuning on val ----
            tuned_thr = 0.5
            val_point = None
            violated = False

            if tt_enabled and optimize_for.startswith("max_f1"):
                pipe_train.fit(X_train, y_train)
                # Try to get proba; if not, derive from decision_function
                try:
                    p_val = pipe_train.predict_proba(X_val)
                    p_val = p_val[:, 1] if p_val.ndim == 2 else p_val
                except Exception:
                    dec = pipe_train.decision_function(X_val)
                    if dec.ndim == 2 and dec.shape[1] == 2:
                        dec = dec[:, 1]
                    # Min-max scale to [0,1]
                    mn, mx = float(dec.min()), float(dec.max()) + 1e-12
                    p_val = (dec - mn) / (mx - mn)

                tuned_thr, val_point, violated = _best_threshold_on_val(y_val, p_val, min_recall=min_recall)

                mlflow.log_dict({
                    "enabled": True,
                    "min_recall": min_recall,
                    "optimize_for": optimize_for,
                    "chosen_threshold": float(tuned_thr),
                    "val_operating_point": val_point,
                    "violation_of_constraint": bool(violated)
                }, f"threshold_tuning_{name}.json")

                # optional: log threshold curve as artifact
                pr, rc, thr = precision_recall_curve(y_val, p_val)
                plt.figure(); plt.plot(rc, pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Val PRC (for threshold) - {name}")
                mlflow.log_figure(plt.gcf(), f"val_prc_{name}.png"); plt.close()

            # 2) Refit on TRAIN+VAL, score on TEST
            #    (Threshold chosen is fixed; we do NOT re-tune on test)
            clf_final = _make_model(name, params, random_state=rs)
            pipe = Pipeline([("pre", pre), ("clf", clf_final)])
            pipe.fit(pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0))

            # Probability on test (for AUPRC/AUROC and PRC curve)
            try:
                proba_test = pipe.predict_proba(X_test)
                proba_test = proba_test[:, 1] if isinstance(proba_test, np.ndarray) and proba_test.ndim == 2 else proba_test
            except Exception:
                dec = pipe.decision_function(X_test)
                if dec.ndim == 2 and dec.shape[1] == 2:
                    dec = dec[:, 1]
                mn, mx = float(dec.min()), float(dec.max()) + 1e-12
                proba_test = (dec - mn) / (mx - mn)

            # threshold-free metrics
            try:
                auroc = roc_auc_score(y_test, proba_test)
            except Exception:
                auroc = float("nan")
            try:
                auprc = average_precision_score(y_test, proba_test)
            except Exception:
                auprc = float("nan")

            # thresholded metrics (tuned or default 0.5)
            thr_use = float(tuned_thr) if tt_enabled else 0.5
            yhat_test = (proba_test >= thr_use).astype(int)
            prec = precision_score(y_test, yhat_test, zero_division=0)
            rec  = recall_score(y_test, yhat_test,  zero_division=0)
            f1   = f1_score(y_test, yhat_test,     zero_division=0)

            # confusion matrix
            cm = confusion_matrix(y_test, yhat_test, labels=[0,1])
            cm_dict = {
                "tn": int(cm[0,0]), "fp": int(cm[0,1]),
                "fn": int(cm[1,0]), "tp": int(cm[1,1])
            }

            # Log to MLflow
            mlflow.log_metrics({
                "test_auprc": float(auprc),
                "test_auroc": float(auroc),
                "test_precision": float(prec),
                "test_recall": float(rec),
                "test_f1": float(f1),
            })
            mlflow.log_dict({
                "used_threshold": thr_use,
                "constraint_min_recall": min_recall if tt_enabled else None,
                "constraint_violated": bool(violated) if tt_enabled else None,
                "confusion_matrix": cm_dict
            }, f"test_operating_point_{name}.json")

            # PR curve on test
            p, r, _ = precision_recall_curve(y_test, proba_test)
            plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Test PRC {name}")
            mlflow.log_figure(plt.gcf(), f"test_prc_{name}.png"); plt.close()

            # Confusion matrix heatmap
            try:
                plt.figure()
                plt.imshow(cm, interpolation='nearest')
                plt.title(f"Confusion Matrix @ thr={thr_use:.3f} - {name}")
                plt.xlabel("Predicted"); plt.ylabel("Actual")
                plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
                for (i,j), val in np.ndenumerate(cm):
                    plt.text(j, i, int(val), ha='center', va='center')
                mlflow.log_figure(plt.gcf(), f"confusion_matrix_{name}.png"); plt.close()
            except Exception:
                pass

            # Persist the trained model
            mlflow.sklearn.log_model(pipe, "model")

            # Compose leaderboard row
            leaderboard.append({
                "name": name,
                "auprc": float(auprc),
                "auroc": float(auroc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "threshold": thr_use,
            })

    # Sort leaderboard by the plan's primary metric
    key = (primary_metric if primary_metric in {"auprc","auroc","f1","precision","recall"} else "auprc")
    return sorted(leaderboard, key=lambda x: x.get(key, float("nan")), reverse=True)
