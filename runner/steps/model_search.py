import io, json, boto3, numpy as np, pandas as pd, pyarrow.parquet as pq
from urllib.parse import urlparse
from ruamel.yaml import YAML
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from matplotlib import pyplot as plt

yaml = YAML(typ="safe")

# ----------------------------
# IO helpers
# ----------------------------
def _read_parquet_s3(uri):
    p = urlparse(uri); s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

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

def _load_preprocessor(exp_name):
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(exp_name)
    runs = client.search_runs(
        exp.experiment_id,
        "tags.mlflow.runName = 'features'",
        order_by=["attributes.start_time DESC"]
    )
    art_uri = runs[0].info.artifact_uri + "/preprocessor"
    return mlflow.sklearn.load_model(art_uri)

# ----------------------------
# Model factory from AI plan
# ----------------------------
def _build_candidates_from_plan(plan, cfg):
    """
    Returns dict {alias_name: estimator} limited to models present in plan.models.
    Uses plan fit_params only where they are valid constructor hyperparameters.
    (We do NOT try to pass early_stopping or callbacks into constructors.)
    """
    if not plan:
        # Fallback to previous behavior
        rs = cfg["split"]["random_state"]
        return {
            "logreg": LogisticRegression(
                max_iter=200, class_weight="balanced", n_jobs=-1, random_state=rs
            ),
            "xgb": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=rs,
            ),
            "lgbm": LGBMClassifier(
                n_estimators=300,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=rs,
            ),
            "cat": CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=rs,
            ),
        }

    # map plan names -> our aliases
    name_map = {
        "logreg": "logreg",
        "linear": "logreg",
        "elasticnet": "logreg",  # not supported here; keep as logreg baseline
        "xgboost": "xgb",
        "lightgbm": "lgbm",
        "catboost": "cat",
    }

    chosen = {}
    for m in (plan.get("models") or []):
        pname = (m.get("name") or "").lower()
        alias = name_map.get(pname)
        if not alias:
            continue

        # Copy so we can safely pop / filter
        raw_fit_params = dict(m.get("fit_params") or {})
        rs = cfg["split"]["random_state"]

        if alias == "logreg":
            # Only keep recognized constructor params from fit_params
            logreg_kwargs = {
                k: v
                for k, v in raw_fit_params.items()
                if k in ["C", "penalty", "solver"]
            }
            est = LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=rs,
                **logreg_kwargs,
            )

        elif alias == "xgb":
            # Avoid passing eval_metric & early_stopping_rounds into __init__
            eval_metric = raw_fit_params.pop("eval_metric", "logloss")
            raw_fit_params.pop("early_stopping_rounds", None)

            # Only keep safe ctor params from fit_params (e.g., scale_pos_weight)
            xgb_ctor_extras = {}
            if "scale_pos_weight" in raw_fit_params:
                xgb_ctor_extras["scale_pos_weight"] = raw_fit_params["scale_pos_weight"]

            est = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric=eval_metric,
                n_jobs=-1,
                random_state=rs,
                **xgb_ctor_extras,
            )

        elif alias == "lgbm":
            # LightGBM: ignore 'callbacks' strings in ctor; keep is_unbalance only
            raw_fit_params.pop("callbacks", None)
            lgbm_ctor_extras = {}
            if "is_unbalance" in raw_fit_params:
                lgbm_ctor_extras["is_unbalance"] = raw_fit_params["is_unbalance"]

            est = LGBMClassifier(
                n_estimators=300,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=rs,
                **lgbm_ctor_extras,
            )

        elif alias == "cat":
            # CatBoost: do NOT pass early_stopping_rounds into __init__
            raw_fit_params.pop("early_stopping_rounds", None)

            cat_ctor_extras = {}
            if "auto_class_weights" in raw_fit_params:
                cat_ctor_extras["auto_class_weights"] = raw_fit_params["auto_class_weights"]

            est = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=rs,
                **cat_ctor_extras,
            )
        else:
            continue

        chosen[alias] = est

    # if plan produced nothing (edge case), fallback to defaults
    if not chosen:
        return _build_candidates_from_plan(None, cfg)
    return chosen

def _primary_metric(plan, cfg):
    pm = (plan or {}).get("evaluation", {}).get("primary_metric")
    if pm:
        return pm.lower()
    # fallback to cfg
    pm = cfg.get("metrics", {}).get("primary")
    return (pm or "auprc").lower()

def _sort_results(results, primary_metric):
    key = primary_metric
    # missing/NaN-safe sorting
    def metric_val(x):
        v = x.get(key)
        return -1e9 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    return sorted(results, key=lambda x: -metric_val(x))

# ----------------------------
# Main
# ----------------------------
def run_model_search(cfg, features_base):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(cfg["experiment_name"])

    # Load data
    train = _read_parquet_s3(f"{features_base}/train.parquet")
    val = _read_parquet_s3(f"{features_base}/val.parquet")
    target = cfg["target"]

    Xtr, ytr = train.drop(columns=[target]), train[target]
    Xv, yv = val.drop(columns=[target]), val[target]

    # Load preprocessor from the correct experiment
    pre = _load_preprocessor(cfg["experiment_name"])

    # Load plan (if present)
    plan = _load_plan_from_state()
    primary_metric = _primary_metric(plan, cfg)

    # Label encoding (shared across models)
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    # Map validation labels to seen train classes; unknown → NaN
    label2id = {lbl: i for i, lbl in enumerate(le.classes_)}
    yv_enc = yv.map(label2id)

    with mlflow.start_run(run_name="label_mapping", nested=True):
        mlflow.log_dict({"classes": le.classes_.tolist()}, "label_mapping.json")

    # Build candidate estimators from plan (or defaults)
    candidates = _build_candidates_from_plan(plan, cfg)

    # Also log which candidates we used and the primary metric
    with mlflow.start_run(run_name="model_search_plan_info", nested=True):
        mlflow.log_dict(
            {
                "primary_metric": primary_metric,
                "candidates": list(candidates.keys()),
                "plan_present": bool(plan),
            },
            "plan_info.json",
        )

    results = []
    for alias, clf in candidates.items():
        with mlflow.start_run(run_name=f"cand_{alias}"):
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            # try-fit; be robust to occasional failures
            try:
                pipe.fit(Xtr, ytr_enc)
            except Exception as e:
                mlflow.set_tag("fit_error", str(e))
                mlflow.sklearn.log_model(pipe, "model")
                results.append(
                    {"name": alias, "auprc": float("nan"), "auroc": float("nan"), "f1": float("nan")}
                )
                continue

            # restrict val to rows whose labels were seen in training
            mask = yv_enc.notna()
            Xv_use = Xv[mask]
            yv_enc_use = yv_enc[mask].astype(int)

            if len(yv_enc_use) == 0:
                mlflow.set_tag("metrics_skipped", "no_val_labels_in_train_classes")
                auprc = float("nan")
                auroc = float("nan")
                f1m = float("nan")
                mlflow.sklearn.log_model(pipe, "model")
                results.append({"name": alias, "auprc": auprc, "auroc": auroc, "f1": f1m})
                continue

            # Predict
            try:
                proba = pipe.predict_proba(Xv_use)
            except Exception:
                # some linear models w/ 'ovr' might not expose predict_proba for multiclass if solver unsupported
                try:
                    # fallback: decision_function -> pseudo prob via minmax
                    dec = pipe.decision_function(Xv_use)
                    dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
                    proba = dec
                except Exception:
                    proba = None

            try:
                y_pred_enc = pipe.predict(Xv_use)
            except Exception:
                y_pred_enc = None

            present = np.unique(yv_enc_use)
            # harmonize proba shape
            if proba is not None:
                if proba.ndim == 1:
                    proba = np.vstack([1 - proba, proba]).T
                # take only columns for present classes
                proba_present = proba[:, present]
            else:
                proba_present = None

            # Metrics (macro)
            try:
                yv_b = label_binarize(yv_enc_use, classes=present)
            except Exception:
                yv_b = None

            # AUC-ROC
            try:
                auroc = roc_auc_score(
                    yv_b, proba_present, average="macro", multi_class="ovr"
                )
            except Exception:
                auroc = float("nan")

            # AUPRC
            try:
                auprc = average_precision_score(
                    yv_b, proba_present, average="macro"
                )
            except Exception:
                auprc = float("nan")

            # F1
            try:
                f1m = f1_score(yv_enc_use, y_pred_enc, average="macro")
            except Exception:
                f1m = float("nan")

            mlflow.log_metrics(
                {"auprc": float(auprc), "auroc": float(auroc), "f1": float(f1m)}
            )

            # Pretty PR curve for the most frequent class in val
            try:
                top_cls_enc = yv_enc_use.value_counts().idxmax()
                top_idx = list(present).index(top_cls_enc)
                p, r, _ = precision_recall_curve(
                    (yv_enc_use == top_cls_enc).astype(int),
                    proba_present[:, top_idx],
                )
                plt.figure()
                plt.plot(r, p)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                top_cls_label = le.inverse_transform([top_cls_enc])[0]
                plt.title(f"PRC {alias} (OVR: '{top_cls_label}')")
                mlflow.log_figure(
                    plt.gcf(), f"prc_{alias}_ovr_{top_cls_label}.png"
                )
                plt.close()
            except Exception:
                pass

            mlflow.sklearn.log_model(pipe, "model")
            results.append(
                {"name": alias, "auprc": float(auprc), "auroc": float(auroc), "f1": float(f1m)}
            )

    # Sort by the plan's primary metric (default auprc)
    return _sort_results(results, primary_metric)
