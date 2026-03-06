import io, json, time, boto3, numpy as np, pandas as pd, pyarrow.parquet as pq
from urllib.parse import urlparse
import optuna, mlflow
from ruamel.yaml import YAML

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.preprocessing import FunctionTransformer, label_binarize, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

yaml = YAML(typ="safe")

# ----------------------------
# IO helpers
# ----------------------------
def _read_parquet_s3(uri):
    p = urlparse(uri); s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _read_s3_text(uri: str) -> str:
    p = urlparse(uri); s3 = boto3.client("s3")
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
# Preprocessor loading + date safety
# ----------------------------
def _coerce_datetimes_to_month_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime64 columns to YYYY-MM strings so downstream encoders won't choke."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    for col in out.columns:
        if np.issubdtype(out[col].dtype, np.datetime64):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.to_period("M").astype(str)
    return out

def _load_pre():
    """
    Load the most recent 'features' preprocessor across all experiments.
    If the loaded preprocessor is effectively empty, build a simple
    auto-preprocessor that:
      - coerces datetimes to YYYY-MM strings
      - one-hot encodes non-numeric cols
      - passes numeric cols through
    """
    client = mlflow.tracking.MlflowClient()

    if hasattr(client, "search_experiments"):
        exps = client.search_experiments()
    else:
        exps = client.list_experiments()

    best = None
    for exp in exps:
        runs = client.search_runs(
            [exp.experiment_id],
            "tags.mlflow.runName = 'features'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            r = runs[0]
            if (best is None) or (r.info.start_time > best.info.start_time):
                best = r

    if best is None:
        raise RuntimeError("Could not locate a 'features' run in MLflow to load preprocessor.")

    base_pre = mlflow.sklearn.load_model(best.info.artifact_uri + "/preprocessor")

    # Case 1: preprocessor is a non-empty ColumnTransformer -> use as-is
    if isinstance(base_pre, ColumnTransformer) and len(base_pre.transformers) > 0:
        safe_pre = Pipeline([
            ("fix_dates", FunctionTransformer(_coerce_datetimes_to_month_str, validate=False)),
            ("pre", base_pre),
        ])
        return safe_pre

    # Case 2: preprocessor is "empty" or not a ColumnTransformer
    # Build an automatic numeric + categorical pipeline.
    auto_pre = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )

    safe_pre = Pipeline([
        ("fix_dates", FunctionTransformer(_coerce_datetimes_to_month_str, validate=False)),
        ("pre", auto_pre),
    ])
    return safe_pre


# ----------------------------
# AI-plan helpers
# ----------------------------
_NAME_MAP = {
    "logreg": "logreg",
    "linear": "logreg",
    "elasticnet": "logreg",  # we’ll treat elasticnet request as a logreg baseline here
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "catboost": "cat",
}

def _find_model_block(plan, alias):
    if not plan: return None
    target_name = None
    # reverse-map alias -> canonical plan name
    for k, v in _NAME_MAP.items():
        if v == alias:
            target_name = k
            break
    if target_name is None:  # alias not mapped
        return None
    for m in (plan.get("models") or []):
        if str(m.get("name","")).lower() == target_name:
            return m
    return None

def _sample_from_space(trial, key, spec):
    # dict -> numeric range / structured spec
    if isinstance(spec, dict):
        # 1) Explicit "categorical" format: {"categorical": [...]}
        if "categorical" in spec:
            return trial.suggest_categorical(key, list(spec["categorical"]))

        # 2) Explicit "loguniform" format: {"loguniform": [low, high]}
        if "loguniform" in spec:
            low, high = spec["loguniform"]
            return trial.suggest_float(key, float(low), float(high), log=True)

        # 3) Generic low/high (+ optional log) format
        if "low" in spec and "high" in spec:
            a, b = spec["low"], spec["high"]
            log = bool(spec.get("log", False))

            # ints (but not bools!) -> suggest_int
            if (
                isinstance(a, int) and not isinstance(a, bool) and
                isinstance(b, int) and not isinstance(b, bool)
            ):
                return trial.suggest_int(key, int(a), int(b), log=log)

            # floats -> suggest_float
            return trial.suggest_float(key, float(a), float(b), log=log)

        # Fallback: treat as fixed value (plan already resolved it)
        return spec

    # list / tuple -> categorical OR range
    if isinstance(spec, (list, tuple)):
        # If length == 2 and both numeric *and not bools*, treat as numeric range
        if len(spec) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec
        ):
            a, b = spec
            if isinstance(a, int) and isinstance(b, int):
                return trial.suggest_int(key, int(a), int(b))
            return trial.suggest_float(key, float(a), float(b))

        # Otherwise: treat as categorical (this now correctly handles [True, False])
        return trial.suggest_categorical(key, list(spec))

    # scalar -> fixed
    return spec



def _params_from_plan(trial, alias, plan_space: dict | None):
    """
    Build params dict for a given model alias using the plan-defined search space.
    If no plan space given, fall back to sensible defaults.
    """
    if alias == "xgb":
        defaults = {
            "n_estimators": [200, 1000],
            "max_depth": [3, 10],
            "learning_rate": {"loguniform": [1e-3, 0.2]},
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
        }
    elif alias == "lgbm":
        defaults = {
            "n_estimators": [300, 1200],
            "num_leaves": [31, 255],
            "learning_rate": {"loguniform": [1e-3, 0.2]},
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
        }
    elif alias == "logreg":
        defaults = {
            "C": {"loguniform": [1e-3, 10.0]},
            "class_weight": {"categorical": [None, "balanced"]},
        }
    else:
        # not supported in evaluate step, skip
        defaults = {}

    space = dict(defaults)
    if plan_space:
        # overlay/override with plan definitions
        space.update(plan_space)

    params = {}
    for k, spec in space.items():
        params[k] = _sample_from_space(trial, k, spec)
    return params

def _build_estimator(alias, params, fit_params):
    # remove things that should NOT go into the estimator's __init__
    rs = fit_params.pop("random_state", None)
    fit_params.pop("early_stopping_rounds", None)

    # Allow plan to override eval_metric, but don't pass it twice
    eval_metric = fit_params.pop("eval_metric", None)

    # 🔑 Strip “search-spec” dicts like {"type": ..., "choices": ...}
    clean_params = {
        k: v for k, v in params.items()
        if not isinstance(v, dict)
    }

    if alias == "xgb":
        est = XGBClassifier(
            tree_method="hist",
            eval_metric=eval_metric or "logloss",
            n_jobs=-1,
            random_state=rs if rs is not None else 42,
            **clean_params,
            **fit_params,
        )

    elif alias == "lgbm":
        # ---- handle is_unbalance vs scale_pos_weight conflict ----
        spw = clean_params.get("scale_pos_weight", None)
        is_unb = clean_params.get("is_unbalance", None)

        if spw is not None and spw != 1:
            # Prefer explicit scale_pos_weight, drop is_unbalance
            clean_params.pop("is_unbalance", None)
        elif is_unb is not None:
            # scale_pos_weight is None or == 1 -> drop it, keep is_unbalance
            clean_params.pop("scale_pos_weight", None)

        est = LGBMClassifier(
            n_jobs=-1,
            random_state=rs if rs is not None else 42,
            **clean_params,
            **fit_params,
        )

    elif alias == "logreg":
        lr_params = {k: v for k, v in clean_params.items() if k != "solver"}
        est = LogisticRegression(
            max_iter=2000,
            n_jobs=-1 if "n_jobs" in LogisticRegression().get_params() else None,
            **lr_params,
        )
    else:
        raise ValueError(f"unsupported model '{alias}'")

    return est





# ----------------------------
# Objective factory (primary metric from plan if available)
# ----------------------------
def _metric_value(primary_metric, y_true, proba, present_classes=None):
    primary_metric = (primary_metric or "auprc").lower()
    if primary_metric == "auroc":
        # macro OVR for multiclass, or binary ROC-AUC
        if proba.ndim == 1:
            return roc_auc_score(y_true, proba)
        if present_classes is None:
            present_classes = np.arange(proba.shape[1])
        yb = label_binarize(y_true, classes=present_classes)
        return roc_auc_score(yb, proba[:, present_classes], average="macro", multi_class="ovr")
    # default AUPRC
    if proba.ndim == 1:
        return average_precision_score(y_true, proba, pos_label=1)
    if present_classes is None:
        present_classes = np.arange(proba.shape[1])
    yb = label_binarize(y_true, classes=present_classes)
    # macro-average across classes
    try:
        return average_precision_score(yb, proba[:, present_classes], average="macro")
    except Exception:
        return 0.0

def objective_factory(alias, Xtr, ytr, Xv, yv, pre, plan, random_state: int = 42):
    model_block = _find_model_block(plan, alias) or {}
    fit_params = dict((model_block.get("fit_params") or {}))
    plan_space = model_block.get("hpo") or {}
    primary_metric = ((plan or {}).get("evaluation") or {}).get("primary_metric", "auprc")

    def obj(trial):
        params = _params_from_plan(trial, alias, plan_space)
        clf = _build_estimator(alias, params, fit_params.copy())

        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)

        # Predict proba; robust to binary/multiclass
        try:
            proba = pipe.predict_proba(Xv)
        except Exception:
            # fall back to decision_function → [0,1] scaling
            dec = pipe.decision_function(Xv)
            if dec.ndim == 1:
                mn, mx = float(dec.min()), float(dec.max()) + 1e-12
                proba = (dec - mn) / (mx - mn)
            else:
                mn, mx = dec.min(axis=0), dec.max(axis=0) + 1e-12
                proba = (dec - mn) / (mx - mn)

        # Align shapes for metric computation
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            # binary proba: keep positive class prob only for binary metrics
            pos = proba[:, 1]
        elif isinstance(proba, np.ndarray) and proba.ndim == 1:
            pos = proba
        else:
            pos = None

        # choose metric value
        if (proba is not None) and isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] > 2:
            present = np.unique(yv)
            mv = _metric_value(primary_metric, yv, proba, present_classes=present)
        else:
            mv = _metric_value(primary_metric, yv, pos if pos is not None else proba)

        trial.set_user_attr(primary_metric, float(mv))
        return float(mv)

    return obj

def _encode_target(y: pd.Series) -> pd.Series:
    """
    Ensure binary targets are numeric 0/1.
    Handles:
      - bool dtype
      - 'false'/'true' strings
      - '0'/'1' (and '0.0'/'1.0') strings
    Leaves other types as-is.
    """
    # Boolean -> 0/1
    if y.dtype == bool:
        return y.astype(int)

    if y.dtype == object:
        # Normalize unique non-null values as lowercase strings without spaces
        vals = {str(v).strip().lower() for v in y.dropna().unique()}

        # 'false'/'true'
        if vals.issubset({"false", "true"}):
            return (
                y.astype(str)
                 .str.strip()
                 .str.lower()
                 .map({"false": 0, "true": 1})
                 .astype(int)
            )

        # '0'/'1' (or '0.0'/'1.0')
        if vals.issubset({"0", "1", "0.0", "1.0"}):
            return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

        # Generic attempt: try to coerce to numeric and check if it's binary
        y_num = pd.to_numeric(y, errors="ignore")
        if np.issubdtype(y_num.dtype, np.number):
            uniq = set(y_num.dropna().unique())
            if uniq.issubset({0, 1}):
                return y_num.astype(int)

    # Already numeric / multi-class / something else → return as-is
    return y


# ----------------------------
# Driver
# ----------------------------
def run_hpo(cfg, features_base, top_models):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(cfg["experiment_name"])

    # Data
    # train = _read_parquet_s3(f"{features_base}/train.parquet")
    # val   = _read_parquet_s3(f"{features_base}/val.parquet")
    # Xtr, ytr = train.drop(columns=[cfg["target"]]), train[cfg["target"]]
    # Xv,  yv  = val.drop(columns=[cfg["target"]]),   val[cfg["target"]]
    train = _read_parquet_s3(f"{features_base}/train.parquet")
    val   = _read_parquet_s3(f"{features_base}/val.parquet")

    Xtr = train.drop(columns=[cfg["target"]])
    ytr = _encode_target(train[cfg["target"]])

    Xv  = val.drop(columns=[cfg["target"]])
    yv  = _encode_target(val[cfg["target"]])

    # Preprocessor
    pre = _load_pre()
    rs = int(cfg.get("split", {}).get("random_state", 42))

    # AI plan (budget + per-model spaces)
    plan = _load_plan_from_state() or {}
    plan_hpo = (plan.get("hpo") or {})
    n_trials = int(plan_hpo.get("n_trials", cfg["hpo"]["n_trials"]))
    if "timeout_min" in plan_hpo:
        timeout = int(plan_hpo["timeout_min"])
    else:
        timeout = int(cfg["hpo"]["timeout_minutes"])
    timeout_min = timeout
    timeout_sec = None if timeout_min is None else int(timeout_min) * 60

    # Respect evaluate's supported models
    supported = {"xgb", "lgbm", "logreg"}
    models = [m for m in top_models if m in supported]

    results = {}
    for alias in models:
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{alias}_study",
            sampler=optuna.samplers.TPESampler(seed=rs),
        )
        with mlflow.start_run(run_name=f"hpo_{alias}"):
            # log the space we're about to sample (for transparency)
            mb = _find_model_block(plan, alias) or {}
            mlflow.log_dict({"alias": alias,
                             "plan_space": mb.get("hpo", {}),
                             "fit_params": mb.get("fit_params", {}),
                             "budget": {"n_trials": n_trials, "timeout_min": timeout_min}},
                            f"hpo_{alias}_plan.json")

            study.optimize(
                objective_factory(alias, Xtr, ytr, Xv, yv, pre, plan, rs),
                n_trials=n_trials,
                timeout=timeout_sec
            )
            best = dict(study.best_params)
            best_value = float(study.best_value)
            best["best_value"] = best_value

            # Persist best
            mlflow.log_params({f"{alias}_{k}": v for k, v in best.items() if k != "best_value"})
            primary_metric = ((plan.get("evaluation") or {}).get("primary_metric", "auprc")).lower()
            mlflow.log_metric(f"{alias}_best_{primary_metric}", best_value)

            results[alias] = best

    return results
