import io, os, json, math, boto3
from urllib.parse import urlparse
from datetime import datetime
import mlflow
from ruamel.yaml import YAML
from runner.utils.mlflow_utils import setup_mlflow

yaml = YAML(typ="safe")

def _yaml_to_str(data) -> str:
    buf = io.StringIO()
    yaml.dump(data, buf)        # ruamel.yaml needs a stream
    return buf.getvalue()

def _read_s3_text(uri: str) -> str:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _put_s3_bytes(bucket, key, data: bytes, content_type="application/octet-stream"):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"

def _load_policies():
    try:
        with open("configs/policies.yaml") as f:
            pol = yaml.load(f) or {}
    except Exception:
        pol = {}
    defaults = {
        "thresholds": {
            "high_cardinality_unique_ratio": 0.5,
            "imbalance_ratio": 3.0
        },
        "min_recall": 0.6,
        "time_aware_split_min_days": 180,
        "allow_text": False
    }
    # merge (shallow)
    for k, v in defaults.items():
        pol.setdefault(k, v)
    return pol

def _class_counts(card):
    cc = (card.get("target") or {}).get("class_counts") or {}
    total = sum(cc.values()) if cc else 0
    if not total: return cc, total, None
    # try to infer pos label from card
    pos = (card.get("target") or {}).get("positive_class")
    return cc, total, pos

def _compute_scale_pos_weight(cc, pos_label):
    if pos_label is None or str(pos_label) not in [str(k) for k in cc.keys()]:
        # try common positives
        if 1 in cc: pos_label = 1
        elif "1" in cc: pos_label = "1"
        elif True in cc: pos_label = True
        else:
            # pick minority
            pos_label = min(cc, key=cc.get)
    pos = cc.get(pos_label, cc.get(str(pos_label), 0))
    neg = sum(cc.values()) - pos
    return max(1.0, float(neg) / max(1, pos)), pos_label

def _collect_columns(card):
    cols = card.get("columns", [])
    cats_low, cats_high, nums, bools, texts, dts, ids = [], [], [], [], [], [], []
    high_card_threshold = (card.get("policy") or {}).get("thresholds", {}).get("high_cardinality_unique_ratio", 0.5)
    for c in cols:
        t = c.get("inferred_type")
        nm = c.get("name")
        if c.get("role") == "target": 
            continue
        if t == "categorical":
            (cats_high if c.get("unique_ratio", 0) > high_card_threshold else cats_low).append(nm)
        elif t == "numeric":
            nums.append(nm)
        elif t == "boolean":
            bools.append(nm)
        elif t == "text":
            texts.append(nm)
        elif t == "datetime":
            dts.append(nm)
        elif t == "id":
            ids.append(nm)
    return {
        "categorical_low": cats_low,
        "categorical_high": cats_high,
        "numeric": nums,
        "boolean": bools,
        "text": texts,
        "datetime": dts,
        "ids": ids
    }

def _maybe_time_aware(card, policies, cfg_timestamp_col):
    # If there’s a declared timestamp, and coverage_days big enough → recommend temporal split
    dcols = [c for c in card.get("columns", []) if c.get("inferred_type") == "datetime"]
    cov_days = None
    if cfg_timestamp_col:
        for c in dcols:
            if c["name"] == cfg_timestamp_col:
                cov_days = c.get("datetime", {}).get("coverage_days")
                break
    else:
        if dcols:
            cov_days = dcols[0].get("datetime", {}).get("coverage_days")
    if cov_days is None: 
        return False
    return float(cov_days) >= float(policies["time_aware_split_min_days"])

def run_plan(cfg, state):
    setup_mlflow(cfg["experiment_name"])
    if "data_card_s3" not in state:
        raise RuntimeError("data_card_s3 missing in state; run EDA first.")
    card = json.loads(_read_s3_text(state["data_card_s3"]))
    policies = _load_policies()
    card["policy"] = policies  # attach for easy access in helpers

    n_rows = (card.get("dataset") or {}).get("n_rows", 0)
    target = (card.get("target") or {}).get("name")
    task = (card.get("target") or {}).get("task") or cfg.get("task_type")
    imbalance_ratio = (card.get("target") or {}).get("imbalance_ratio")
    suspected_ids = (card.get("quality") or {}).get("suspected_ids", [])
    leakage_cols = (card.get("quality") or {}).get("possible_leakage_columns", [])
    types = _collect_columns(card)

    # class imbalance + scale_pos_weight (binary only)
    cc, total, pos_label = _class_counts(card)
    scale_pos_weight = None
    if task == "binary" and cc:
        spw, pos_label = _compute_scale_pos_weight(cc, pos_label)
        scale_pos_weight = round(spw, 4)

    time_aware = _maybe_time_aware(card, policies, cfg.get("timestamp_column"))

    # feature plan
    feature_plan = {
        "drop": sorted(set(suspected_ids + leakage_cols)),
        "imputation": {
            "numeric": "median",
            "categorical": "most_frequent",
            "boolean": "most_frequent"
        },
        "encoding": {
            "low_cardinality": {"method": "onehot"},
            "high_cardinality": {"method": "target_encoding"},
            "high_cardinality_threshold": policies["thresholds"]["high_cardinality_unique_ratio"]
        },
        "scaling": {
            "apply_to": "numeric",
            "when_models_include": ["logreg", "linear"]
        },
        "datetime": {
            "columns": types["datetime"],
            "derived": ["month_sin_cos"] if cfg.get("timestamp_column") else []
        },
        "text": {
            "columns": types["text"],
            "strategy": "ignore" if not policies.get("allow_text") else "tfidf"
        },
        "columns_by_type": types  # for downstream steps to reference
    }

    # candidate models (rules-only, sized by rows)
    smallish = n_rows <= 200_000
    models = []
    if task in ("binary", "multiclass"):
        models += [
            {
                "name": "lightgbm",
                "only_if": True,
                "fit_params": {"is_unbalance": bool(imbalance_ratio and imbalance_ratio >= policies["thresholds"]["imbalance_ratio"])},
                "hpo": {
                    "num_leaves": [15, 255],
                    "max_depth": [-1, 12],
                    "learning_rate": {"loguniform": [1e-3, 0.2]},
                    "feature_fraction": [0.6, 1.0],
                    "bagging_fraction": [0.6, 1.0],
                    "min_data_in_leaf": [max(10, int(n_rows * 0.001)), max(20, int(n_rows * 0.05))],
                    "lambda_l1": [0.0, 5.0],
                    "lambda_l2": [0.0, 10.0]
                }
            },
            {
                "name": "xgboost",
                "only_if": True,
                "fit_params": {"scale_pos_weight": scale_pos_weight} if (task == "binary" and scale_pos_weight) else {},
                "hpo": {
                    "max_depth": [3, 12],
                    "min_child_weight": [1, 20],
                    "subsample": [0.5, 1.0],
                    "colsample_bytree": [0.5, 1.0],
                    "eta": {"loguniform": [1e-3, 0.2]},
                    "gamma": [0.0, 10.0],
                    "lambda": [0.0, 10.0],
                    "alpha": [0.0, 5.0],
                    "n_estimators": [200, 1200]
                }
            },
            {
                "name": "catboost",
                "only_if": smallish,  # faster on smaller datasets in our setup
                "fit_params": {"auto_class_weights": "Balanced" if (task == "binary" and imbalance_ratio and imbalance_ratio >= policies["thresholds"]["imbalance_ratio"]) else None},
                "hpo": {
                    "depth": [4, 10],
                    "learning_rate": {"loguniform": [1e-3, 0.2]},
                    "l2_leaf_reg": [1.0, 10.0],
                    "iterations": [300, 1500]
                }
            },
            {
                "name": "logreg",
                "only_if": True,
                "requires_scaling": True,
                "penalty": "l2",
                "hpo": {"C": {"loguniform": [1e-3, 10.0]} }
            }
        ]
    else:  # regression
        models += [
            {
                "name": "lightgbm",
                "hpo": {
                    "num_leaves": [15, 255],
                    "max_depth": [-1, 12],
                    "learning_rate": {"loguniform": [1e-3, 0.2]},
                    "feature_fraction": [0.6, 1.0],
                    "bagging_fraction": [0.6, 1.0],
                    "min_data_in_leaf": [max(10, int(n_rows * 0.001)), max(20, int(n_rows * 0.05))],
                    "lambda_l1": [0.0, 5.0],
                    "lambda_l2": [0.0, 10.0]
                }
            },
            {
                "name": "xgboost",
                "hpo": {
                    "max_depth": [3, 12],
                    "min_child_weight": [1, 20],
                    "subsample": [0.5, 1.0],
                    "colsample_bytree": [0.5, 1.0],
                    "eta": {"loguniform": [1e-3, 0.2]},
                    "gamma": [0.0, 10.0],
                    "lambda": [0.0, 10.0],
                    "alpha": [0.0, 5.0],
                    "n_estimators": [200, 1200]
                }
            },
            {"name": "elasticnet", "requires_scaling": True, "hpo": {"alpha": {"loguniform": [1e-4, 10.0]}, "l1_ratio": [0.0, 1.0]}}
        ]

    eval_plan = {
        "primary_metric": cfg.get("metrics", {}).get("primary", "auprc" if task in ("binary", "multiclass") else "rmse"),
        "secondary_metrics": cfg.get("metrics", {}).get("secondary", []),
        "threshold_tuning": {
            "enabled": task == "binary",
            "constraint": {"min_recall": policies.get("min_recall", 0.6)},
            "optimize_for": "max_f1_at_recall_constraint"
        }
    }

    split_plan = {
        "method": "time" if time_aware else cfg.get("split", {}).get("method", "stratified" if task in ("binary","multiclass") else "random"),
        "params": {
            "test_size": cfg.get("split", {}).get("test_size", 0.2),
            "val_size": cfg.get("split", {}).get("val_size", 0.2),
            "random_state": cfg.get("split", {}).get("random_state", 42),
            "timestamp_column": cfg.get("timestamp_column")
        }
    }

    plan = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "n_rows": n_rows,
            "n_cols": (card.get("dataset") or {}).get("n_cols"),
            "target": target,
            "task": task,
            "positive_class": pos_label
        },
        "features": feature_plan,
        "models": models,
        "hpo": {
            "n_trials": cfg.get("hpo", {}).get("n_trials", 40),
            "timeout_min": cfg.get("hpo", {}).get("timeout_minutes", 60),
            "top_k_candidates": cfg.get("hpo", {}).get("top_k_candidates", 2)
        },
        "evaluation": eval_plan,
        "split": split_plan
    }

    rationale_lines = []
    rationale_lines.append(f"# AI Plan Rationale")
    rationale_lines.append(f"- Task inferred: **{task}**; target: `{target}`.")
    if task == "binary" and imbalance_ratio:
        rationale_lines.append(f"- Class imbalance ratio ≈ **{imbalance_ratio}**; computed scale_pos_weight ≈ **{scale_pos_weight}**.")
    if suspected_ids:
        rationale_lines.append(f"- Dropping suspected IDs: {', '.join(suspected_ids)}.")
    if leakage_cols:
        rationale_lines.append(f"- Possible leakage columns flagged: {', '.join(leakage_cols)}.")
    if types["categorical_high"]:
        rationale_lines.append(f"- High-cardinality categoricals: {', '.join(types['categorical_high'])} → target encoding.")
    if _maybe_time_aware(card, policies, cfg.get("timestamp_column")):
        rationale_lines.append(f"- Time coverage suggests temporal split.")
    rationale_lines.append(f"- Models include tree boosters (LGBM/XGB), CatBoost (if small), and a linear baseline.")
    rationale = "\n".join(rationale_lines) + "\n"

    # Log to MLflow
    with mlflow.start_run(run_name="plan"):
        plan_yaml = _yaml_to_str(plan)
        mlflow.log_text(plan_yaml, "plan/ai_plan.yaml")
        mlflow.log_text(rationale, "plan/ai_rationale.md")

    # Mirror to S3
    bucket = cfg["s3"]["bucket"].split("s3://",1)[1]
    key_base = f'{cfg["s3"]["prefix"]}/plan'
    plan_yaml = plan_yaml if 'plan_yaml' in locals() else _yaml_to_str(plan) 
    plan_s3 = _put_s3_bytes(bucket, f"{key_base}/ai_plan.yaml",
                        plan_yaml.encode("utf-8"),
                        content_type="text/yaml")
    rationale_s3 = _put_s3_bytes(bucket, f"{key_base}/ai_rationale.md",
                                rationale.encode("utf-8"),
                                content_type="text/markdown")

    return {"ai_plan": plan, "ai_plan_s3": plan_s3, "ai_rationale_s3": rationale_s3}
