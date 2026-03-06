import io, os, json, math, boto3, pandas as pd, numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from urllib.parse import urlparse
from ruamel.yaml import YAML
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from runner.utils.mlflow_utils import setup_mlflow
import mlflow

yaml = YAML(typ="safe")

# ----------------------------
# IO helpers
# ----------------------------
def _read_parquet_s3(uri):
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _to_s3_parquet(df, s3_uri):
    bucket = s3_uri.split("s3://",1)[1].split("/",1)[0]
    key = s3_uri.split(bucket+"/",1)[1]
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def _read_s3_text(uri: str) -> str:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _load_plan(ai_plan_s3: str) -> dict:
    text = _read_s3_text(ai_plan_s3)
    return yaml.load(text) or {}

def _load_state_dotjson() -> dict:
    try:
        return json.load(open(".state.json"))
    except FileNotFoundError:
        return {}

# ----------------------------
# Encoders
# ----------------------------
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Minimal target mean encoder (1-d output per input column).
    - Binary classification: encodes category → P(y==positive)
    - Multiclass: encodes category → mean of label==positive_class (falls back to minority if not provided)
    - Regression: encodes category → mean(y)
    Unseen categories map to global mean. Works with Pandas series input.
    """
    def __init__(self, positive_class=None):
        self.positive_class = positive_class
        self._maps = {}
        self._global = {}
        self._is_regression = False

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetMeanEncoder requires y during fit")
        # If X is DataFrame with multiple columns, encode each separately
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # decide task via y unique
        y_ser = pd.Series(y)
        uniq = pd.unique(y_ser.dropna())
        self._is_regression = pd.api.types.is_float_dtype(y_ser) and len(uniq) > 5

        # choose positive for classification (binary/multiclass)
        pos = self.positive_class
        if not self._is_regression:
            if pos is None:
                # try common positives; else pick minority
                counts = y_ser.value_counts(dropna=False)
                if 1 in counts: pos = 1
                elif "1" in counts: pos = "1"
                elif True in counts: pos = True
                else:
                    pos = counts.idxmin()
        self._pos = pos

        for col in X.columns:
            xcol = pd.Series(X[col])
            df = pd.DataFrame({"x": xcol, "y": y_ser})
            df = df.dropna(subset=["x", "y"])
            if len(df) == 0:
                self._maps[col] = {}
                self._global[col] = None
                continue
            if self._is_regression:
                grp = df.groupby("x")["y"].mean()
                gmean = df["y"].mean()
            else:
                grp = df.groupby("x")["y"].apply(lambda s: np.mean(s == pos))
                gmean = np.mean(df["y"] == pos)
            self._maps[col] = grp.to_dict()
            self._global[col] = float(gmean) if np.isfinite(gmean) else None
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        out = {}
        for col in X.columns:
            mp = self._maps.get(col, {})
            g = self._global.get(col, None)
            out[col + "_tmean"] = X[col].map(mp).astype(float)
            if g is not None:
                out[col + "_tmean"] = out[col + "_tmean"].fillna(g)
        return pd.DataFrame(out, index=X.index)

# ----------------------------
# Core run
# ----------------------------
def run_features(cfg, dataset_uri, ai_plan_s3: str = None):
    """
    Backward compatible:
    - If ai_plan_s3 is None, the function will attempt to read it from .state.json.
    - If still missing, falls back to your original behavior (OHE for all cats, no scaling).
    """
    setup_mlflow(cfg["experiment_name"])
    with mlflow.start_run(run_name="features"):
        df = _read_parquet_s3(dataset_uri)

        # Try to load plan
        plan = None
        if ai_plan_s3 is None:
            state = _load_state_dotjson()
            ai_plan_s3 = state.get("ai_plan_s3")
        if ai_plan_s3:
            try:
                plan = _load_plan(ai_plan_s3)
            except Exception as e:
                mlflow.log_text(str(e), "features/_plan_load_error.txt")
                plan = None

        target = cfg["target"]
        # Keep legacy safety behavior
        if target in df.columns:
            df[target].fillna(1, inplace=True)

        # Timestamp parsing if present
        ts_col = (plan or {}).get("split", {}).get("params", {}).get("timestamp_column") or cfg.get("timestamp_column")
        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        # ----------------------------
        # PLAN-DRIVEN COLUMN SELECTION
        # ----------------------------
        drop_from_plan = []
        low_card_cats = []
        high_card_cats = []
        text_cols = []
        num_cols_hint = []
        bool_cols_hint = []

        if plan:
            features_plan = plan.get("features", {})
            drop_from_plan = list(features_plan.get("drop", []) or [])

            cbt = (features_plan.get("columns_by_type") or {})
            low_card_cats = list(cbt.get("categorical_low", []) or [])
            high_card_cats = list(cbt.get("categorical_high", []) or [])
            text_cols = list(cbt.get("text", []) or [])
            num_cols_hint = list(cbt.get("numeric", []) or [])
            bool_cols_hint = list(cbt.get("boolean", []) or [])

            # Text strategy: drop if 'ignore'
            text_strategy = (features_plan.get("text") or {}).get("strategy", "ignore")
            if text_strategy == "ignore":
                drop_from_plan.extend(text_cols)

        # Always drop label/id/timestamp from features
        base_drops = [c for c in [target, cfg.get("id_column"), ts_col] if c and c in df.columns]
        drop_cols = sorted(set((drop_from_plan or []) + base_drops))

        y = df[target]
        X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # If no plan, fall back to old num/cat detection
        if not plan:
            num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
            cat_low = [c for c in X_all.columns if c not in num_cols]
            cat_high = []
            scale_needed = False
            split_method = cfg["split"].get("method", "stratified" if cfg["task_type"]=="classification" else "random")
        else:
            # Intersect plan-suggested types with current columns
            existing = set(X_all.columns)
            cat_low = [c for c in low_card_cats if c in existing]
            cat_high = [c for c in high_card_cats if c in existing]
            # treat booleans like low-card categoricals (safer than numeric here)
            cat_low += [c for c in bool_cols_hint if c in existing and c not in cat_low]
            # numeric columns from hint; if empty, infer
            num_cols = [c for c in num_cols_hint if c in existing]
            if not num_cols:
                num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()

            # Scaling only if requested AND models include a linear/logreg
            model_names = [m.get("name") for m in (plan.get("models") or [])]
            scale_needed = ("logreg" in model_names) or ("linear" in model_names) or ("elasticnet" in model_names)

            split_method = (plan.get("split") or {}).get("method") or \
                           (cfg["split"]["method"] if cfg["task_type"]=="classification" else "random")

        # Ensure unique & order-stable lists
        def _uniq_keep(seq):
            seen = set(); out=[]
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        cat_low = _uniq_keep(cat_low)
        cat_high = _uniq_keep(cat_high)
        # Recompute numerics if plan wasn't explicit
        if not plan or not num_cols_hint:
            numeric_candidates = X_all.select_dtypes(include=[np.number]).columns
            num_cols = [c for c in numeric_candidates if c not in cat_low + cat_high]

        # ----------------------------
        # BUILD PREPROCESSOR
        # ----------------------------
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))] + \
                            ([("scaler", StandardScaler())] if scale_needed else []))
        cat_low_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        # Positive class hint for TME
        pos_class = (plan or {}).get("dataset", {}).get("positive_class", None)
        cat_high_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("tmean", TargetMeanEncoder(positive_class=pos_class))
        ])

        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_low: transformers.append(("cat_low", cat_low_pipe, cat_low))
        if cat_high: transformers.append(("cat_high", cat_high_pipe, cat_high))

        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        # ----------------------------
        # SPLIT (time-aware if requested by plan)
        # ----------------------------
        test_size = float(cfg["split"]["test_size"])
        val_size = float(cfg["split"]["validation_size"])
        rnd = int(cfg["split"]["random_state"])

        if split_method == "time" and ts_col and ts_col in df.columns:
            # Sort by time ascending, take tail for test, then tail for val from remaining
            df_sorted = df.sort_values(ts_col)
            n = len(df_sorted)
            n_test = int(round(n * test_size))
            n_trainval = n - n_test
            n_val = int(round(n_trainval * (val_size / max(1e-9, (1 - test_size)))))

            train = df_sorted.iloc[: n_trainval - n_val]
            val   = df_sorted.iloc[n_trainval - n_val : n_trainval]
            test  = df_sorted.iloc[n_trainval : ]

            X_train, y_train = train.drop(columns=drop_cols, errors="ignore"), train[target]
            X_val,   y_val   = val.drop(columns=drop_cols, errors="ignore"),   val[target]
            X_test,  y_test  = test.drop(columns=drop_cols, errors="ignore"),  test[target]
        else:
            # safe stratify
            stratify = None
            if cfg["task_type"] == "classification":
                vc = y.value_counts()
                if (len(vc) >= 2) and (vc.min() >= 2):
                    stratify = y

            # split: test first, then val
            X, y_use = X_all, y
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y_use, test_size=test_size, random_state=rnd, stratify=stratify
            )
            val_size_adj = val_size / max(1e-9, (1 - test_size))
            stratify_tv = y_trainval if stratify is not None else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_size_adj, random_state=rnd, stratify=stratify_tv
            )

        # ----------------------------
        # FIT PREPROCESSOR ON TRAIN ONLY (avoid leakage)
        # ----------------------------
        pre.fit(X_train, y_train)

        # Persist splits as raw (untransformed) with target (like before)
        bucket = cfg["s3"]["bucket"].replace("s3://","")
        base = f'{cfg["s3"]["prefix"].strip("/")}/features'
        for name, dfx, yv in [("train", X_train, y_train),
                              ("val",   X_val,   y_val),
                              ("test",  X_test,  y_test)]:
            _to_s3_parquet(pd.concat([dfx, yv.rename(target)], axis=1),
                           f's3://{bucket}/{base}/{name}.parquet')

        # Log schema + plan consumption
        feature_schema = {
            "used_plan": bool(plan is not None),
            "drop_cols": drop_cols,
            "num_cols": num_cols,
            "cat_low_cols": cat_low,
            "cat_high_cols": cat_high,
            "timestamp_column": ts_col,
            "split_method": split_method
        }
        mlflow.log_dict(feature_schema, "feature_schema.json")
        mlflow.sklearn.log_model(pre, "preprocessor")

        return {
            "features_base": f's3://{bucket}/{base}',
            "num_cols": num_cols,
            "cat_cols": cat_low + cat_high,
            "cat_low_cols": cat_low,
            "cat_high_cols": cat_high,
            "used_plan": bool(plan is not None)
        }
