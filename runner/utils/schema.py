from __future__ import annotations
import math, re, numpy as np, pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

# Minimal, dependency-light builder for a machine-readable "data card"

_DEFAULTS = {
    "sample_rows": 100,
    "max_topk": 10,
    "max_unique_ratio_for_cat": 0.5,
    "id_regex": r"(_ID$|^ID$|ID_)",
    "text_min_char_len": 20,
    "datetime_formats": ["%Y-%m-%d", "%Y-%m", "%Y/%m/%d"],
}

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_boolish(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    # occasionally 0/1 ints masquerade as bool
    try:
        vals = pd.Series(s.dropna().unique())[:10]
        return len(vals) <= 2 and set(pd.to_numeric(vals, errors="coerce").dropna().astype(int).astype(str)) <= {"0","1"}
    except Exception:
        return False

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _is_text(s: pd.Series, min_len: int) -> bool:
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    avg_len = sample.str.len().mean()
    return avg_len >= min_len

def _infer_type(name: str, s: pd.Series, id_column: Optional[str], id_regex: str, text_min_char_len: int) -> str:
    if id_column and name == id_column:
        return "id"
    if re.search(id_regex, name, flags=re.IGNORECASE):
        return "id"
    if _is_datetime(s):
        return "datetime"
    if _is_boolish(s):
        return "boolean"
    if _is_numeric(s):
        return "numeric"
    if _is_text(s, text_min_char_len):
        return "text"
    return "categorical"

def _topk(s: pd.Series, k: int):
    vc = s.value_counts(dropna=False).head(k)
    out = []
    for idx, cnt in vc.items():
        try:
            v = None if (isinstance(idx, float) and math.isnan(idx)) else idx
        except Exception:
            v = str(idx)
        out.append({"value": v, "count": int(cnt)})
    return out

def _pearson(x: pd.Series, y: pd.Series) -> Optional[float]:
    try:
        x2 = pd.to_numeric(x, errors="coerce")
        y2 = pd.to_numeric(y, errors="coerce")
        mask = x2.notna() & y2.notna()
        if mask.sum() < 3:
            return None
        r = np.corrcoef(x2[mask], y2[mask])[0,1]
        if np.isfinite(r):
            return float(r)
    except Exception:
        pass
    return None

def _mutual_info(x: pd.Series, y: pd.Series, task_type: Optional[str]) -> Optional[float]:
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import OrdinalEncoder
        X = pd.DataFrame({"x": x})
        # encode non-numeric columns ordinally
        if not pd.api.types.is_numeric_dtype(X["x"]):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X["x"] = enc.fit_transform(X[["x"]])
        # drop nan pairs
        df2 = pd.concat([X["x"], y], axis=1).dropna()
        if len(df2) < 20:
            return None
        if (task_type or "").lower().startswith("regress"):
            mi = mutual_info_regression(df2[["x"]], df2[y.name])
        else:
            mi = mutual_info_classif(df2[["x"]], df2[y.name])
        val = float(mi[0])
        return val if np.isfinite(val) else None
    except Exception:
        return None

def make_data_card(
    df: pd.DataFrame,
    dataset_uri: str,
    target: Optional[str],
    task_type: Optional[str],
    id_column: Optional[str],
    eda_cfg: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    cfg = {**_DEFAULTS, **(eda_cfg or {})}

    n_rows, n_cols = int(len(df)), int(len(df.columns))
    created_at = datetime.utcnow().isoformat() + "Z"

    card: Dict[str, Any] = {
        "dataset": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "name": dataset_uri.split("/")[-1],
            "source": dataset_uri,
        },
        "target": {
            "name": target,
            "task": task_type,
            "positive_class": None,
            "classes": [],
            "class_counts": {},
            "imbalance_ratio": None
        },
        "columns": [],
        "quality": {
            "suspected_ids": [],
            "possible_leakage_columns": [],
            "dup_row_pct": 0.0
        },
        "created_at": created_at
    }

    # target summary
    y = None
    if target and target in df.columns:
        y = df[target]
        if (task_type or "").lower().startswith("regress"):
            # regression: no classes
            card["target"]["task"] = "regression"
        else:
            # assume classification when values look discrete with few unique
            uniq = int(y.nunique(dropna=True))
            if uniq <= 2:
                card["target"]["task"] = "binary"
            else:
                card["target"]["task"] = "multiclass"
            counts = y.value_counts(dropna=False).to_dict()
            card["target"]["class_counts"] = {str(k): int(v) for k, v in counts.items()}
            card["target"]["classes"] = [str(k) for k in counts.keys()]
            if uniq == 2:
                # pick positive as the larger of {1, "1", True} if present; else the minority class
                if 1 in counts: card["target"]["positive_class"] = 1
                elif "1" in counts: card["target"]["positive_class"] = "1"
                elif True in counts: card["target"]["positive_class"] = True
                else:
                    # minority class
                    card["target"]["positive_class"] = min(counts, key=counts.get)
            total = sum(counts.values())
            if total > 0 and len(counts) >= 2:
                major = max(counts.values())
                minor = min(counts.values())
                card["target"]["imbalance_ratio"] = round(major / max(1, minor), 4)

    # dataset-level quality
    try:
        dup_pct = 100.0 * (1.0 - len(df.drop_duplicates()) / max(1, len(df)))
        card["quality"]["dup_row_pct"] = round(dup_pct, 4)
    except Exception:
        pass

    suspected_ids = []
    leakage_cols = []

    # per-column
    for name in df.columns:
        s = df[name]
        inferred_type = _infer_type(name, s, id_column=id_column, id_regex=cfg["id_regex"], text_min_char_len=cfg["text_min_char_len"])

        role = "feature"
        if target and name == target:
            role = "target"
        elif id_column and name == id_column:
            role = "id"

        missing_pct = float(s.isna().mean() * 100.0)
        n_unique = int(s.nunique(dropna=True))
        unique_ratio = float(n_unique / max(1, len(s)))
        flags = {
            "constant": bool(n_unique <= 1),
            "high_cardinality": bool(unique_ratio > cfg["max_unique_ratio_for_cat"])
        }

        # numeric stats
        stats_num = None
        if inferred_type == "numeric":
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            stats_num = {
                "mean": _nanfloat(desc.get("mean")),
                "std": _nanfloat(desc.get("std")),
                "min": _nanfloat(desc.get("min")),
                "q1": _nanfloat(desc.get("25%")),
                "median": _nanfloat(desc.get("50%")),
                "q3": _nanfloat(desc.get("75%")),
                "max": _nanfloat(desc.get("max")),
            }

        # datetime stats
        if inferred_type == "datetime":
            try:
                dt_min = s.min()
                dt_max = s.max()
                coverage_days = float((dt_max - dt_min).days) if pd.notna(dt_min) and pd.notna(dt_max) else None
            except Exception:
                dt_min = dt_max = coverage_days = None
        else:
            dt_min = dt_max = coverage_days = None

        # associations with target
        assoc = {"pearson_w_target": None, "mutual_info_w_target": None}
        if y is not None and name != (target or ""):
            if inferred_type == "numeric" and pd.api.types.is_numeric_dtype(y):
                assoc["pearson_w_target"] = _pearson(s, y)
            # light-weight MI for both numeric/categorical with a discrete or continuous y
            assoc["mutual_info_w_target"] = _mutual_info(s, y, task_type=card["target"]["task"])

        # top-k for discrete-ish columns
        topk = None
        if inferred_type in {"categorical", "boolean"}:
            topk = _topk(s, int(cfg["max_topk"]))

        col_entry = {
            "name": name,
            "role": role,
            "inferred_type": inferred_type,
            "missing_pct": round(missing_pct, 4),
            "n_unique": n_unique,
            "unique_ratio": round(unique_ratio, 6),
            "flags": flags,
            "stats_num": stats_num,
            "topk": topk,
            "assoc": assoc,
            "datetime": {
                "min": None if dt_min is None or pd.isna(dt_min) else str(dt_min),
                "max": None if dt_max is None or pd.isna(dt_max) else str(dt_max),
                "coverage_days": coverage_days
            }
        }
        card["columns"].append(col_entry)

        # collect suspected ids & leakage
        if inferred_type in {"id"} or (unique_ratio > 0.9 and pd.api.types.is_object_dtype(s)):
            suspected_ids.append(name)
        if y is not None and name != (target or ""):
            try:
                same = (s == y).sum()
                if same >= len(df) - 1:  # near-perfect match
                    leakage_cols.append(name)
            except Exception:
                pass

    card["quality"]["suspected_ids"] = suspected_ids
    card["quality"]["possible_leakage_columns"] = list(sorted(set(leakage_cols)))
    return card

def _nanfloat(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None
