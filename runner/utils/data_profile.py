# runner/utils/data_profile.py
import pandas as pd
from typing import Dict, Any

def profile_dataframe(df: pd.DataFrame, max_cols: int = 50) -> Dict[str, Any]:
    """
    Returns a compact JSON-serializable schema/stat summary of the dataframe.
    """
    df_sample = df.copy()
    if df_sample.shape[1] > max_cols:
        df_sample = df_sample.iloc[:, :max_cols]

    summary = {}
    for col in df_sample.columns:
        s = df_sample[col]
        col_info = {
            "dtype": str(s.dtype),
            "non_null_count": int(s.notna().sum()),
            "null_count": int(s.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(s):
            col_info.update(
                {
                    "min": float(s.min()) if s.notna().any() else None,
                    "max": float(s.max()) if s.notna().any() else None,
                    "mean": float(s.mean()) if s.notna().any() else None,
                    "std": float(s.std()) if s.notna().any() else None,
                    "n_unique": int(s.nunique()),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_info.update(
                {
                    "min": s.min().strftime("%Y-%m-%d %H:%M:%S") if s.notna().any() else None,
                    "max": s.max().strftime("%Y-%m-%d %H:%M:%S") if s.notna().any() else None,
                    "n_unique": int(s.nunique()),
                }
            )
        else:
            col_info.update(
                {
                    "n_unique": int(s.nunique()),
                    "top_values": s.value_counts().head(5).to_dict(),
                }
            )
        summary[col] = col_info
    return summary
