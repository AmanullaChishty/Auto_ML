# runner/steps/ai_plan_llm.py
import io, os, json, boto3
from urllib.parse import urlparse
from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Dict

import mlflow
from ruamel.yaml import YAML

from runner.utils.mlflow_utils import setup_mlflow

yaml = YAML(typ="safe")


def _read_s3_text(uri: str) -> str:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _put_s3_bytes(bucket: str, key: str, data: bytes, content_type="application/octet-stream"):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"

def _deep_merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _deep_merge(out.get(k), v)
        return out
    return deepcopy(over if over is not None else base)

def _validate_plan(p):
    # Minimal shape checks; keep permissive
    assert "features" in p and "models" in p and "evaluation" in p and "split" in p, "plan missing keys"
    return p

def _load_rules_plan(cfg, state):
    """
    Calls your existing rules planner to get a baseline plan.
    Assumes you added Step-2 as runner/steps/plan.py with run_plan(cfg, state).
    """
    from runner.steps.plan import run_plan as _rules
    return _rules(cfg, state)

def _card_preview(card: dict, max_cols: int = 60) -> dict:
    # Trim to avoid huge prompts
    c2 = deepcopy(card)
    cols = c2.get("columns", [])
    if len(cols) > max_cols:
        c2["columns"] = cols[:max_cols]
        c2["columns_truncated"] = True
    return c2

# helper to turn a dict into a YAML string using ruamel
def _yaml_to_str(data) -> str:
    buf = io.StringIO()
    yaml.dump(data, buf)
    return buf.getvalue()

def parse_llm_json(raw_resp: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response that may be wrapped in ```json fences.
    Raises JSONDecodeError on failure instead of silently returning {}.
    """
    if raw_resp is None:
        raise json.JSONDecodeError("LLM response is None", "", 0)

    s = raw_resp.strip()

    # Strip leading ```... fence if present
    if s.startswith("```"):
        # Drop the first line (``` or ```json)
        lines = s.splitlines()
        lines = lines[1:]  # drop fence line

        # If the first remaining line is just "json", drop that too
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]

        s = "\n".join(lines)
        # Strip trailing ``` if present
        s = re.sub(r"```$", "", s.strip())

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Log for debugging, then propagate so caller can decide to retry
        print("Failed to parse LLM JSON:", e)
        print("Raw response snippet:", s[:500])
        raise

def run_ai_plan_llm(cfg, state):
    """
    1) Build baseline (rules) plan.
    2) Ask LLM for a plan (same schema) using the EDA data card.
    3) Merge → write merged plan to S3 as ai_plan_merged.yaml and return as ai_plan_s3.
    """
    setup_mlflow(cfg["experiment_name"])

    if "data_card_s3" not in state:
        raise RuntimeError("data_card_s3 missing; run EDA first")

    # 1) Baseline rules plan (also writes to S3)
    rules_out = _load_rules_plan(cfg, state)
    rules_plan = rules_out["ai_plan"]
    rules_plan_s3 = rules_out["ai_plan_s3"]

    # 2) Load card & call LLM
    card = json.loads(_read_s3_text(state["data_card_s3"]))
    preview = _card_preview(card)

    # Build prompt
    system = (
        "You are an AutoML planning expert. Produce a complete YAML-equivalent JSON plan "
        "that my pipeline can use, given an EDA 'data card'. Follow the schema:\n"
        "{version:int, dataset:{n_rows:int,n_cols:int,target:str,task:str,positive_class?:any},"
        " features:{drop?:[str],imputation:{numeric:str,categorical:str,boolean?:str},"
        " encoding:{low_cardinality:{method:str},high_cardinality:{method:str},high_cardinality_threshold?:float},"
        " scaling?:{apply_to:str,when_models_include?:[str]},"
        " datetime?:{columns?:[str],derived?:[str]}, text?:{columns?:[str],strategy?:str},"
        " columns_by_type?:object},"
        " models:[{name:str,only_if?:bool,requires_scaling?:bool,fit_params?:object,hpo?:object,penalty?:str}],"
        " hpo:{n_trials:int,timeout_min:int,top_k_candidates:int},"
        " evaluation:{primary_metric:str,secondary_metrics?:[str],threshold_tuning?:{enabled:bool,constraint?:{min_recall?:float},optimize_for?:str}},"
        " split:{method:str,params:object} }"
    )
    user = json.dumps({
        "cfg_task_type": cfg.get("task_type"),
        "cfg_target": cfg.get("target"),
        "cfg_timestamp_column": cfg.get("timestamp_column"),
        "policies": {"metrics": cfg.get("metrics")},
        "data_card": preview
    }, indent=2)

    
    from orchestrator.ai_client import ClaudeBedrockClient
    client = ClaudeBedrockClient(region_name = os.getenv("AWS_DEFAULT_REGION"), inference_profile_id = os.getenv("BEDROCK_INFERENCE_PROFILE_ID"))
    
    # ==== retry loop around LLM call + JSON parse + plan validation ====
    max_retries = 3
    last_error: Exception | None = None
    merged = None
    llm_plan = {}
    rationale = "LLM plan generated."
    raw_resp = None

    for attempt in range(1, max_retries + 1):
        print(f"Calling Bedrock LLM for AI plan... (attempt {attempt}/{max_retries})")
        raw_resp = client.chat(system_prompt=system.strip(), user_prompt=user)

        try:
            llm_json = parse_llm_json(raw_resp)
        except json.JSONDecodeError as e:
            last_error = e
            print(
                f"Attempt {attempt} failed: JSON parse error from LLM: {e}. "
                + ("Retrying..." if attempt < max_retries else "No retries left.")
            )
            continue

        print("LLM returned JSON:", llm_json)

        if not isinstance(llm_json, dict):
            # Extremely defensive: treat anything non-dict as invalid
            last_error = TypeError(f"LLM JSON is not a dict, got {type(llm_json)}")
            print(
                f"Attempt {attempt} failed: {last_error}. "
                + ("Retrying..." if attempt < max_retries else "No retries left.")
            )
            continue

        if "plan" in llm_json and isinstance(llm_json["plan"], dict):
            llm_plan = llm_json["plan"]
            rationale = llm_json.get("rationale", "LLM plan generated.")
        else:
            # Assume the whole body is the plan, but it must be a dict
            if isinstance(llm_json, dict):
                llm_plan = llm_json
                rationale = llm_json.get("rationale", "LLM plan generated.")
            else:
                last_error = TypeError(
                    f"LLM response structure invalid; expected dict or {{'plan': dict}}, got {type(llm_json)}"
                )
                print(
                    f"Attempt {attempt} failed: {last_error}. "
                    + ("Retrying..." if attempt < max_retries else "No retries left.")
                )
                continue

        # Try merging and validating; if that fails, we retry as well
        try:
            merged_candidate = _deep_merge(rules_plan, llm_plan)
            _validate_plan(merged_candidate)
        except AssertionError as e:
            last_error = e
            print(
                f"Attempt {attempt} failed plan validation: {e}. "
                + ("Retrying..." if attempt < max_retries else "No retries left.")
            )
            continue
        except Exception as e:
            last_error = e
            print(
                f"Attempt {attempt} failed during merge/validation: {e}. "
                + ("Retrying..." if attempt < max_retries else "No retries left.")
            )
            continue

        # If we got here, this attempt succeeded
        merged = merged_candidate
        break

    # After retries, if still no valid merged plan, raise
    if merged is None:
        raise RuntimeError(
            f"Failed to obtain a valid AI plan from Claude after {max_retries} attempts. "
            f"Last error: {last_error}\n\nLast raw response:\n{raw_resp}"
        )

    # Persist to S3 (rules, llm, merged)
    bucket = cfg["s3"]["bucket"].split("s3://",1)[1]
    base   = f'{cfg["s3"]["prefix"].strip("/")}/plan'
    ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

     # Convert to YAML strings once
    rules_yaml = _yaml_to_str(rules_plan)
    llm_yaml = _yaml_to_str(llm_plan)
    merged_yaml = _yaml_to_str(merged)

    rules_uri = _put_s3_bytes(
        bucket,
        f"{base}/ai_plan_rules_{ts}.yaml",
        rules_yaml.encode("utf-8"),
        "text/yaml",
    )
    llm_uri = _put_s3_bytes(
        bucket,
        f"{base}/ai_plan_llm_{ts}.yaml",
        llm_yaml.encode("utf-8"),
        "text/yaml",
    )
    merged_uri = _put_s3_bytes(
        bucket,
        f"{base}/ai_plan_merged_{ts}.yaml",
        merged_yaml.encode("utf-8"),
        "text/yaml",
    )
    rat_uri = _put_s3_bytes(
        bucket,
        f"{base}/ai_rationale_llm_{ts}.md",
        rationale.encode("utf-8"),
        "text/markdown",
    )

    # Log to MLflow
    with mlflow.start_run(run_name="ai_plan"):
        mlflow.log_text(merged_yaml, "plan/ai_plan_merged.yaml")
        mlflow.log_text(rules_yaml, "plan/ai_plan_rules.yaml")
        mlflow.log_text(llm_yaml, "plan/ai_plan_llm.yaml")
        mlflow.log_text(rationale, "plan/ai_rationale_llm.md")

    # Return with backward-compatible keys (downstream reads ai_plan_s3)
    return {
        "ai_plan": merged,
        "ai_plan_s3": merged_uri,
        "ai_rationale_s3": rat_uri,
        "ai_plan_rules_s3": rules_uri,
        "ai_plan_llm_s3": llm_uri,
    }
