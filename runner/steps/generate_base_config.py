# runner/steps/generate_base_config.py

import os, mlflow, json, yaml, boto3, io
import pandas as pd
from typing import Optional

from orchestrator.ai_client import ClaudeBedrockClient
from urllib.parse import urlparse
from runner.utils.data_profile import profile_dataframe
from runner.utils.mlflow_utils import setup_mlflow



def _read_parquet_s3(uri):
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def run_generate_base_config( cfg, dataset_uri: str, target: Optional[str] = None,):
    """
    Generate configs/base.yaml by calling an LLM (Claude via Bedrock),
    using the detailed EDA data_card.json stored in S3.

    The EDA step is assumed to have already produced data_card.json and
    stored its S3 URI in a state file, or you can pass the URI explicitly.
    """
    setup_mlflow(cfg["experiment_name"])
    with mlflow.start_run(run_name="ai-base-config"):
        print("Loading data from S3...")
        df = _read_parquet_s3(dataset_uri)
        project_name = cfg["project_name"]
        experiment_name = cfg["experiment_name" ]   

        schema_summary = profile_dataframe(df)
        mlflow.log_dict(schema_summary, "schema_summary.json")
        # 3. Build prompt for the LLM

        # System prompt: describe role & expectations
        system_prompt = """
            You are an expert ML engineer and AutoML configuration assistant.
            Given:
            - A detailed EDA profile (data card) of a dataset
            - Project and experiment metadata

            You will generate a complete YAML configuration file called base.yaml
            for our ML pipeline.
            """

        # User prompt: we reference their sample structure, but we do not box the model
        user_prompt_parts = [
            "We have dataset schema and basic stats.",
            "Here is the JSON representation of that data:",
            "",
            json.dumps(schema_summary, indent=2),
            "",
            "Using this information, generate a detailed YAML configuration named `base.yaml`.",
            "",
            "High-level expectations:",
            "- The YAML should be valid and parseable (no markdown fences).",
            "- Use comments (# ...) when you make assumptions.",
            "- Incorporate reasonable defaults based on the data distribution and problem characteristics.",
            "",
            "Configuration guidance:",
            f"- project_name should be: {project_name}",
            f"- experiment_name should be: {experiment_name}",
        ]

        if target:
            user_prompt_parts.append(f"- The target column (label) is: {target}")
        else:
            user_prompt_parts.append(
                "- If possible, infer the most likely target column from the data card and document your reasoning in comments."
            )

        user_prompt_parts.extend(
            [
                "",
                "You have flexibility to decide:",
                "- The task_type (e.g., classification, regression, survival, ranking), based on the semantics of the target and data.",
                "- The most appropriate primary evaluation metric and secondary metrics.",
                "- Any additional configuration fields relevant for this project (e.g., split strategy, positive_label, id/timestamp columns, HPO setup, EDA settings, etc.).",
                "",
                "However, you must:",
                "- Use valid YAML.",
                "- Prefer field names compatible with a typical AutoML / ML orchestration pipeline, for example:",
                "  project_name, experiment_name, task_type, target, id_column, timestamp_column, positive_label, split, metrics, hpo, eda.",
                "- Not wrap the YAML in triple backticks or any other markup.",
                "",
                "Return ONLY the YAML.",
            ]
        )

        user_prompt = "\n".join(user_prompt_parts)
        mlflow.log_text(user_prompt,'user_prompt.txt')

        client = ClaudeBedrockClient(region_name = os.getenv("AWS_DEFAULT_REGION"), inference_profile_id = os.getenv("BEDROCK_INFERENCE_PROFILE_ID"))

        # ==== retry loop for YAML generation/parse ====
        max_retries = 3
        last_error = None
        parsed = None
        yaml_text = None

        for attempt in range(1, max_retries + 1):
            print(f"Calling Bedrock LLM to generate base.yaml... (attempt {attempt}/{max_retries})")
            yaml_text = client.chat(
                system_prompt=system_prompt.strip(),
                user_prompt=user_prompt,
            )
            mlflow.log_text(yaml_text, f'yaml_text_attempt_{attempt}.txt')

            print("Validating returned YAML...")
            try:
                parsed = yaml.safe_load(yaml_text)
            except yaml.YAMLError as e:
                last_error = e
                print(
                    f"Attempt {attempt} failed: Claude returned invalid YAML: {e}. "
                    "Retrying..." if attempt < max_retries else "No retries left."
                )
                continue  # try again

            # If we parsed successfully, break out of retry loop
            if parsed is not None and isinstance(parsed, dict):
                break
            else:
                last_error = RuntimeError(
                    f"Parsed YAML is not a mapping/dict. Got: {type(parsed)}\nRaw text:\n{yaml_text}"
                )
                print(
                    f"Attempt {attempt} failed: parsed YAML not a dict. "
                    f"Type was {type(parsed)}. "
                    "Retrying..." if attempt < max_retries else "No retries left."
                )
                parsed = None  # reset and retry

        # After retries, if still no valid parsed dict, raise
        if parsed is None or not isinstance(parsed, dict):
            # If last_error is from YAML parsing, wrap it in the original-style message
            if isinstance(last_error, yaml.YAMLError):
                raise RuntimeError(
                    f"Claude returned invalid YAML after {max_retries} attempts: {last_error}\n\n"
                    f"Last raw text from model:\n{yaml_text}"
                )
            else:
                # Covers the "not a dict" case
                raise RuntimeError(
                    f"Failed to get valid dict-like YAML from Claude after {max_retries} attempts. "
                    f"Last error: {last_error}"
                )
        # ==== END NEW RETRY LOGIC ====
            
        output_path = "configs/base.yaml"
        # 6. Write YAML to output_path (e.g., configs/base.yaml)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        first_five_lines = []
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                lines = f.readlines()
                first_five_lines = lines[:5]
        
        yaml_content = yaml.safe_dump(parsed, sort_keys=False)     
        with open(output_path, "w") as f:
            if first_five_lines:
                f.writelines(first_five_lines)
                # Ensure there's exactly one newline before YAML content
                if not first_five_lines[-1].endswith("\n"):
                    f.write("\n")
            f.write(yaml_content)

        print(f"Generated base config at: {output_path}")
        return {"generated_base_config": output_path}

