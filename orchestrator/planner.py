import json

POLICY = {
  "min_recall": 0.6
}

def next_action(state):
    if "dataset_uri" not in state: return "ingest"
    if "generated_base_config" not in state: return "ai"
    if "eda_s3" not in state: return "eda"
    if "ai_plan_s3" not in state: return "plan"          # NEW
    if "features_base" not in state: return "features"
    if "model_search" not in state: return "model_search"
    if "hpo" not in state: return "hpo"
    if "final_leaderboard" not in state: return "evaluate"
    if "selected" not in state or "registry" not in state: return "select_and_register"
    if "report_s3" not in state: return "report"
    return "done"
