import os,json
from ruamel.yaml import YAML
import typer

from runner.steps import ingest, eda, features, model_search, hpo, evaluate, select_and_register as sr, report, plan, predict,ai_plan_llm, generate_base_config as ai_base_config

app = typer.Typer(no_args_is_help=True)
yaml = YAML(typ="safe")

def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

def load_cfg(path="configs/base.yaml"):
    with open(path) as f:
        cfg = yaml.load(f)
    return _expand_env(cfg)

def _load_state():
    try:
        return json.load(open(".state.json"))
    except FileNotFoundError:
        return {}

def _save_state(state):
    json.dump(state, open(".state.json","w"))

@app.command("ingest-step")
def ingest_step():
    cfg = load_cfg()
    state = _load_state()
    out = ingest.run_ingest(cfg)
    state.update(out); _save_state(state)
    print(json.dumps(state, indent=2))
    
@app.command("ai-base-step")
def ai_base_step():
    state = _load_state()
    cfg = load_cfg()
    out = ai_base_config.run_generate_base_config(cfg,state["dataset_uri"])
    state.update(out); _save_state(state)
    print(json.dumps(state, indent=2))

@app.command("eda-step")
def eda_step():
    cfg = load_cfg()
    state = _load_state()
    out = eda.run_eda(cfg, state["dataset_uri"])
    state.update(out); _save_state(state)
    print(json.dumps(state, indent=2))
    
@app.command("ai-plan-step") 
def ai_plan_step():
    cfg = load_cfg()
    state = _load_state()
    out = ai_plan_llm.run_ai_plan_llm(cfg, state)
    state.update(out); _save_state(state)
    print(json.dumps({"ai_plan_s3": out["ai_plan_s3"], "ai_rationale_s3": out["ai_rationale_s3"]}, indent=2))

@app.command("features-step")
def features_step():
    cfg = load_cfg()
    state = _load_state()
    out = features.run_features(cfg, state["dataset_uri"], state.get("ai_plan_s3")) 
    state.update(out); _save_state(state)
    print(json.dumps(state, indent=2))

@app.command("model-search-step")
def model_search_step():
    cfg = load_cfg()
    state = _load_state()
    out = model_search.run_model_search(cfg, state["features_base"])
    state["model_search"] = out; _save_state(state)
    print(json.dumps(out, indent=2))

@app.command("hpo-step")
def hpo_step():
    cfg = load_cfg()
    state = _load_state()
    top = [r["name"] for r in state["model_search"]]
    out = hpo.run_hpo(cfg, state["features_base"], top)
    state["hpo"] = out; _save_state(state)
    print(json.dumps(out, indent=2))

@app.command("evaluate-step")
def evaluate_step():
    cfg = load_cfg()
    state = _load_state()
    out = evaluate.run_evaluate(cfg, state["features_base"], state["hpo"])
    state["final_leaderboard"] = out; _save_state(state)
    print(json.dumps(out, indent=2))

@app.command("select-and-register-step")
def select_and_register_step(min_recall: float = 0.6):
    cfg = load_cfg()
    state = _load_state()
    chosen = sr.select_model(cfg, state["final_leaderboard"], min_recall=min_recall)
    reg = sr.register_selected(cfg, chosen)
    state["selected"] = chosen; state["registry"] = reg; _save_state(state)
    print(json.dumps({"selected": chosen, "registry": reg}, indent=2))

@app.command("report-step")
def report_step():
    cfg = load_cfg()
    state = _load_state()
    out = report.run_report(cfg, state)
    state.update(out); _save_state(state)
    print(json.dumps(out, indent=2))

@app.command("predict-step")
def predict_step(
    input_uri: str = typer.Argument(..., help="S3 path to CSV or Parquet to score, e.g., s3://bucket/path/file.parquet"),
    model_name: str = typer.Option(None, help="Registry model name (defaults to cfg.registry.name)"),
    model_version: str = typer.Option(None, help="Specific model version to load (defaults to latest numeric)"),
    output_prefix: str = typer.Option(None, help="S3 prefix to write outputs under (defaults to <cfg.s3.prefix>/predictions/<UTC>)."),
):
    cfg = load_cfg()
    state = _load_state()
    out = predict.run_predict(cfg, input_uri, model_name=model_name, model_version=model_version, output_prefix=output_prefix)
    state["last_predict"] = out; _save_state(state)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()

