import os, io, json, tempfile, boto3,re
from urllib.parse import urlparse
from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML
import mlflow

from runner.utils.mlflow_utils import setup_mlflow

yaml = YAML(typ="safe")

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) for s3://bucket/key"""
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

def bucket_region(s3_client, bucket: str) -> str:
    """Find bucket region; normalize legacy 'EU' to 'eu-west-1'."""
    resp = s3_client.get_bucket_location(Bucket=bucket)
    loc = resp.get("LocationConstraint")
    if not loc or loc == "":  # us-east-1 returns None/empty
        return "us-east-1"
    if loc == "EU":
        return "eu-west-1"
    return loc

def presign_s3_uri(s3_uri: str, expires: int = 360000) -> str:
    """Generate a pre-signed HTTPS URL for an s3://... object."""
    bucket, key = parse_s3_uri(s3_uri)
    # Discover region then create a regional client for signing
    probe = boto3.client("s3")
    region = bucket_region(probe, bucket)
    regional = boto3.client("s3", region_name=region)
    return regional.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def _read_s3_text(uri: str) -> str:
    p = urlparse(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _load_plan_from_state():
    try:
        state = json.load(open(".state.json"))
    except FileNotFoundError:
        return {}
    s3_uri = state.get("ai_plan_s3")
    if not s3_uri:
        return {}
    try:
        return yaml.load(_read_s3_text(s3_uri)) or {}
    except Exception:
        return {}

def _put_s3_bytes(s3, bucket: str, key: str, data: bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"

def _download_artifact_bytes(run_id: str, artifact_path: str) -> bytes | None:
    """Download an MLflow artifact into memory. Returns bytes or None if missing."""
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        with open(local_path, "rb") as f:
            return f.read()
    except Exception:
        return None

def run_report(cfg, state):
    setup_mlflow(cfg["experiment_name"])
    env = Environment(loader=FileSystemLoader("runner/templates"))
    report_tmpl = env.get_template("report.html.j2")
    card_tmpl   = env.get_template("model_card.html.j2")

    # MLflow / selected run
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(cfg["experiment_name"])
    sel_run_id = state["registry"]["run_id"]
    sel_name   = state["selected"]["name"]

    # Load the exact selected run
    run = client.get_run(sel_run_id)
    art_uri = run.info.artifact_uri

    # Pull threshold/constraint info from artifacts & plan/tags
    plan = _load_plan_from_state() or {}
    primary_metric = ((plan.get("evaluation") or {}).get("primary_metric")
                      or cfg.get("metrics", {}).get("primary", "auprc")).lower()
    tt_conf = ((plan.get("evaluation") or {}).get("threshold_tuning") or {})
    min_recall = float((tt_conf.get("constraint") or {}).get("min_recall", 0.6))

    # From evaluate step, we logged these files
    opp_path = f"test_operating_point_{sel_name}.json"
    opp_bytes = _download_artifact_bytes(sel_run_id, opp_path)
    used_threshold = 0.5
    violated = None
    cm_dict = None
    if opp_bytes:
        opp = json.loads(opp_bytes.decode("utf-8"))
        used_threshold = float(opp.get("used_threshold", 0.5))
        violated = opp.get("constraint_violated")
        cm_dict = opp.get("confusion_matrix")

    # Images: PR curve + Confusion matrix
    pr_png_bytes = _download_artifact_bytes(sel_run_id, f"test_prc_{sel_name}.png")
    cm_png_bytes = _download_artifact_bytes(sel_run_id, f"confusion_matrix_{sel_name}.png")

    # Prepare S3 paths (unique per selected run)
    s3 = boto3.client("s3")
    bucket = cfg["s3"]["bucket"].split("s3://",1)[1]
    base_key = f'{cfg["s3"]["prefix"].strip("/")}/reports/{sel_run_id}'

    # Upload images if present
    pr_png_uri = None
    cm_png_uri = None
    pr_png_url = None  # HTTPS (pre-signed)
    cm_png_url = None  # HTTPS (pre-signed)
    presign_ttl = int(cfg.get("s3", {}).get("presign_expires", 360000))

    if pr_png_bytes:
        pr_png_uri = _put_s3_bytes(
            s3, bucket, f"{base_key}/test_prc_{sel_name}.png", pr_png_bytes, content_type="image/png"
        )
        pr_png_url = presign_s3_uri(pr_png_uri, expires=presign_ttl)

    if cm_png_bytes:
        cm_png_uri = _put_s3_bytes(
            s3, bucket, f"{base_key}/confusion_matrix_{sel_name}.png", cm_png_bytes, content_type="image/png"
        )
        cm_png_url = presign_s3_uri(cm_png_uri, expires=presign_ttl)

    # Compose context objects
    leaderboard = state["final_leaderboard"]
    selected    = state["selected"]
    registry    = state["registry"]
    

    # Render report
    report_html = report_tmpl.render(
        project_name=cfg["project_name"],
        dataset_uri=state["dataset_uri"],
        target=cfg["target"],
        leaderboard=leaderboard,
        selected=selected,
        registry=registry,
        primary_metric=primary_metric,
        used_threshold=used_threshold,
        min_recall=min_recall,
        constraint_violated=violated,
        confusion=cm_dict,
        pr_png_uri=pr_png_url,
        cm_png_uri=cm_png_url,
        mlflow_run_id=sel_run_id,
        mlflow_experiment=exp.name,
    )

    # Render model card
    card_html = card_tmpl.render(
        project_name=cfg["project_name"],
        model_name=selected["name"],
        registry=registry,
        target=cfg["target"],
        primary_metric=primary_metric,
        metrics={
            "auprc": selected.get("auprc"),
            "auroc": selected.get("auroc"),
            "precision": selected.get("precision"),
            "recall": selected.get("recall"),
            "f1": selected.get("f1"),
        },
        used_threshold=used_threshold,
        min_recall=min_recall,
        constraint_violated=violated,
        confusion=cm_dict,
        pr_png_uri=pr_png_url,
        cm_png_uri=cm_png_url,
        mlflow_run_id=sel_run_id,
        mlflow_experiment=exp.name,
        selection_meta=selected.get("selection", {}),
    )

    # Write to S3
    report_uri = _put_s3_bytes(s3, bucket, f"{base_key}/index.html", report_html.encode("utf-8"), content_type="text/html")
    card_uri   = _put_s3_bytes(s3, bucket, f"{base_key}/model_card.html", card_html.encode("utf-8"), content_type="text/html")
    report_url = presign_s3_uri(report_uri, expires=presign_ttl)
    card_url   = presign_s3_uri(card_uri,   expires=presign_ttl)

    # Also log report to its own MLflow run
    with mlflow.start_run(run_name="report"):
        mlflow.log_text(report_html, "report.html")
        mlflow.log_text(card_html, "model_card.html")

    return {
        "report_s3": report_uri,
        "model_card_s3": card_uri,
        "report_url": report_url,           
        "model_card_url": card_url,         
        "pr_png_url": pr_png_url,           
        "cm_png_url": cm_png_url,           
    }
