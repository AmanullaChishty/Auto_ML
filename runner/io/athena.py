import time
import boto3
import os
import pandas as pd
from dotenv import load_dotenv

def get_athena_client(region_name: str = "us-east-1") -> boto3.client:
    """
    Create an Athena client using credentials from environment variables.
    Assumes .env has AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_SESSION_TOKEN.
    """
    # Load .env into environment
    load_dotenv()

    # You can rely on boto3's default credential chain once env vars are set
    client = boto3.client(
        "athena",
        region_name=region_name,
    )
    return client

def fetch_df():
    database="arcsm_dev_inspec_claims_v1"
    output_location="s3://automl-poc-frailty-us-east-1/athena/query-results/"
    workgroup="frailty-poc"
    region=os.getenv('AWS_DEFAULT_REGION')
    sleep_time=2.0
    query = '''SELECT * FROM "arcsm_dev_inspec_claims_v1"."plan_member_assignment_rt" limit 10000;'''
    
    client = get_athena_client(region_name=region)
    
    params = {
        "QueryString": query,
        "QueryExecutionContext": {"Database": database},
        "ResultConfiguration": {"OutputLocation": output_location},
        "WorkGroup": workgroup,
    }
    
    if workgroup:
        params["WorkGroup"] = workgroup
    
    response = client.start_query_execution(**params)
    query_execution_id = response["QueryExecutionId"]
    
    while True:
        status_response = client.get_query_execution(QueryExecutionId=query_execution_id)
        status = status_response["QueryExecution"]["Status"]["State"]

        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break

        time.sleep(sleep_time)

    if status != "SUCCEEDED":
        reason = status_response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
        raise RuntimeError(f"Athena query {status}: {reason}")
    
    paginator = client.get_paginator("get_query_results")
    results_iter = paginator.paginate(QueryExecutionId=query_execution_id)
    
    columns = None
    rows = []

    for page in results_iter:
        for row in page["ResultSet"]["Rows"]:
            # Header row
            if columns is None:
                columns = [col.get("VarCharValue", "") for col in row["Data"]]
                continue

            # Data row
            values = [col.get("VarCharValue", None) for col in row["Data"]]
            rows.append(values)
            
    if columns is None:
        return pd.DataFrame()  # no data

    df = pd.DataFrame(rows, columns=columns)
    return df
    
    