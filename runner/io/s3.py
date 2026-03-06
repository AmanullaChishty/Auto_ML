import os, io, pandas as pd, boto3, datetime
from pyarrow import Table as ArrowTable
import pyarrow.parquet as pq

def write_parquet_df(df: pd.DataFrame, base_prefix: str):
    s3 = boto3.client("s3")
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bucket = os.environ["S3_BUCKET"].split("s3://",1)[1]
    prefix = f"{base_prefix}/datasets/{ts}"
    buf = io.BytesIO()
    table = ArrowTable.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/data.parquet", Body=buf.getvalue())
    s3.put_object(Bucket=bucket, Key=f"{prefix}/_SUCCESS", Body=b"")
    return f"s3://{bucket}/{prefix}/data.parquet"
