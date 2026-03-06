import os, pandas as pd
import redshift_connector

def fetch_df():
    conn = redshift_connector.connect(
        iam=True,
        host=os.getenv("REDSHIFT_HOST"),
        port=int(os.getenv("REDSHIFT_PORT","5439")),
        database=os.getenv("REDSHIFT_DB"),
        db_user=os.getenv("REDSHIFT_USER"),
        cluster_identifier=os.getenv("CLUSTER_IDENTIFIER"),
        region=os.getenv('AWS_DEFAULT_REGION')
    )
    sql = os.getenv("REDSHIFT_SQL")
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)
