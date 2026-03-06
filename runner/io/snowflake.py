import os, pandas as pd
import snowflake.connector

def fetch_df():
    ctx = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PAT"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        disable_ocsp_checks=True,
        ocsp_fail_open=True
    )
    sql = """
    WITH header_p AS (
  SELECT
    PERSON_ID,
    CLAIM_HEADER_ID,
    UPPER(CLAIM_TYPE) AS CLAIM_TYPE,
    DRG,
    LPAD(CAST(BILL_TYPE AS VARCHAR), 3, '0') AS BILL_TYPE3,
    TRY_TO_DATE(SERVICE_FROM_DATE) AS svc_date
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_HEADER
  WHERE SERVICE_FROM_DATE IS NOT NULL
  LIMIT 1000000
),
months AS (
  /* enumerate (person, index_month, index_date) from actual header dates for members */
  SELECT
    m.PERSON_ID,
    DATE_TRUNC('MONTH', h.svc_date)::DATE AS index_month,
    LAST_DAY(h.svc_date)::DATE          AS index_date
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_MEMBER m
  JOIN header_p h
    ON h.PERSON_ID = m.PERSON_ID
  GROUP BY 1,2,3
  LIMIT 1000000
),
line_flags AS (
  SELECT
    CLAIM_HEADER_ID,
    MAX(CASE WHEN REVENUE_CODE LIKE '045%' OR REVENUE_CODE = '0981' THEN 1 ELSE 0 END) AS has_er_revenue
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_LINE
  GROUP BY 1
  LIMIT 1000000
),
header_flags AS (
  /* one row per header with an acute flag */
  SELECT
    h.CLAIM_HEADER_ID,
    h.PERSON_ID,
    h.svc_date,
    IFF(
      (h.CLAIM_TYPE = 'INST' AND (h.DRG IS NOT NULL OR h.BILL_TYPE3 LIKE '11%' OR h.BILL_TYPE3 LIKE '12%'))
      OR NVL(l.has_er_revenue, 0) = 1,
      1, 0
    ) AS is_acute
  FROM header_p h
  LEFT JOIN line_flags l
    ON l.CLAIM_HEADER_ID = h.CLAIM_HEADER_ID
  LIMIT 1000000
),
acute_future AS (
  SELECT
    mm.PERSON_ID,
    mm.index_month,
    MAX(is_acute) AS y_acute_90
  FROM months mm
  JOIN header_flags hf
    ON hf.PERSON_ID = mm.PERSON_ID
   AND hf.svc_date > mm.index_date
   AND hf.svc_date <= DATEADD(day, 90, mm.index_date)
  GROUP BY 1,2
  LIMIT 1000000
),

ed_visits_180 AS (
SELECT
  mm.PERSON_ID, mm.index_month,
  COUNT(DISTINCT h.CLAIM_HEADER_ID) AS ed_visits_180d
FROM months mm
JOIN LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_HEADER h
  ON h.PERSON_ID=mm.PERSON_ID
 AND h.SERVICE_FROM_DATE BETWEEN mm.index_date - INTERVAL '180 day' AND mm.index_date
WHERE EXISTS (
  SELECT 1 FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_LINE l
  WHERE l.CLAIM_HEADER_ID=h.CLAIM_HEADER_ID
    AND (l.REVENUE_CODE LIKE '045%' OR l.REVENUE_CODE='0981')
)
GROUP BY 1,2
LIMIT 1000000
),

params AS (
  SELECT TO_DATE('2025-06-30') AS idx
),
fills AS (
  SELECT
      ph.PERSON_ID,
      GREATEST(
        COALESCE(TRY_TO_DATE(ph.FILL_DATE), TRY_TO_TIMESTAMP_NTZ(ph.FILL_DATE)::DATE),
        DATEADD(day, -180, p.idx)
      ) AS start_,
      LEAST(
        DATEADD(
          day, TRY_TO_NUMBER(ph.DAYS_SUPPLY),
          COALESCE(TRY_TO_DATE(ph.FILL_DATE), TRY_TO_TIMESTAMP_NTZ(ph.FILL_DATE)::DATE)
        ),
        p.idx
      ) AS stop_
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_PHARMACY ph
  CROSS JOIN params p
  WHERE ph.DRUG_NAME ILIKE '%STATIN%'
    AND TRY_TO_NUMBER(ph.DAYS_SUPPLY) IS NOT NULL
    AND COALESCE(TRY_TO_DATE(ph.FILL_DATE), TRY_TO_TIMESTAMP_NTZ(ph.FILL_DATE)::DATE)
        BETWEEN DATEADD(day, -180, p.idx) AND p.idx
),
span AS (
  SELECT
    PERSON_ID,
    /* clamp negatives; NVL guards against any remaining NULL spans */
    NVL(GREATEST(0, DATEDIFF('day', start_, stop_)), 0) AS covered_days
  FROM fills
), 
pdc_statin_180 as (
SELECT
  PERSON_ID,
  SUM(covered_days) / 180.0 AS pdc_statin_180d
FROM span
GROUP BY PERSON_ID
LIMIT 100000
),

/* ------- NEW: build base off months so INDEX_MONTH is never NULL ------- */
FINAL_ACUTE_ED_PDC AS (
  SELECT
    mm.PERSON_ID,
    mm.INDEX_MONTH,
    NVL(ed.ED_VISITS_180D, 0)      AS ED_VISITS_180D,
    NVL(pdc.PDC_STATIN_180D, 0.0)  AS PDC_STATIN_180D,
    NVL(af.Y_ACUTE_90, 0)          AS Y_ACUTE_90
  FROM months mm
  LEFT JOIN ed_visits_180 ed
    ON ed.PERSON_ID = mm.PERSON_ID AND ed.index_month = mm.index_month
  LEFT JOIN acute_future af
    ON af.PERSON_ID = mm.PERSON_ID AND af.index_month = mm.index_month
  LEFT JOIN pdc_statin_180 pdc       -- person-level; repeats across months
    ON pdc.PERSON_ID = mm.PERSON_ID
),

header_full AS (
  SELECT
    PERSON_ID,
    CLAIM_HEADER_ID,
    UPPER(CLAIM_TYPE)                          AS CLAIM_TYPE,
    TRY_TO_DATE(SERVICE_FROM_DATE)             AS svc_from_dt,
    TRY_TO_DATE(SERVICE_THRU_DATE)             AS svc_thru_dt,
    LPAD(CAST(BILL_TYPE AS VARCHAR),3,'0')     AS BILL_TYPE3,
    DRG, ADMISSION_TYPE, ATTENDING_PROVIDER_ID,
    TRY_TO_NUMBER(ALLOWED_AMOUNT)              AS allowed_amt,
    TRY_TO_NUMBER(PAID_AMOUNT)                 AS paid_amt,
    TRY_TO_NUMBER(COPAY_AMOUNT)                AS copay_amt,
    TRY_TO_NUMBER(COINSURANCE_AMOUNT)          AS coins_amt,
    TRY_TO_NUMBER(DEDUCTIBLE_AMOUNT)           AS ded_amt
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_HEADER
  WHERE SERVICE_FROM_DATE IS NOT NULL
  LIMIT 100000
),

/* reuse your line_flags for ED; build an acute header flag for lookbacks */
is_acute_header AS (
  SELECT
    h.CLAIM_HEADER_ID, h.PERSON_ID, h.svc_from_dt,
    IFF(
      (h.CLAIM_TYPE = 'INST' AND (h.DRG IS NOT NULL OR h.BILL_TYPE3 LIKE '11%' OR h.BILL_TYPE3 LIKE '12%'))
      OR NVL(l.has_er_revenue,0)=1,
      1, 0
    ) AS is_acute
  FROM header_full h
  LEFT JOIN line_flags l ON l.CLAIM_HEADER_ID = h.CLAIM_HEADER_ID
  LIMIT 100000
),

/* 1) Acute IP count in 180d */
ip_acute_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         SUM(IFF(a.is_acute=1,1,0)) AS ip_acute_180d
  FROM months mm
  JOIN is_acute_header a
    ON a.PERSON_ID = mm.PERSON_ID
   AND a.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* 2) Length of stay days in 180d */
los_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         SUM( GREATEST(1, DATEDIFF(day, h.svc_from_dt, COALESCE(h.svc_thru_dt,h.svc_from_dt)) + 1) ) AS los_days_180d
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* 3) Out-of-pocket & 4) Allowed in 180d */
oop_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         SUM(NVL(h.copay_amt,0)+NVL(h.coins_amt,0)+NVL(h.ded_amt,0)) AS oop_180d
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),
allowed_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         SUM(NVL(h.allowed_amt,0)) AS allowed_180d
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* 5) Ambulatory (non-ED, non-acute IP) visits in 180d */
ambulatory_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         COUNT(DISTINCT h.CLAIM_HEADER_ID) AS ambulatory_visits_180d
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
  LEFT JOIN line_flags l ON l.CLAIM_HEADER_ID = h.CLAIM_HEADER_ID
  WHERE h.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
    AND NVL(l.has_er_revenue,0)=0
    AND NOT (h.CLAIM_TYPE='INST' AND (h.DRG IS NOT NULL OR h.BILL_TYPE3 LIKE '11%' OR h.BILL_TYPE3 LIKE '12%'))
  GROUP BY 1,2
  LIMIT 100000
),

/* 6–7) Diagnosis volume & breadth (12m) */
diag_join AS (
  SELECT d.PERSON_ID, d.CLAIM_HEADER_ID,
         UPPER(d.DIAGNOSIS_CODE) AS dx,
         h.svc_from_dt
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_DIAGNOSIS d
  JOIN header_full h ON h.CLAIM_HEADER_ID = d.CLAIM_HEADER_ID
  LIMIT 100000
),
dx_12m AS (
  SELECT mm.PERSON_ID, mm.index_month,
         COUNT(*)                                  AS dx_count_12m,
         COUNT(DISTINCT SUBSTR(dx,1,3))           AS dx_unique3_12m
  FROM months mm
  JOIN diag_join dj
    ON dj.PERSON_ID = mm.PERSON_ID
   AND dj.svc_from_dt BETWEEN DATEADD(month,-12, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* 8–10) Pharmacy volume, polypharmacy, prescribers; 14) trend */
rx_cast AS (
  SELECT
    PERSON_ID,
    COALESCE(TRY_TO_DATE(FILL_DATE), TRY_TO_TIMESTAMP_NTZ(FILL_DATE)::DATE) AS fill_dt,
    NDC, PRESCRIBER_ID,
    TRY_TO_NUMBER(DAYS_SUPPLY) AS days_supply,
    TRY_TO_NUMBER(PAID_AMOUNT) AS rx_paid,
    DRUG_NAME
  FROM LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.CLAIMS_ONLY_PHARMACY
  WHERE FILL_DATE IS NOT NULL
  LIMIT 100000
),
rx_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         COUNT(*)                   AS fills_180d,
         COUNT(DISTINCT NDC)        AS rx_ndc_unique_180d,
         COUNT(DISTINCT PRESCRIBER_ID) AS rx_prescribers_180d,
         SUM(NVL(rx_paid,0))        AS rx_paid_180d
  FROM months mm
  JOIN rx_cast r
    ON r.PERSON_ID = mm.PERSON_ID
   AND r.fill_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* 11) Statin exposure & recency */
statin_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         COUNT(*) AS statin_fills_180d,
         DATEDIFF(day, MAX(r.fill_dt), MAX(mm.index_date)) AS days_since_last_statin
  FROM months mm
  JOIN rx_cast r
    ON r.PERSON_ID = mm.PERSON_ID
   AND r.fill_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
   AND r.DRUG_NAME ILIKE '%STATIN%'
  GROUP BY 1,2
  LIMIT 100000
),

/* 12) Care recency */
recency AS (
  SELECT mm.PERSON_ID, mm.index_month,
         DATEDIFF(day, MAX(h.svc_from_dt), MAX(mm.index_date)) AS days_since_last_claim
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt < mm.index_date
  GROUP BY 1,2
  LIMIT 100000
),

/* 13) Provider contact & specialty spread (last 180d) */
provider_180 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         COUNT(DISTINCT h.ATTENDING_PROVIDER_ID)                 AS providers_180d,
         COUNT(DISTINCT COALESCE(pp.SPECIALTY,'UNK'))            AS provider_specialties_180d
  FROM months mm
  JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt BETWEEN DATEADD(day,-180, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  LEFT JOIN LIFE_SCIENCES_INSTALLATION2.LIFE_SCIENCES_INSTALLATION2_20251027.PLAN_PROVIDER pp
    ON pp.PROVIDER_ID = h.ATTENDING_PROVIDER_ID
  GROUP BY 1,2
  LIMIT 100000
),

/* 14) Short-term utilization trend (30d vs prior 30d) */
claim_counts_60 AS (
  SELECT mm.PERSON_ID, mm.index_month,
         SUM(IFF(h.svc_from_dt BETWEEN DATEADD(day,-30, mm.index_date) AND DATEADD(day,-1, mm.index_date),1,0))  AS claims_30d,
         SUM(IFF(h.svc_from_dt BETWEEN DATEADD(day,-60, mm.index_date) AND DATEADD(day,-31, mm.index_date),1,0)) AS claims_prev30d
  FROM months mm
  LEFT JOIN header_full h
    ON h.PERSON_ID = mm.PERSON_ID
   AND h.svc_from_dt BETWEEN DATEADD(day,-60, mm.index_date) AND DATEADD(day,-1, mm.index_date)
  GROUP BY 1,2
  LIMIT 100000
),

/* ---- Join onto the (now) month-anchored FINAL_ACUTE_ED_PDC ---- */
FINAL_FEATURES AS (
  SELECT
    base.PERSON_ID,
    base.INDEX_MONTH,
    /* existing */
    NVL(base.ED_VISITS_180D,        0)   AS ED_VISITS_180D,
    NVL(base.PDC_STATIN_180D,       0.0) AS PDC_STATIN_180D,
    NVL(base.Y_ACUTE_90,            0)   AS Y_ACUTE_90,
    /* new features */
    NVL(ip.ip_acute_180d,           0)   AS ip_acute_180d,
    NVL(los.los_days_180d,          0)   AS los_days_180d,
    NVL(oo.oop_180d,                0)   AS oop_180d,
    NVL(alw.allowed_180d,           0)   AS allowed_180d,
    NVL(amb.ambulatory_visits_180d, 0)   AS ambulatory_visits_180d,
    NVL(dx.dx_count_12m,            0)   AS dx_count_12m,
    NVL(dx.dx_unique3_12m,          0)   AS dx_unique3_12m,
    NVL(rx.fills_180d,              0)   AS fills_180d,
    NVL(rx.rx_ndc_unique_180d,      0)   AS rx_ndc_unique_180d,
    NVL(rx.rx_prescribers_180d,     0)   AS rx_prescribers_180d,
    NVL(rx.rx_paid_180d,            0)   AS rx_paid_180d,
    NVL(st.statin_fills_180d,       0)   AS statin_fills_180d,
    NVL(st.days_since_last_statin,  9999) AS days_since_last_statin,
    NVL(rc.days_since_last_claim,   9999) AS days_since_last_claim,
    NVL(pr.providers_180d,          0)   AS providers_180d,
    NVL(pr.provider_specialties_180d,0)  AS provider_specialties_180d,
    NVL(cc.claims_30d,              0)   AS claims_30d,
    NVL(cc.claims_prev30d,          0)   AS claims_prev30d
  FROM FINAL_ACUTE_ED_PDC base
  LEFT JOIN ip_acute_180   ip ON ip.PERSON_ID=base.PERSON_ID AND ip.index_month=base.index_month
  LEFT JOIN los_180        los ON los.PERSON_ID=base.PERSON_ID AND los.index_month=base.index_month
  LEFT JOIN oop_180        oo  ON oo.PERSON_ID =base.PERSON_ID AND oo.index_month =base.index_month
  LEFT JOIN allowed_180    alw ON alw.PERSON_ID=base.PERSON_ID AND alw.index_month=base.index_month
  LEFT JOIN ambulatory_180 amb ON amb.PERSON_ID=base.PERSON_ID AND amb.index_month=base.index_month
  LEFT JOIN dx_12m         dx  ON dx.PERSON_ID =base.PERSON_ID AND dx.index_month =base.index_month
  LEFT JOIN rx_180         rx  ON rx.PERSON_ID =base.PERSON_ID AND rx.index_month =base.index_month
  LEFT JOIN statin_180     st  ON st.PERSON_ID =base.PERSON_ID AND st.index_month =base.index_month
  LEFT JOIN recency        rc  ON rc.PERSON_ID =base.PERSON_ID AND rc.index_month =base.index_month
  LEFT JOIN provider_180   pr  ON pr.PERSON_ID =base.PERSON_ID AND pr.index_month =base.index_month
  LEFT JOIN claim_counts_60 cc ON cc.PERSON_ID =base.PERSON_ID AND cc.index_month =base.index_month
),
pos AS (
  SELECT f.*, ROW_NUMBER() OVER (ORDER BY HASH(f.person_id, f.index_month)) AS rn
  FROM FINAL_FEATURES f
  WHERE f.Y_ACUTE_90 = 1
),
neg AS (
  SELECT f.*, ROW_NUMBER() OVER (ORDER BY HASH(f.person_id, f.index_month)) AS rn
  FROM FINAL_FEATURES f
  WHERE f.Y_ACUTE_90 = 0
),
pos_cnt AS ( SELECT COUNT(*) AS c FROM FINAL_FEATURES WHERE Y_ACUTE_90 = 1 ),
take_pos AS (
  SELECT p.*
  FROM pos p
  CROSS JOIN pos_cnt pc
  WHERE p.rn <= LEAST(100, pc.c)
),
take_neg AS (
  SELECT n.*
  FROM neg n
  CROSS JOIN pos_cnt pc
  WHERE n.rn <= CASE WHEN pc.c >= 100 THEN 100 ELSE 200 - pc.c END
),
stratified_raw AS (
  SELECT * FROM take_pos
  UNION ALL
  SELECT * FROM take_neg
)

SELECT
  PERSON_ID,
  INDEX_MONTH,
  ED_VISITS_180D,
  PDC_STATIN_180D,
  Y_ACUTE_90,
  ip_acute_180d,
  los_days_180d,
  oop_180d,
  allowed_180d,
  ambulatory_visits_180d,
  dx_count_12m,
  dx_unique3_12m,
  fills_180d,
  rx_ndc_unique_180d,
  rx_prescribers_180d,
  rx_paid_180d,
  statin_fills_180d,
  days_since_last_statin,
  days_since_last_claim,
  providers_180d,
  provider_specialties_180d,
  claims_30d,
  claims_prev30d
FROM stratified_raw
ORDER BY INDEX_MONTH DESC, PERSON_ID limit 200;"""
    cur = ctx.cursor()
    try:
        cur.execute(sql)
        df = cur.fetch_pandas_all()
    finally:
        cur.close(); ctx.close()
    return df
