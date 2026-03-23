import duckdb
import pandas as pd
import os
import json
from datetime import datetime

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DB_PATH = "data/computeguard.duckdb"
OUTPUT_PATH = "data/processed/features.csv"
QUALITY_REPORT_PATH = "data/processed/quality_report.json"

# ── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_raw_data(con):
    print("\n[1/4] Loading raw CSVs into DuckDB...")

    csv_files = [
        f for f in os.listdir(RAW_DIR) if f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"      Found {len(csv_files)} daily files")

    # Create table structure first
    con.execute("""
        CREATE OR REPLACE TABLE raw_drives (
            date DATE,
            serial_number VARCHAR,
            model VARCHAR,
            failure INTEGER,
            smart_5_raw BIGINT,
            smart_9_raw BIGINT,
            smart_187_raw BIGINT,
            smart_188_raw BIGINT,
            smart_197_raw BIGINT,
            smart_198_raw BIGINT
        )
    """)

    # Load files in batches to avoid memory issues
    batch_size = 10
    total_loaded = 0

    for i in range(0, len(csv_files), batch_size):
        batch_files = csv_files[i:i + batch_size]
        file_paths = [os.path.join(RAW_DIR, f) for f in batch_files]

        print(f"      Processing batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size} ({len(batch_files)} files)")

        # Use UNION ALL to combine multiple files
        union_queries = []
        for file_path in file_paths:
            union_queries.append(f"""
                SELECT
                    date::DATE                  AS date,
                    serial_number,
                    model,
                    CAST(failure AS INTEGER)    AS failure,
                    CAST(smart_5_raw   AS BIGINT) AS smart_5_raw,
                    CAST(smart_9_raw   AS BIGINT) AS smart_9_raw,
                    CAST(smart_187_raw AS BIGINT) AS smart_187_raw,
                    CAST(smart_188_raw AS BIGINT) AS smart_188_raw,
                    CAST(smart_197_raw AS BIGINT) AS smart_197_raw,
                    CAST(smart_198_raw AS BIGINT) AS smart_198_raw
                FROM read_csv_auto('{file_path}', ignore_errors=True)
                WHERE serial_number IS NOT NULL
            """)

        union_sql = " UNION ALL ".join(union_queries)

        con.execute(f"INSERT INTO raw_drives {union_sql}")

        batch_count = con.execute("SELECT COUNT(*) FROM raw_drives").fetchone()[0] - total_loaded
        total_loaded += batch_count
        print(f"        Loaded {batch_count:,} records (total: {total_loaded:,})")

    count = con.execute("SELECT COUNT(*) FROM raw_drives").fetchone()[0]
    print(f"      Total loaded: {count:,} records")
    return count

# ── 2. TRANSFORM ─────────────────────────────────────────────────────────────

def transform_features(con):
    print("\n[2/4] Engineering features...")

    con.execute("""
        CREATE OR REPLACE TABLE features AS
        WITH base AS (
            SELECT
                date,
                serial_number,
                model,
                failure,
                COALESCE(smart_5_raw,   0) AS smart_5_raw,
                COALESCE(smart_9_raw,   0) AS smart_9_raw,
                COALESCE(smart_187_raw, 0) AS smart_187_raw,
                COALESCE(smart_188_raw, 0) AS smart_188_raw,
                COALESCE(smart_197_raw, 0) AS smart_197_raw,
                COALESCE(smart_198_raw, 0) AS smart_198_raw
            FROM raw_drives
        ),
        with_lags AS (
            SELECT
                *,
                -- 7-day rolling averages (trend over last week)
                AVG(smart_5_raw)   OVER w7 AS smart_5_7d_avg,
                AVG(smart_187_raw) OVER w7 AS smart_187_7d_avg,
                AVG(smart_197_raw) OVER w7 AS smart_197_7d_avg,
                AVG(smart_198_raw) OVER w7 AS smart_198_7d_avg,

                -- 30-day rolling averages (longer trend)
                AVG(smart_5_raw)   OVER w30 AS smart_5_30d_avg,
                AVG(smart_187_raw) OVER w30 AS smart_187_30d_avg,

                -- Day-over-day changes (sudden spikes)
                smart_5_raw   - LAG(smart_5_raw,   1) OVER ws AS smart_5_delta,
                smart_197_raw - LAG(smart_197_raw, 1) OVER ws AS smart_197_delta,
                smart_198_raw - LAG(smart_198_raw, 1) OVER ws AS smart_198_delta,

                -- Drive age in days
                smart_9_raw / 24 AS drive_age_days,

                -- Failure in next 30 days (label for model)
                MAX(failure) OVER (
                    PARTITION BY serial_number
                    ORDER BY date
                    ROWS BETWEEN 1 FOLLOWING AND 30 FOLLOWING
                ) AS failure_in_30d

            FROM base
            WINDOW
                ws  AS (PARTITION BY serial_number ORDER BY date),
                w7  AS (PARTITION BY serial_number ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
                w30 AS (PARTITION BY serial_number ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
        )
        SELECT * FROM with_lags
        WHERE failure_in_30d IS NOT NULL
    """)

    count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    failures = con.execute(
        "SELECT SUM(failure_in_30d) FROM features"
    ).fetchone()[0]
    print(f"      Feature rows  : {count:,}")
    print(f"      Positive labels (failures in 30d): {int(failures):,}")
    print(f"      Failure rate  : {failures/count*100:.4f}%")

# ── 3. QUALITY CHECKS ────────────────────────────────────────────────────────

def run_quality_checks(con):
    print("\n[3/4] Running data quality checks...")

    checks = {}
    passed = 0
    failed = 0

    # Check 1 — Null rate
    null_counts = con.execute("""
        SELECT
            COUNT(*) - COUNT(smart_5_raw)   AS smart_5_nulls,
            COUNT(*) - COUNT(smart_187_raw) AS smart_187_nulls,
            COUNT(*) - COUNT(smart_197_raw) AS smart_197_nulls,
            COUNT(*)                        AS total
        FROM features
    """).fetchone()

    null_rate = (null_counts[0] + null_counts[1] + null_counts[2]) / (
        null_counts[3] * 3
    )
    check1_pass = null_rate < 0.05
    checks["null_rate"] = {
        "value": round(null_rate, 4),
        "threshold": 0.05,
        "passed": check1_pass,
        "message": f"Null rate {null_rate:.2%} {'< 5% OK' if check1_pass else '> 5% FAIL'}"
    }
    passed += check1_pass
    failed += not check1_pass
    print(f"      Check 1 — Null rate      : {'PASS' if check1_pass else 'FAIL'} ({null_rate:.2%})")

    # Check 2 — Row count
    row_count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    check2_pass = row_count > 100000
    checks["row_count"] = {
        "value": row_count,
        "threshold": 100000,
        "passed": check2_pass,
        "message": f"{row_count:,} rows {'OK' if check2_pass else 'too few'}"
    }
    passed += check2_pass
    failed += not check2_pass
    print(f"      Check 2 — Row count      : {'PASS' if check2_pass else 'FAIL'} ({row_count:,})")

    # Check 3 — Value range (smart_9 = drive age, must be positive)
    neg_age = con.execute(
        "SELECT COUNT(*) FROM features WHERE drive_age_days < 0"
    ).fetchone()[0]
    check3_pass = neg_age == 0
    checks["value_range"] = {
        "value": neg_age,
        "threshold": 0,
        "passed": check3_pass,
        "message": f"{neg_age} negative drive ages {'OK' if check3_pass else 'FAIL'}"
    }
    passed += check3_pass
    failed += not check3_pass
    print(f"      Check 3 — Value range    : {'PASS' if check3_pass else 'FAIL'} ({neg_age} negative ages)")

    # Check 4 — Label balance (failure rate must not be 0 or 100%)
    failure_rate = con.execute(
        "SELECT AVG(failure_in_30d) FROM features"
    ).fetchone()[0]
    check4_pass = 0.0001 < failure_rate < 0.5
    checks["label_balance"] = {
        "value": round(failure_rate, 6),
        "threshold": "0.0001 < x < 0.5",
        "passed": check4_pass,
        "message": f"Failure rate {failure_rate:.4%} {'OK' if check4_pass else 'FAIL'}"
    }
    passed += check4_pass
    failed += not check4_pass
    print(f"      Check 4 — Label balance  : {'PASS' if check4_pass else 'FAIL'} ({failure_rate:.4%})")

    # Check 5 — Schema check (all required columns exist)
    required_cols = [
        "date", "serial_number", "model", "failure",
        "smart_5_raw", "smart_9_raw", "smart_187_raw",
        "smart_197_raw", "smart_198_raw",
        "smart_5_7d_avg", "smart_5_30d_avg",
        "drive_age_days", "failure_in_30d"
    ]
    actual_cols = [
        row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='features'"
        ).fetchall()
    ]
    missing = [c for c in required_cols if c not in actual_cols]
    check5_pass = len(missing) == 0
    checks["schema"] = {
        "value": missing,
        "passed": check5_pass,
        "message": f"Missing columns: {missing if missing else 'None — OK'}"
    }
    passed += check5_pass
    failed += not check5_pass
    print(f"      Check 5 — Schema         : {'PASS' if check5_pass else 'FAIL'} (missing: {missing})")

    print(f"\n      Results: {passed}/5 passed, {failed}/5 failed")

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {"passed": passed, "failed": failed, "total": 5},
        "checks": checks
    }

    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"      Report saved: {QUALITY_REPORT_PATH}")

    if failed > 0:
        raise ValueError(f"Data quality failed: {failed} check(s) did not pass. See {QUALITY_REPORT_PATH}")

    return report

# ── 4. EXPORT ─────────────────────────────────────────────────────────────────

def export_features(con):
    print("\n[4/4] Exporting features to CSV...")
    df = con.execute("SELECT * FROM features").df()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved: {OUTPUT_PATH}")
    print(f"      Shape: {df.shape}")
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 55)
    print("  ComputeGuard — Data Pipeline")
    print("=" * 55)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    load_raw_data(con)
    transform_features(con)
    run_quality_checks(con)
    df = export_features(con)

    con.close()

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55)
    return df

if __name__ == "__main__":
    run_pipeline()