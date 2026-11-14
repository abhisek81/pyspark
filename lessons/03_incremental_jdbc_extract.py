"""
Lesson 03 - Incremental JDBC extract (Postgres) using a high-watermark column

What you'll learn
  * Reading a database table with spark.read.format("jdbc")
  * Pushing a filter down to the database with a subquery in "dbtable"/"query"
  * Parallel JDBC reads: partitionColumn, lowerBound, upperBound, numPartitions
  * The high-watermark pattern for incremental (delta) loads
  * Persisting the watermark safely so re-runs don't lose or duplicate data

Key concepts (plain English)
  * Full loads copy the whole table every run - simple but slow and expensive.
  * Incremental loads copy only rows changed since last time. You need a column
    that always increases when a row changes, usually updated_at.
  * The "watermark" is the max(updated_at) you successfully loaded last time.
    Next run: WHERE updated_at > watermark. Save the new watermark ONLY after
    the write succeeds, otherwise a failure would skip data forever.
  * Overlap trick: subtract a few minutes from the watermark to catch rows
    committed late by long transactions, then de-duplicate downstream.

Real-world use case
  Copy new/changed rows from an OLTP Postgres "orders" table into the data lake
  every hour without hammering the production database.

Run
  # Demo mode (no database needed): simulates the source with local data
  python lessons/03_incremental_jdbc_extract.py
  # Real Postgres:
  #   export PG_URL=jdbc:postgresql://<host>:5432/<db> PG_USER=... PG_PASSWORD=...
  #   export PG_TABLE=public.orders USE_JDBC=1
  #   spark-submit --packages org.postgresql:postgresql:42.7.3 \
  #       lessons/03_incremental_jdbc_extract.py
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

USE_JDBC = os.getenv("USE_JDBC") == "1"
OVERLAP = timedelta(minutes=5)
DEFAULT_WATERMARK = "1970-01-01 00:00:00"


def build_spark() -> SparkSession:
    return (SparkSession.builder.appName("lesson03-incremental-jdbc")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


# --------------------------------------------------------------------------
# Watermark state. In production this lives in a control table, DynamoDB,
# or a small JSON file in object storage - never only in memory.
# --------------------------------------------------------------------------
def read_watermark(state_path: str) -> str:
    if not os.path.exists(state_path):
        return DEFAULT_WATERMARK
    with open(state_path) as f:
        return json.load(f)["last_updated_at"]


def save_watermark(state_path: str, value: str) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_updated_at": value}, f)
    os.replace(tmp, state_path)        # atomic rename: no half-written state


# --------------------------------------------------------------------------
# Source readers
# --------------------------------------------------------------------------
def jdbc_bounds(spark: SparkSession, since: str) -> tuple[int, int]:
    """Ask the DB for the id range of changed rows so we can split the read."""
    q = (f"(SELECT COALESCE(MIN(order_id),0) AS lo, COALESCE(MAX(order_id),0) AS hi "
         f"FROM {os.environ['PG_TABLE']} WHERE updated_at > '{since}') b")
    row = (spark.read.format("jdbc").option("url", os.environ["PG_URL"])
           .option("user", os.environ["PG_USER"])
           .option("password", os.environ["PG_PASSWORD"])
           .option("dbtable", q).load().first())
    return int(row["lo"]), int(row["hi"])


def read_source_jdbc(spark: SparkSession, since: str) -> DataFrame:
    lo, hi = jdbc_bounds(spark, since)
    # The WHERE clause runs INSIDE Postgres - only changed rows cross the network.
    subquery = (f"(SELECT * FROM {os.environ['PG_TABLE']} "
                f"WHERE updated_at > '{since}') src")
    return (
        spark.read.format("jdbc")
        .option("url", os.environ["PG_URL"])
        .option("user", os.environ["PG_USER"])
        .option("password", os.environ["PG_PASSWORD"])
        .option("driver", "org.postgresql.Driver")
        .option("dbtable", subquery)
        # Parallelism: Spark issues numPartitions queries, each covering a
        # slice of order_id between the bounds. Bounds only decide the
        # slicing; they do NOT filter rows.
        .option("partitionColumn", "order_id")
        .option("lowerBound", lo)
        .option("upperBound", hi + 1)
        .option("numPartitions", 4)
        .option("fetchsize", 10000)          # rows per round trip
        .load()
    )


_DEMO_ROWS = [
    (1, "C01", 100.0, "2026-11-01 10:00:00"),
    (2, "C02", 250.0, "2026-11-01 11:30:00"),
    (3, "C03", 80.0, "2026-11-02 09:15:00"),
    (2, "C02", 275.0, "2026-11-02 12:00:00"),   # order 2 updated later
    (4, "C04", 40.0, "2026-11-03 08:00:00"),
]


def read_source_demo(spark: SparkSession, since: str, up_to: str) -> DataFrame:
    """Simulates the DB: the table 'grows' over time as up_to increases."""
    df = spark.createDataFrame(_DEMO_ROWS, "order_id INT, customer_id STRING, amount DOUBLE, updated_at STRING")
    df = df.withColumn("updated_at", F.to_timestamp("updated_at"))
    return df.filter((F.col("updated_at") > F.lit(since)) & (F.col("updated_at") <= F.lit(up_to)))


# --------------------------------------------------------------------------
# One incremental run
# --------------------------------------------------------------------------
def run_increment(spark: SparkSession, lake: str, state: str, demo_now: str | None = None) -> None:
    last = read_watermark(state)
    since_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S") - OVERLAP
    since = max(since_dt, datetime(1970, 1, 1)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n-- run: watermark={last}, querying updated_at > {since}")

    src = read_source_jdbc(spark, since) if USE_JDBC else read_source_demo(spark, since, demo_now)
    src = src.withColumn("extract_ts", F.current_timestamp()).cache()

    n = src.count()
    if n == 0:
        print("No new rows; watermark unchanged.")
        return

    new_wm = src.agg(F.max("updated_at")).first()[0].strftime("%Y-%m-%d %H:%M:%S")
    # Append raw changes; downstream (lesson 06/07) dedupes to latest version.
    src.withColumn("load_date", F.to_date("extract_ts")).write.mode("append") \
        .partitionBy("load_date").parquet(lake)
    save_watermark(state, new_wm)              # only AFTER a successful write
    print(f"Loaded {n} rows, new watermark={new_wm}")
    src.unpersist()


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson03_")
    lake, state = os.path.join(base, "lake", "orders_raw"), os.path.join(base, "wm.json")
    try:
        if USE_JDBC:
            run_increment(spark, lake, state)
        else:
            # Three scheduled runs as the source table grows.
            for now in ["2026-11-01 23:59:59", "2026-11-02 23:59:59", "2026-11-03 23:59:59"]:
                run_increment(spark, lake, state, demo_now=now)
            run_increment(spark, lake, state, demo_now="2026-11-03 23:59:59")  # nothing new

        print("\n== everything landed in the lake (note order 2 twice) ==")
        spark.read.parquet(lake).orderBy("order_id", "updated_at").show(truncate=False)
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Notice the overlap re-reads a row on run 2. Add a dedupe step that keeps
#      only the latest updated_at per order_id (preview of lesson 06).
#   2. Replace the JSON watermark file with a small Parquet/Delta control table
#      keyed by table name so one job can manage many tables.
#   3. Hard deletes are invisible to this pattern. Research soft deletes,
#      CDC (lesson 15) and periodic full-key reconciliation as solutions.
#
# Interview tips
#   * lowerBound/upperBound do not filter - they only decide partition strides.
#   * Use the "query" or a subquery in "dbtable" to push predicates down.
#   * Commit the watermark after the write, and use overlap + dedupe for safety.
