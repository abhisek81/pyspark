"""
Lesson 09 - Medallion architecture: bronze -> silver -> gold pipeline

What you'll learn
  * What belongs in each layer (bronze / silver / gold) and why
  * Bronze: raw, append-only, with ingestion metadata (source file, load time)
  * Silver: typed, cleaned, deduplicated, conformed - "one version of the truth"
  * Gold: business-level aggregates shaped for dashboards and ML
  * Incremental silver updates with MERGE, and rebuilding gold from silver

Key concepts (plain English)
  * Bronze = "what we received". Keep it raw so you can always reprocess when
    business rules change or a bug is found. Never clean data in bronze.
  * Silver = "what is true". Apply schema, data quality (lesson 05), dedupe
    (lesson 06), and upserts (lesson 07). Analysts and data scientists who need
    detail use silver.
  * Gold = "what the business asks". Aggregated, denormalised, named in
    business language (daily_revenue_by_country). Small and fast.
  * Each layer is its own Delta table, so every step is ACID and re-runnable.

Real-world use case
  Order events arrive as JSON files every few minutes. Finance needs a daily
  revenue dashboard; data science needs clean order-level history.

Run
  pip install pyspark==3.5.1 delta-spark==3.2.0
  python lessons/09_medallion_architecture.py
"""

import json
import os
import shutil
import tempfile

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

ORDER_SCHEMA = StructType([
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("country", StringType()),
    StructField("status", StringType()),
    StructField("amount", StringType()),       # arrives as text from upstream!
    StructField("event_ts", StringType()),
])


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("lesson09-medallion").master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def land_batch(landing: str, name: str, events: list[dict]) -> str:
    folder = os.path.join(landing, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "events.json"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return folder


# ----------------------------- BRONZE --------------------------------------
def to_bronze(spark: SparkSession, batch_dir: str, bronze: str) -> None:
    df = (spark.read.schema(ORDER_SCHEMA).json(batch_dir)
          # Lineage columns: where did each row come from and when?
          .withColumn("_source_file", F.input_file_name())
          .withColumn("_ingested_at", F.current_timestamp()))
    df.write.format("delta").mode("append").save(bronze)


# ----------------------------- SILVER --------------------------------------
def bronze_to_silver(spark: SparkSession, bronze: str, silver: str) -> None:
    b = spark.read.format("delta").load(bronze)

    typed = (b.select(
                F.trim("order_id").alias("order_id"),
                F.trim("customer_id").alias("customer_id"),
                F.upper(F.trim("country")).alias("country"),
                F.upper(F.trim("status")).alias("status"),
                F.col("amount").cast("decimal(12,2)").alias("amount"),
                F.to_timestamp("event_ts").alias("event_ts"),
                "_ingested_at")
             # Basic quality gate (see lesson 05 for the full version).
             .filter(F.col("order_id").isNotNull() & F.col("event_ts").isNotNull()
                     & F.col("amount").isNotNull()))

    # Latest event per order wins (lesson 06).
    w = Window.partitionBy("order_id").orderBy(F.col("event_ts").desc(), F.col("_ingested_at").desc())
    latest = typed.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

    if not DeltaTable.isDeltaTable(spark, silver):
        latest.write.format("delta").save(silver)
        return

    # Upsert, but never let an OLDER event overwrite a newer one (out-of-order data).
    (DeltaTable.forPath(spark, silver).alias("t")
     .merge(latest.alias("s"), "t.order_id = s.order_id")
     .whenMatchedUpdateAll(condition="s.event_ts > t.event_ts")
     .whenNotMatchedInsertAll()
     .execute())


# ------------------------------ GOLD ---------------------------------------
def silver_to_gold(spark: SparkSession, silver: str, gold: str) -> None:
    s = spark.read.format("delta").load(silver)
    gold_df = (s.filter(F.col("status").isin("PAID", "SHIPPED"))
               .groupBy(F.to_date("event_ts").alias("order_date"), "country")
               .agg(F.sum("amount").alias("revenue"),
                    F.countDistinct("order_id").alias("orders"),
                    F.countDistinct("customer_id").alias("customers"))
               .withColumn("avg_order_value", F.round(F.col("revenue") / F.col("orders"), 2)))
    # Gold is small: a full rebuild is simple and always consistent.
    (gold_df.write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").save(gold))


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson09_")
    landing = os.path.join(base, "landing")
    bronze, silver, gold = (os.path.join(base, x) for x in ("bronze", "silver", "gold"))
    try:
        batch1 = land_batch(landing, "batch1", [
            {"order_id": "O-1", "customer_id": "C1", "country": "in", "status": "paid", "amount": "100.50", "event_ts": "2026-11-01 10:00:00"},
            {"order_id": "O-2", "customer_id": "C2", "country": "US", "status": "PAID", "amount": "250", "event_ts": "2026-11-01 11:00:00"},
            {"order_id": "O-3", "customer_id": "C3", "country": "US", "status": "CREATED", "amount": "abc", "event_ts": "2026-11-01 12:00:00"},
        ])
        batch2 = land_batch(landing, "batch2", [
            {"order_id": "O-2", "customer_id": "C2", "country": "US", "status": "SHIPPED", "amount": "250", "event_ts": "2026-11-02 09:00:00"},
            {"order_id": "O-1", "customer_id": "C1", "country": "IN", "status": "CREATED", "amount": "100.50", "event_ts": "2026-11-01 09:00:00"},  # late + older
            {"order_id": "O-4", "customer_id": "C1", "country": "IN", "status": "PAID", "amount": "40", "event_ts": "2026-11-02 10:00:00"},
        ])

        for i, batch in enumerate([batch1, batch2], start=1):
            to_bronze(spark, batch, bronze)
            bronze_to_silver(spark, bronze, silver)
            silver_to_gold(spark, silver, gold)
            print(f"\n######## after batch {i} ########")
            print("BRONZE (raw, everything kept):")
            spark.read.format("delta").load(bronze).drop("_source_file").show(truncate=False)
            print("SILVER (typed, clean, latest per order):")
            spark.read.format("delta").load(silver).orderBy("order_id").show(truncate=False)
            print("GOLD (daily revenue by country):")
            spark.read.format("delta").load(gold).orderBy("order_date", "country").show()
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. bronze_to_silver re-reads ALL of bronze each run. Make it incremental:
#      track the max _ingested_at processed (lesson 03) or use Delta Change Data
#      Feed (delta.enableChangeDataFeed) to read only new bronze rows.
#   2. O-3 has amount "abc" and silently disappears. Route it to a quarantine
#      table instead (lesson 05).
#   3. Convert bronze ingestion to Structured Streaming with Auto Loader-style
#      file source (spark.readStream.schema(...).json(landing)).
#
# Interview tips
#   * Bronze is raw + append-only so you can replay; silver is cleaned/conformed;
#     gold is aggregated for consumers.
#   * Guard against out-of-order events with a MERGE condition on event time.
#   * input_file_name() / _metadata.file_path give lineage for debugging.
