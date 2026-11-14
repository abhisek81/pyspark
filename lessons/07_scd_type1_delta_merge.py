"""
Lesson 07 - SCD Type 1 upsert with Delta Lake MERGE

What you'll learn
  * Setting up a local SparkSession with Delta Lake
  * MERGE INTO: update matched rows, insert new rows, optionally delete
  * Python DeltaTable API vs SQL MERGE syntax (same engine underneath)
  * Avoiding no-op updates with a change-detection condition (row hash)
  * Why the source MUST be deduplicated before MERGE

Key concepts (plain English)
  * SCD = Slowly Changing Dimension: reference data (customers, products) that
    changes occasionally. Type 1 = overwrite the old value, keep no history.
    Use it for corrections (typo in a name) or attributes where history
    doesn't matter.
  * Plain Parquet can't update rows in place - you'd rewrite the whole table.
    Delta Lake adds a transaction log on top of Parquet so MERGE/UPDATE/DELETE
    are ACID: readers never see half-applied changes.
  * If two source rows match the same target row, MERGE fails with
    "multiple source rows matched". Always dedupe first (lesson 06).

Real-world use case
  A daily customer extract from the CRM must be applied to dim_customer:
  new customers inserted, changed customers updated in place.

Run
  pip install pyspark==3.5.1 delta-spark==3.2.0
  python lessons/07_scd_type1_delta_merge.py
  # or: spark-submit --packages io.delta:delta-spark_2.12:3.2.0 \
  #   --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  #   --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  #   lessons/07_scd_type1_delta_merge.py
"""

import os
import shutil
import tempfile

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

TRACKED_COLS = ["name", "email", "city", "tier"]


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("lesson07-scd1")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    # Adds the matching delta jars when delta-spark was installed via pip.
    return configure_spark_with_delta_pip(builder).getOrCreate()


def with_row_hash(df: DataFrame) -> DataFrame:
    """Hash of tracked columns. Comparing one hash is cheaper and null-safe
    compared to writing a.col <> b.col OR ... for every column."""
    return df.withColumn(
        "row_hash",
        F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("<null>"))
                                   for c in TRACKED_COLS]), 256))


def initial_load(spark: SparkSession, path: str) -> None:
    rows = [
        (1, "Asha Rao", "asha@example.com", "Pune", "GOLD"),
        (2, "Ben Ford", "ben@example.com", "Austin", "SILVER"),
        (3, "Chen Li", "chen@example.com", "Shanghai", "BRONZE"),
    ]
    df = spark.createDataFrame(rows, "customer_id INT, name STRING, email STRING, city STRING, tier STRING")
    (with_row_hash(df).withColumn("updated_at", F.current_timestamp())
        .write.format("delta").mode("overwrite").save(path))


def daily_extract(spark: SparkSession) -> DataFrame:
    rows = [
        (2, "Ben Ford", "ben.ford@example.com", "Austin", "GOLD", "2026-11-02 09:00:00"),   # changed
        (2, "Ben Ford", "ben.ford@example.com", "Dallas", "GOLD", "2026-11-02 10:00:00"),   # changed again!
        (3, "Chen Li", "chen@example.com", "Shanghai", "BRONZE", "2026-11-02 09:00:00"),    # unchanged
        (4, "Dev Patel", "dev@example.com", "Mumbai", "SILVER", "2026-11-02 09:30:00"),     # new
    ]
    return spark.createDataFrame(
        rows, "customer_id INT, name STRING, email STRING, city STRING, tier STRING, src_ts STRING")


def dedupe_source(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("customer_id").orderBy(F.col("src_ts").desc())
    return df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn", "src_ts")


def merge_scd1(spark: SparkSession, path: str, updates: DataFrame) -> None:
    target = DeltaTable.forPath(spark, path)
    src = with_row_hash(updates).withColumn("updated_at", F.current_timestamp())
    (
        target.alias("t")
        .merge(src.alias("s"), "t.customer_id = s.customer_id")
        # Only rewrite rows whose tracked values actually changed. Without this
        # condition, unchanged customers are rewritten every day: more files,
        # misleading updated_at, wasted compute.
        .whenMatchedUpdate(condition="t.row_hash <> s.row_hash", set={
            "name": "s.name", "email": "s.email", "city": "s.city",
            "tier": "s.tier", "row_hash": "s.row_hash", "updated_at": "s.updated_at",
        })
        .whenNotMatchedInsertAll()
        .execute()
    )


def merge_scd1_sql(spark: SparkSession, path: str, updates: DataFrame) -> None:
    """Same logic in SQL - handy when your team prefers SQL."""
    with_row_hash(updates).withColumn("updated_at", F.current_timestamp()) \
        .createOrReplaceTempView("customer_updates")
    spark.sql(f"""
        MERGE INTO delta.`{path}` AS t
        USING customer_updates AS s
        ON t.customer_id = s.customer_id
        WHEN MATCHED AND t.row_hash <> s.row_hash THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson07_")
    path = os.path.join(base, "dim_customer")
    try:
        initial_load(spark, path)
        print("== before ==")
        spark.read.format("delta").load(path).orderBy("customer_id").show(truncate=False)

        raw = daily_extract(spark)
        try:
            merge_scd1(spark, path, raw.drop("src_ts"))       # NOT deduped
        except Exception as e:  # noqa: BLE001 - demo of a real failure
            print("MERGE failed as expected without dedupe:\n ", str(e).splitlines()[0])

        merge_scd1(spark, path, dedupe_source(raw))
        print("== after SCD1 merge ==")
        spark.read.format("delta").load(path).orderBy("customer_id").show(truncate=False)

        # Delta keeps a log of every operation - see numTargetRowsUpdated etc.
        DeltaTable.forPath(spark, path).history() \
            .select("version", "operation", "operationMetrics").show(truncate=False)
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Run merge_scd1_sql with the same deduped data a second time. Check the
#      history: how many rows were updated? Why (hint: row_hash)?
#   2. Add whenNotMatchedBySourceDelete() to delete customers missing from a
#      FULL extract. Why is that dangerous with incremental extracts?
#   3. Add a soft-delete flag column instead of physically deleting rows.
#
# Interview tips
#   * SCD1 = overwrite, SCD2 = history rows (next lesson), SCD3 = previous-value column.
#   * "Multiple source rows matched" -> dedupe the source on the merge key.
#   * Explain how Delta MERGE works: find touched files, rewrite them, commit log.
