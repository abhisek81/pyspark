"""
Lesson 08 - SCD Type 2 history tracking with Delta Lake MERGE

What you'll learn
  * The SCD2 table shape: surrogate key, effective_from, effective_to, is_current
  * The "staged updates" trick: expire old row AND insert new row in ONE MERGE
  * Detecting changes with a hash of tracked columns
  * Querying "as of" a date: what did this customer look like last month?
  * Keeping SCD2 loads idempotent when the same extract is applied twice

Key concepts (plain English)
  * Type 2 keeps history. When a tracked attribute changes, we do NOT update
    the row. We close the current row (effective_to = now, is_current = false)
    and insert a new current row. Facts (orders) can then join to the version
    of the customer that was valid at order time.
  * A single MERGE can only do one action per source row. But a change needs
    two actions (update old + insert new). Solution: feed MERGE each changed
    record TWICE - once with a real merge key (to match and expire the old row)
    and once with a NULL merge key (never matches -> gets inserted).

Real-world use case
  dim_customer must keep every change of city and tier so revenue reports by
  tier are correct for the tier the customer had at the time of purchase.

Run
  pip install pyspark==3.5.1 delta-spark==3.2.0
  python lessons/08_scd_type2_delta_merge.py
"""

import os
import shutil
import tempfile

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

TRACKED = ["city", "tier"]
HIGH_DATE = "9999-12-31 00:00:00"


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("lesson08-scd2").master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def hash_cols(df: DataFrame) -> DataFrame:
    return df.withColumn("row_hash", F.sha2(F.concat_ws("||", *[
        F.coalesce(F.col(c), F.lit("<null>")) for c in TRACKED]), 256))


def surrogate_key(df: DataFrame) -> DataFrame:
    """Deterministic surrogate key per version: hash(business key + start time).
    Deterministic beats uuid() here - Delta MERGE may scan the source more than
    once, and non-deterministic values could differ between scans."""
    return df.withColumn("customer_sk", F.sha2(F.concat_ws(
        "|", F.col("customer_id").cast("string"), F.col("change_ts").cast("string")), 256))


def snapshot(spark: SparkSession, rows: list[tuple], as_of: str) -> DataFrame:
    """A source extract; as_of is when the change became effective."""
    return hash_cols(spark.createDataFrame(rows, "customer_id INT, name STRING, city STRING, tier STRING")
                     .withColumn("change_ts", F.to_timestamp(F.lit(as_of))))


def initial_load(spark: SparkSession, path: str, src: DataFrame) -> None:
    (surrogate_key(src)
        .withColumn("effective_from", F.col("change_ts"))
        .withColumn("effective_to", F.to_timestamp(F.lit(HIGH_DATE)))
        .withColumn("is_current", F.lit(True))
        .drop("change_ts")
        .write.format("delta").mode("overwrite").save(path))


def merge_scd2(spark: SparkSession, path: str, src: DataFrame) -> None:
    target = DeltaTable.forPath(spark, path)
    current = target.toDF().filter("is_current")

    # Rows whose tracked attributes changed vs the current version.
    changed = (src.alias("s").join(current.alias("t"), "customer_id")
               .filter("s.row_hash <> t.row_hash").select("s.*"))

    # Staged updates:
    #   merge_key = customer_id -> matches current row -> expire it
    #   merge_key = NULL        -> never matches       -> insert new version
    # Brand-new customers only appear with their real key and fall into
    # "not matched" -> insert.
    staged = surrogate_key(
        src.withColumn("merge_key", F.col("customer_id"))
        .unionByName(changed.withColumn("merge_key", F.lit(None).cast("int"))))

    (
        target.alias("t")
        .merge(staged.alias("s"), "t.customer_id = s.merge_key AND t.is_current = true")
        .whenMatchedUpdate(
            condition="t.row_hash <> s.row_hash",
            set={"is_current": "false", "effective_to": "s.change_ts"})
        .whenNotMatchedInsert(
            # Only insert if new customer OR a real change. Guard against
            # re-running the same extract: unchanged rows with NULL key never
            # exist (they're not in 'changed'), and unchanged rows with a real
            # key match and are skipped by the update condition.
            values={
                "customer_sk": "s.customer_sk",
                "customer_id": "s.customer_id",
                "name": "s.name",
                "city": "s.city",
                "tier": "s.tier",
                "row_hash": "s.row_hash",
                "effective_from": "s.change_ts",
                "effective_to": f"timestamp'{HIGH_DATE}'",
                "is_current": "true",
            })
        .execute()
    )


def show(spark: SparkSession, path: str, title: str) -> None:
    print(f"== {title} ==")
    (spark.read.format("delta").load(path)
     .select("customer_id", "name", "city", "tier", "effective_from", "effective_to", "is_current")
     .orderBy("customer_id", "effective_from").show(truncate=False))


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson08_")
    path = os.path.join(base, "dim_customer_scd2")
    try:
        day1 = snapshot(spark, [(1, "Asha", "Pune", "GOLD"), (2, "Ben", "Austin", "SILVER")],
                        "2026-10-01 00:00:00")
        initial_load(spark, path, day1)
        show(spark, path, "day 1")

        day2 = snapshot(spark, [(1, "Asha", "Pune", "GOLD"),          # unchanged
                                (2, "Ben", "Dallas", "GOLD"),         # moved + upgraded
                                (3, "Chen", "Shanghai", "BRONZE")],   # new
                        "2026-11-01 00:00:00")
        merge_scd2(spark, path, day2)
        show(spark, path, "day 2")

        merge_scd2(spark, path, day2)                 # re-run: must change nothing
        show(spark, path, "day 2 re-run (idempotent)")

        # Point-in-time lookup: Ben's tier on 2026-10-15
        as_of = "2026-10-15 00:00:00"
        print(f"== dimension as of {as_of} ==")
        (spark.read.format("delta").load(path)
         .filter((F.col("effective_from") <= as_of) & (F.col("effective_to") > as_of))
         .select("customer_id", "city", "tier").orderBy("customer_id").show())
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Add an orders fact table with order_ts and join it to the SCD2 dimension
#      using effective_from <= order_ts < effective_to. Sum revenue by tier.
#   2. Make "name" a Type 1 column (overwrite on all versions) while city/tier
#      stay Type 2. This hybrid is very common.
#   3. Handle deletes: when a customer disappears from a full extract, expire
#      the current row and set an is_deleted flag.
#
# Interview tips
#   * Be able to whiteboard the staged-updates (NULL merge key) technique.
#   * Use half-open intervals [from, to) so a timestamp matches exactly one version.
#   * Surrogate keys decouple facts from business keys that may be reused.
