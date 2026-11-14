"""
Lesson 04 - Partitioned data lake writes (partitionBy, overwrite modes)

What you'll learn
  * How partitionBy lays out folders (Hive-style col=value directories)
  * Partition pruning: why filters on partition columns are so fast
  * Static vs DYNAMIC partition overwrite (and the classic data-loss bug)
  * Controlling file counts per partition with repartition before write
  * Choosing good partition columns (and avoiding over-partitioning)

Key concepts (plain English)
  * A data lake is just files in folders (S3, ADLS, GCS, HDFS). Partitioning
    puts each value of a column (e.g. event_date) in its own folder:
        events/event_date=2026-11-01/part-0000.parquet
  * Queries that filter on event_date read ONLY the matching folders.
  * mode("overwrite") by default deletes the WHOLE table folder - re-running
    one day would wipe all other days. Setting
    spark.sql.sources.partitionOverwriteMode=dynamic makes overwrite replace
    only the partitions present in the DataFrame being written. This is what
    makes daily backfills safe and idempotent.
  * Rule of thumb: partition columns should have low-to-medium cardinality
    (dates, regions) and each partition should hold >= ~128MB of data.

Real-world use case
  A daily job writes click events to s3://lake/events/. Days are sometimes
  re-processed (late data, bug fixes) and must not affect other days.

Run
  python lessons/04_partitioned_data_lake_writes.py
  # On a real lake just change LAKE_ROOT, e.g. s3a://my-bucket/lake
"""

import os
import shutil
import tempfile

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

LAKE_ROOT = os.getenv("LAKE_ROOT")   # None -> use a temp folder for the demo


def build_spark() -> SparkSession:
    return (SparkSession.builder.appName("lesson04-partitioned-writes")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8").getOrCreate())


def make_events(spark: SparkSession, days: list[str], version: str) -> DataFrame:
    """Generate fake click events: 1,000 rows per day across 3 regions."""
    df = spark.range(0, 1000 * len(days)).withColumn(
        "event_date", F.element_at(F.array(*[F.lit(d) for d in days]),
                                   (F.col("id") % len(days) + 1).cast("int")))
    return (df
            .withColumn("region", F.element_at(F.array(F.lit("apac"), F.lit("emea"), F.lit("amer")),
                                               (F.col("id") % 3 + 1).cast("int")))
            .withColumn("user_id", (F.col("id") % 97).cast("string"))
            .withColumn("event_date", F.to_date("event_date"))
            .withColumn("version", F.lit(version)))


def show_layout(path: str) -> None:
    """Print the folder tree so you can SEE what partitionBy does."""
    for root, _dirs, files in sorted(os.walk(path)):
        parquet = [f for f in files if f.endswith(".parquet")]
        if parquet:
            print(f"  {os.path.relpath(root, path):40s} {len(parquet)} file(s)")


def write_partitioned(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    (
        df
        # repartition by the partition columns -> each output folder gets one
        # file per write (instead of up to shuffle.partitions tiny files).
        .repartition("event_date", "region")
        .write
        .mode(mode)
        .partitionBy("event_date", "region")
        .parquet(path)
    )


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    tmp = None if LAKE_ROOT else tempfile.mkdtemp(prefix="lesson04_")
    root = LAKE_ROOT or tmp
    path = os.path.join(root, "events")
    try:
        # 1) Initial load: 3 days.
        write_partitioned(make_events(spark, ["2026-11-01", "2026-11-02", "2026-11-03"], "v1"), path)
        print("Initial layout:")
        show_layout(path)

        # 2) Partition pruning. Look for PartitionFilters in the physical plan.
        one_day = spark.read.parquet(path).filter(F.col("event_date") == "2026-11-02")
        one_day.explain()
        print("rows for 2026-11-02:", one_day.count())

        # 3) THE BUG: static overwrite of a single day wipes everything else.
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "static")
        write_partitioned(make_events(spark, ["2026-11-02"], "v2"), path)
        print("\nAfter STATIC overwrite of 2026-11-02 (other days are gone!):")
        spark.read.parquet(path).groupBy("event_date", "version").count().orderBy("event_date").show()

        # Restore the original three days for the next step.
        write_partitioned(make_events(spark, ["2026-11-01", "2026-11-02", "2026-11-03"], "v1"), path)

        # 4) THE FIX: dynamic overwrite replaces only partitions being written.
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        write_partitioned(make_events(spark, ["2026-11-02"], "v2"), path)
        print("After DYNAMIC overwrite of 2026-11-02 (only that day changed):")
        spark.read.parquet(path).groupBy("event_date", "version").count().orderBy("event_date").show()

        # 5) Append mode adds new partitions (or new files to existing ones).
        write_partitioned(make_events(spark, ["2026-11-04"], "v1"), path, mode="append")
        print("Final layout:")
        show_layout(path)
    finally:
        spark.stop()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Partition by user_id instead of region. Count the folders and files.
#      Why is high-cardinality partitioning harmful (small files, slow listing)?
#   2. Use .option("maxRecordsPerFile", 200) and observe the effect.
#   3. Read with spark.read.option("basePath", path).parquet(path + "/event_date=2026-11-0*")
#      and explain why basePath is needed to keep event_date as a column.
#
# Interview tips
#   * Explain partition pruning vs predicate pushdown (folder skip vs row-group skip).
#   * Always mention partitionOverwriteMode=dynamic for idempotent backfills.
#   * Over-partitioning -> small files problem (see lesson 22).
