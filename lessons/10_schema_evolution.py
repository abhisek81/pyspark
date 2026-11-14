"""
Lesson 10 - Schema evolution and mergeSchema handling

What you'll learn
  * What happens when upstream adds, removes, or changes a column
  * Parquet: reading files with different schemas using mergeSchema
  * Delta: schema ENFORCEMENT (writes rejected) vs schema EVOLUTION (mergeSchema)
  * Auto schema evolution inside MERGE
  * Safe vs unsafe changes (add column = safe, type change = dangerous)

Key concepts (plain English)
  * Upstream systems change. A new column appears in the feed, a column is
    dropped, or an int becomes a string. Your pipeline must either adapt
    safely or fail loudly - never corrupt data silently.
  * Delta enforces the table schema by default: a write with an unexpected
    column FAILS. That's a feature: it stops accidental garbage.
  * When the change is intended, opt in with .option("mergeSchema", "true").
    New columns are added; old rows get NULL for them.
  * Changing a column's type or dropping a column requires an explicit
    overwriteSchema rewrite (or Delta column mapping) - a deliberate migration.

Real-world use case
  The orders feed adds "coupon_code" in v2 and "channel" in v3. The lake table
  should pick up new columns without a manual DDL change, but reject a feed
  where "amount" suddenly arrives as text.

Run
  pip install pyspark==3.5.1 delta-spark==3.2.0
  python lessons/10_schema_evolution.py
"""

import os
import shutil
import tempfile

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("lesson10-schema-evolution").master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def parquet_merge_schema_demo(spark: SparkSession, base: str) -> None:
    print("\n======== Parquet: files with different schemas ========")
    path = os.path.join(base, "pq_orders")
    spark.createDataFrame([(1, 10.0)], "order_id INT, amount DOUBLE") \
        .write.mode("overwrite").parquet(os.path.join(path, "v=1"))
    spark.createDataFrame([(2, 20.0, "SAVE5")], "order_id INT, amount DOUBLE, coupon_code STRING") \
        .write.mode("overwrite").parquet(os.path.join(path, "v=2"))

    print("Without mergeSchema (schema taken from ONE file - column may be missing):")
    spark.read.parquet(path).printSchema()

    print("With mergeSchema (union of all file schemas; costs a footer scan):")
    df = spark.read.option("mergeSchema", "true").parquet(path)
    df.printSchema()
    df.orderBy("order_id").show()


def delta_enforcement_and_evolution(spark: SparkSession, base: str) -> None:
    print("\n======== Delta: enforcement vs evolution ========")
    path = os.path.join(base, "delta_orders")

    v1 = spark.createDataFrame([(1, 10.0)], "order_id INT, amount DOUBLE")
    v1.write.format("delta").save(path)

    v2 = spark.createDataFrame([(2, 20.0, "SAVE5")], "order_id INT, amount DOUBLE, coupon_code STRING")

    # 1) Enforcement: unexpected column -> AnalysisException
    try:
        v2.write.format("delta").mode("append").save(path)
    except Exception as e:  # noqa: BLE001
        print("Append rejected (schema enforcement):", str(e).splitlines()[0][:120])

    # 2) Evolution: opt in for this write only
    v2.write.format("delta").mode("append").option("mergeSchema", "true").save(path)
    print("After mergeSchema append - old row has NULL coupon_code:")
    spark.read.format("delta").load(path).orderBy("order_id").show()

    # 3) Missing columns are fine: a batch without coupon_code just writes NULLs.
    spark.createDataFrame([(3, 30.0)], "order_id INT, amount DOUBLE") \
        .write.format("delta").mode("append").save(path)

    # 4) Type change: amount as STRING -> rejected even WITH mergeSchema.
    bad = spark.createDataFrame([(4, "40.0", None)], "order_id INT, amount STRING, coupon_code STRING")
    try:
        bad.write.format("delta").mode("append").option("mergeSchema", "true").save(path)
    except Exception as e:  # noqa: BLE001
        print("Type change rejected:", str(e).splitlines()[0][:120])
    # The correct fix is usually to CAST in the pipeline, not to change the table:
    bad.withColumn("amount", F.col("amount").cast("double")) \
        .write.format("delta").mode("append").save(path)

    # 5) Evolution inside MERGE: new column "channel" arrives with upserts.
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
    updates = spark.createDataFrame(
        [(1, 12.0, None, "web"), (5, 50.0, None, "app")],
        "order_id INT, amount DOUBLE, coupon_code STRING, channel STRING")
    (DeltaTable.forPath(spark, path).alias("t")
     .merge(updates.alias("s"), "t.order_id = s.order_id")
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .execute())
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "false")

    print("Final table after MERGE with auto schema evolution:")
    final = spark.read.format("delta").load(path)
    final.printSchema()
    final.orderBy("order_id").show()

    print("Schema changes are recorded in table history:")
    DeltaTable.forPath(spark, path).history().select("version", "operation", "operationParameters") \
        .show(truncate=False)


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson10_")
    try:
        parquet_merge_schema_demo(spark, base)
        delta_enforcement_and_evolution(spark, base)
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Rename a column with Delta column mapping:
#      ALTER TABLE ... SET TBLPROPERTIES ('delta.columnMapping.mode'='name',
#      'delta.minReaderVersion'='2','delta.minWriterVersion'='5') then RENAME COLUMN.
#   2. Build a "schema contract" check: compare the incoming DataFrame schema to
#      an expected StructType and fail on removed/retyped columns but allow
#      added ones (log them).
#   3. Change amount from DOUBLE to DECIMAL(12,2) with overwriteSchema and
#      discuss the impact on downstream readers.
#
# Interview tips
#   * Enforcement (reject) is the default in Delta; evolution is opt-in per write.
#   * Adding nullable columns is safe; type changes and drops are migrations.
#   * Parquet mergeSchema is a read-time union and is off by default for cost.
