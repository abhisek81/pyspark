"""
Lesson 01 - Batch ETL: CSV -> cleaned Parquet with explicit schema enforcement

What you'll learn
  * Why you should NEVER rely on inferSchema in production pipelines
  * Defining an explicit StructType schema for CSV input
  * Read modes: PERMISSIVE, DROPMALFORMED, FAILFAST and the _corrupt_record column
  * Cleaning: trimming, standardising case, parsing dates, casting, defaults
  * Writing Parquet with snappy compression and a predictable layout

Key concepts (plain English)
  * Schema-on-read: CSV files carry no types. Spark either guesses (inferSchema,
    which scans the data and is slow + unstable) or you tell it (explicit schema,
    fast + deterministic). Pipelines should always tell it.
  * Corrupt records: rows that don't match the schema. In PERMISSIVE mode Spark
    keeps them, sets bad fields to null, and (if you ask) puts the raw line in a
    special column so you can quarantine it instead of silently losing data.
  * Parquet: a columnar, compressed, typed file format. Downstream readers get the
    schema for free and only read the columns they need.

Real-world use case
  A partner drops a daily orders.csv into a landing zone. You must load it into
  the curated zone as Parquet, rejecting malformed rows into a quarantine folder.

Run
  pip install pyspark==3.5.1
  python lessons/01_batch_etl_csv_to_parquet.py
"""

import os
import shutil
import tempfile

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

# Sample "partner file". Note the problems hidden in it:
#   - extra spaces and mixed-case country codes
#   - a row with a non-numeric quantity ("two")
#   - a row with an impossible date
#   - a completely broken row
SAMPLE_CSV = """order_id,customer_id,order_date,country,quantity,unit_price
1001, C01 ,2026-11-01, in ,2,499.0
1002,C02,2026-11-01,US,1,1299.5
1003,C03,2026-11-02,us,two,15.0
1004,C01,2026-13-45,IN,5,20.0
1005,C04,2026-11-03,GB,3,
this,row,is,completely,broken,data,with,extra,cols
1006,C05,2026-11-03,de,4,9.99
"""

# Explicit schema. order_date is read as a STRING on purpose: parsing dates in
# the reader hides errors (bad dates just become null). Parsing them ourselves
# lets us detect and quarantine them.
RAW_SCHEMA = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("country", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    # Must be declared in the schema for Spark to populate it.
    StructField("_corrupt_record", StringType(), True),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("lesson01-csv-to-parquet")
        .master("local[*]")
        # Small data -> few shuffle partitions. Default 200 is overkill locally.
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def write_sample_file(base_dir: str) -> str:
    landing = os.path.join(base_dir, "landing")
    os.makedirs(landing, exist_ok=True)
    path = os.path.join(landing, "orders.csv")
    with open(path, "w") as f:
        f.write(SAMPLE_CSV)
    return landing


def extract(spark: SparkSession, landing_dir: str) -> DataFrame:
    """Read CSV with an explicit schema in PERMISSIVE mode."""
    df = (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")            # keep bad rows, don't crash
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .schema(RAW_SCHEMA)
        .csv(landing_dir)
    )
    # Spark quirk: queries that reference ONLY the corrupt-record column are
    # disallowed on raw CSV. Caching materialises the parse first.
    return df.cache()


def split_valid_invalid(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Separate rows that parsed cleanly from rows we must quarantine."""
    parsed_date = F.to_date("order_date", "yyyy-MM-dd")
    df = df.withColumn("parsed_date", parsed_date)

    is_bad = (
        F.col("_corrupt_record").isNotNull()     # type mismatch / wrong col count
        | F.col("order_id").isNull()
        | F.col("parsed_date").isNull()          # impossible dates
    )
    invalid = df.filter(is_bad).withColumn(
        "reject_reason",
        F.when(F.col("_corrupt_record").isNotNull(), "malformed_row")
         .when(F.col("parsed_date").isNull(), "invalid_date")
         .otherwise("missing_key"),
    )
    valid = df.filter(~is_bad)
    return valid, invalid


def transform(valid: DataFrame) -> DataFrame:
    """Standardise and enrich clean rows."""
    return (
        valid
        .select(
            "order_id",
            F.trim("customer_id").alias("customer_id"),
            F.col("parsed_date").alias("order_date"),
            F.upper(F.trim("country")).alias("country"),
            "quantity",
            # Business rule: missing price -> 0.0 and flag it, never guess.
            F.coalesce("unit_price", F.lit(0.0)).alias("unit_price"),
            F.col("unit_price").isNull().alias("price_missing"),
        )
        .withColumn("line_total", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("ingested_at", F.current_timestamp())
    )


def load(clean: DataFrame, invalid: DataFrame, base_dir: str) -> None:
    curated = os.path.join(base_dir, "curated", "orders")
    quarantine = os.path.join(base_dir, "quarantine", "orders")

    # mode("overwrite") makes the job re-runnable (idempotent) for a full load.
    clean.write.mode("overwrite").option("compression", "snappy").parquet(curated)
    invalid.write.mode("overwrite").json(quarantine)   # JSON keeps raw text readable
    print(f"Curated -> {curated}")
    print(f"Quarantine -> {quarantine}")


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base_dir = tempfile.mkdtemp(prefix="lesson01_")
    try:
        landing = write_sample_file(base_dir)
        raw = extract(spark, landing)
        print("== Raw (as parsed) ==")
        raw.show(truncate=False)

        valid, invalid = split_valid_invalid(raw)
        clean = transform(valid)

        print("== Clean ==")
        clean.show(truncate=False)
        clean.printSchema()
        print("== Quarantined ==")
        invalid.select("order_id", "reject_reason", "_corrupt_record").show(truncate=False)

        load(clean, invalid, base_dir)

        # Read back to prove Parquet kept the types (no schema needed!)
        back = spark.read.parquet(os.path.join(base_dir, "curated", "orders"))
        back.printSchema()
        print(f"Rows in curated: {back.count()}, rows rejected: {invalid.count()}")
    finally:
        spark.stop()
        shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Switch mode to DROPMALFORMED and FAILFAST. What happens to row 1003 and
#      the broken row in each case? Which would you choose for a finance feed?
#   2. Add a rule that quarantines rows with quantity <= 0 or quantity > 1000.
#   3. Write the curated output partitioned by order_date and compare the folder
#      structure. Then add a row-count reconciliation: raw = clean + rejected.
#
# Interview tips
#   * "Why not inferSchema?" -> extra full scan, types can change between files,
#     and silent nulls. Explicit schemas are a data contract.
#   * Know the three read modes and that _corrupt_record must be in the schema.
#   * Parquet vs CSV: columnar, compressed, typed, supports predicate pushdown.
