"""
Lesson 05 - Data quality checks: nulls, ranges, uniqueness, quarantine bad records

What you'll learn
  * Expressing data quality rules as reusable column expressions
  * Row-level checks (tag + quarantine) vs dataset-level checks (fail the job)
  * Computing many metrics in ONE pass with a single agg() call
  * Uniqueness / duplicate-key detection with window functions
  * Producing a data quality report DataFrame you can store and alert on

Key concepts (plain English)
  * Row-level rule: "amount must be >= 0". Rows that break it are moved to a
    quarantine table with the reason, good rows continue. The job keeps running.
  * Dataset-level rule: "null rate of email < 5%" or "row count > 0". If broken,
    the whole batch is suspicious, so we FAIL the job before publishing it.
  * Doing each check as a separate count() means a separate scan per rule.
    Building all the conditions into one agg() scans the data once.

Real-world use case
  Before publishing the customers table to analysts, enforce rules agreed with
  the business and stop the pipeline if quality drops below thresholds.

Run
  python lessons/05_data_quality_checks.py
"""

from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame, Column, Window
from pyspark.sql import functions as F


@dataclass
class RowRule:
    name: str
    passes: Column          # expression that is TRUE when the row is good


@dataclass
class DatasetRule:
    name: str
    metric: str             # metric name produced by profile()
    op: str                 # "<=" or ">="
    threshold: float


def build_spark() -> SparkSession:
    return (SparkSession.builder.appName("lesson05-data-quality")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


def sample_customers(spark: SparkSession) -> DataFrame:
    rows = [
        (1, "asha@example.com", "IN", 34, 1200.0),
        (2, "ben@example.com", "US", 41, 560.0),
        (3, None, "US", 29, 80.0),                  # missing email
        (4, "chen@example", "CN", 230, 300.0),      # bad email, impossible age
        (5, "dev@example.com", "XX", 22, -50.0),    # unknown country, negative spend
        (2, "ben2@example.com", "US", 41, 10.0),    # duplicate customer_id
        (6, "eve@example.com", "GB", None, 75.0),   # missing age
    ]
    return spark.createDataFrame(
        rows, "customer_id INT, email STRING, country STRING, age INT, lifetime_spend DOUBLE")


VALID_COUNTRIES = ["IN", "US", "GB", "CN", "DE"]

# A null in a comparison yields null, not false. coalesce(..., False) makes
# "null means fail" explicit - an easy bug to miss in DQ code.
ROW_RULES = [
    RowRule("customer_id_not_null", F.col("customer_id").isNotNull()),
    RowRule("email_format", F.coalesce(
        F.col("email").rlike(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$"), F.lit(False))),
    RowRule("age_in_range", F.coalesce(F.col("age").between(0, 120), F.lit(True))),  # null age allowed
    RowRule("country_valid", F.coalesce(F.col("country").isin(VALID_COUNTRIES), F.lit(False))),
    RowRule("spend_non_negative", F.coalesce(F.col("lifetime_spend") >= 0, F.lit(False))),
    RowRule("customer_id_unique", F.col("_dup_count") == 1),
]

DATASET_RULES = [
    DatasetRule("min_rows", "row_count", ">=", 1),
    DatasetRule("email_null_rate", "email_null_rate", "<=", 0.20),
    DatasetRule("age_null_rate", "age_null_rate", "<=", 0.20),
    DatasetRule("bad_row_rate", "bad_row_rate", "<=", 0.60),   # loose for the demo
]


def add_duplicate_count(df: DataFrame) -> DataFrame:
    """Count rows per key with a window so every row knows if it is a dup."""
    w = Window.partitionBy("customer_id")
    return df.withColumn("_dup_count", F.count(F.lit(1)).over(w))


def apply_row_rules(df: DataFrame, rules: list[RowRule]) -> DataFrame:
    """Add an array column listing every rule the row failed."""
    failed = F.array_compact(F.array(*[
        F.when(~r.passes, F.lit(r.name)) for r in rules
    ]))
    return df.withColumn("dq_failures", failed)


def profile(df: DataFrame) -> dict:
    """All dataset metrics in ONE scan."""
    row = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.avg(F.col("email").isNull().cast("int")).alias("email_null_rate"),
        F.avg(F.col("age").isNull().cast("int")).alias("age_null_rate"),
        F.avg((F.size("dq_failures") > 0).cast("int")).alias("bad_row_rate"),
        F.countDistinct("customer_id").alias("distinct_ids"),
    ).first()
    return row.asDict()


def evaluate(metrics: dict, rules: list[DatasetRule]) -> list[tuple]:
    results = []
    for r in rules:
        value = metrics[r.metric] or 0.0
        ok = value <= r.threshold if r.op == "<=" else value >= r.threshold
        results.append((r.name, r.metric, float(value), r.op, float(r.threshold), ok))
    return results


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = apply_row_rules(add_duplicate_count(sample_customers(spark)), ROW_RULES).cache()

        good = df.filter(F.size("dq_failures") == 0).drop("_dup_count", "dq_failures")
        quarantine = (df.filter(F.size("dq_failures") > 0)
                      .withColumn("quarantined_at", F.current_timestamp())
                      .drop("_dup_count"))

        print("== good rows ==")
        good.show()
        print("== quarantined rows with reasons ==")
        quarantine.show(truncate=False)

        # Which rule fails most? Great for dashboards.
        df.select(F.explode("dq_failures").alias("rule")).groupBy("rule").count() \
          .orderBy(F.desc("count")).show(truncate=False)

        metrics = profile(df)
        report = spark.createDataFrame(
            evaluate(metrics, DATASET_RULES),
            "rule STRING, metric STRING, value DOUBLE, op STRING, threshold DOUBLE, passed BOOLEAN")
        print("== dataset-level report ==")
        report.show(truncate=False)

        failed = [r.rule for r in report.collect() if not r.passed]
        if failed:
            # In production: don't publish; raise so the orchestrator alerts.
            raise RuntimeError(f"Data quality gate failed: {failed}")
        print("Quality gate passed - safe to publish 'good' rows.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

# Exercises
#   1. Tighten email_null_rate to 0.10 and watch the gate fail. Where would you
#      log the report so you can trend quality over time?
#   2. For duplicates, keep the row with the highest lifetime_spend in "good"
#      and quarantine only the others (hint: row_number in lesson 06).
#   3. Load the rules from a YAML/JSON config instead of Python lists so analysts
#      can change thresholds without code changes.
#
# Interview tips
#   * Nulls in comparisons are NULL, not FALSE: wrap rules with coalesce.
#   * Compute metrics in one agg pass; avoid many count() actions.
#   * Mention tools: Great Expectations, Deequ/PyDeequ, Soda, Delta constraints.
