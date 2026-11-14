"""
Lesson 06 - Deduplication strategies (dropDuplicates vs window row_number)

What you'll learn
  * distinct() vs dropDuplicates(subset) and why dropDuplicates is NOT deterministic
  * Deterministic "keep latest per key" with row_number() over a window
  * Tie-breaking rules so re-runs always produce the same answer
  * max_by / struct-max trick: dedup without a window
  * Exact duplicates vs business-key duplicates vs near-duplicates

Key concepts (plain English)
  * Pipelines receive duplicates all the time: retries, at-least-once Kafka
    delivery, overlapping incremental extracts (lesson 03).
  * dropDuplicates(["id"]) keeps "some" row per id - whichever Spark sees first,
    which depends on partitioning. Two runs can keep different rows. That is a
    silent correctness bug.
  * row_number() over (partition by id order by updated_at desc, <tiebreaker>)
    numbers rows within each id; keep rn == 1. The ORDER BY decides the winner,
    so the result is reproducible.

Real-world use case
  The raw orders zone contains several versions of each order (status changes,
  retries). The silver table must contain exactly one, the latest, per order.

Run
  python lessons/06_deduplication_strategies.py
"""

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F


def build_spark() -> SparkSession:
    return (SparkSession.builder.appName("lesson06-dedup")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


def sample_raw_orders(spark: SparkSession) -> DataFrame:
    rows = [
        # order_id, status, amount, updated_at, ingest_seq (arrival order)
        ("O-1", "CREATED", 100.0, "2026-11-01 10:00:00", 1),
        ("O-1", "PAID", 100.0, "2026-11-01 10:05:00", 2),
        ("O-1", "PAID", 100.0, "2026-11-01 10:05:00", 3),     # exact duplicate (retry)
        ("O-1", "SHIPPED", 100.0, "2026-11-02 08:00:00", 4),
        ("O-2", "CREATED", 50.0, "2026-11-01 11:00:00", 5),
        ("O-2", "CANCELLED", 50.0, "2026-11-01 12:00:00", 6),
        ("O-2", "REFUNDED", 50.0, "2026-11-01 12:00:00", 7),  # same timestamp! tie
        ("O-3", "CREATED", 75.0, "2026-11-03 09:00:00", 8),
    ]
    return (spark.createDataFrame(rows, "order_id STRING, status STRING, amount DOUBLE, updated_at STRING, ingest_seq LONG")
            .withColumn("updated_at", F.to_timestamp("updated_at")))


def exact_dedup(df: DataFrame) -> DataFrame:
    """Remove rows identical in all business columns (ignores ingest_seq)."""
    return df.dropDuplicates(["order_id", "status", "amount", "updated_at"])


def latest_per_key_window(df: DataFrame) -> DataFrame:
    """The production-grade pattern: deterministic latest-wins."""
    w = Window.partitionBy("order_id").orderBy(
        F.col("updated_at").desc(),
        F.col("ingest_seq").desc(),      # tiebreaker: later arrival wins
    )
    return (df.withColumn("rn", F.row_number().over(w))
              .filter("rn = 1")
              .drop("rn"))


def latest_per_key_max_by(df: DataFrame) -> DataFrame:
    """Same result without a window, using max over a struct.

    Structs compare field by field, left to right, so max(struct(updated_at,
    ingest_seq, ...)) picks the latest row with the same tiebreak. Often
    cheaper than a window because it can partially aggregate before shuffle.
    """
    latest = F.max(F.struct("updated_at", "ingest_seq", "status", "amount")).alias("r")
    return (df.groupBy("order_id").agg(latest)
              .select("order_id", "r.status", "r.amount", "r.updated_at", "r.ingest_seq"))


def flag_duplicates(df: DataFrame) -> DataFrame:
    """Keep all rows but mark which are duplicates - useful for auditing."""
    w = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc(), F.col("ingest_seq").desc())
    return (df.withColumn("version_rank", F.row_number().over(w))
              .withColumn("versions", F.count(F.lit(1)).over(Window.partitionBy("order_id")))
              .withColumn("is_latest", F.col("version_rank") == 1))


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = sample_raw_orders(spark)
        print("== raw ==")
        raw.orderBy("ingest_seq").show(truncate=False)

        print("== exact duplicates removed ==")
        exact_dedup(raw).orderBy("order_id", "updated_at").show(truncate=False)

        print("== dropDuplicates(['order_id']) - arbitrary winner, DON'T use for latest ==")
        raw.dropDuplicates(["order_id"]).show(truncate=False)

        print("== window row_number - deterministic latest ==")
        a = latest_per_key_window(raw)
        a.orderBy("order_id").show(truncate=False)

        print("== struct max - same answer, no window ==")
        b = latest_per_key_max_by(raw)
        b.orderBy("order_id").show(truncate=False)

        # Prove both approaches agree (exceptAll = multiset difference).
        same = a.select(b.columns).exceptAll(b).count() == 0
        print("window and struct-max agree:", same)

        print("== audit view ==")
        flag_duplicates(raw).orderBy("order_id", "version_rank").show(truncate=False)

        # Look at the plans: window -> Exchange + Sort + Window;
        # struct max -> partial HashAggregate before the Exchange.
        a.explain()
        b.explain()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

# Exercises
#   1. Replace row_number with rank() and dense_rank(). What happens to O-2's tie?
#   2. Deduplicate near-duplicates: same customer + same amount within 2 minutes
#      (hint: lag() over a window ordered by time).
#   3. In streaming, use dropDuplicatesWithinWatermark (Spark 3.5) on event_id
#      and explain why a watermark is needed to bound state.
#
# Interview tips
#   * row_number vs rank vs dense_rank - classic question; know tie behavior.
#   * Always add a tiebreaker column to make dedup deterministic.
#   * dropDuplicates keeps an arbitrary row; fine for exact dupes only.
