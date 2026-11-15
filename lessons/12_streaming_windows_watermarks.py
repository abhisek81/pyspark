"""
Lesson 12 - Streaming windowed aggregations with watermarks and late data

What you'll learn
  * Event time vs processing time, and why aggregations must use event time
  * Tumbling windows (fixed, non-overlapping) and sliding windows (overlapping)
  * withWatermark: how Spark decides a window is "done" and frees its state
  * Output modes for aggregations: update vs append (and why append needs a watermark)
  * Measuring late data with numRowsDroppedByWatermark in query progress

Key concepts (plain English)
  * Event time = when the thing happened (inside the message). Processing
    time = when Spark saw it. Networks and retries make events arrive late
    and out of order, so "orders per minute" must group by event time.
  * A window groups events into time buckets: window("event_time", "10 seconds")
    gives buckets 12:00:00-12:00:10, 12:00:10-12:00:20, ... A sliding window
    ("10 seconds", "5 seconds") makes each event land in two buckets.
  * Without a watermark Spark must keep EVERY window in state forever, because
    a late event could still arrive. State grows until the job dies.
  * withWatermark("event_time", "20 seconds") says: "I accept events up to 20s
    later than the newest event time I have seen". Watermark = max(event_time)
    - 20s. Windows ending before the watermark are finalised, their state is
    dropped, and events older than the watermark are discarded.
  * update mode emits a window every time its count changes (fast, but rows
    get revised). append mode emits a window ONCE, only after the watermark
    passes its end (final, but delayed by the watermark threshold).

Real-world use case
  A payments platform computes revenue per merchant per minute for a live
  dashboard and a fraud model. Mobile clients buffer events offline, so some
  arrive minutes late. A 10 minute watermark keeps the numbers correct for
  normal lateness while keeping state (memory) bounded.

Run
  pip install pyspark==3.5.1
  # Local demo mode (no Kafka needed - rate source with simulated late events):
  python lessons/12_streaming_windows_watermarks.py
  # Kafka mode (connection settings come ONLY from environment variables):
  export KAFKA_BOOTSTRAP_SERVERS=localhost:9092 KAFKA_TOPIC=payments
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      lessons/12_streaming_windows_watermarks.py
  # Sample Kafka message value:
  #   {"payment_id": 7, "merchant_id": 2, "amount": 19.99,
  #    "event_time": "2026-11-15T10:00:03"}
"""

import os
import shutil
import tempfile
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StructField, StructType, TimestampType

KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")  # None -> demo mode
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "payments")
RUN_SECONDS = int(os.environ.get("DEMO_SECONDS", "40"))
WATERMARK = os.environ.get("WATERMARK_DELAY", "20 seconds")

PAYMENT_SCHEMA = StructType([
    StructField("payment_id", LongType()),
    StructField("merchant_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("event_time", TimestampType()),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("lesson12-windows-watermarks").master("local[*]")
        # Each shuffle partition holds its own state store; keep it small locally.
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def read_kafka(spark: SparkSession) -> DataFrame:
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )
    # Note: we use the event_time INSIDE the payload, not Kafka's own
    # timestamp column (that one is closer to processing time).
    return raw.select(
        F.from_json(F.col("value").cast("string"), PAYMENT_SCHEMA).alias("p")
    ).select("p.*")


def read_rate_demo(spark: SparkSession) -> DataFrame:
    """Fake payments where every 10th event is 60 seconds late.

    Normal events are 0-5s behind the clock (realistic jitter). The very
    late ones are older than the 20s watermark, so Spark will drop them once
    the watermark has moved past their window - exactly what we want to see.
    """
    rate = spark.readStream.format("rate").option("rowsPerSecond", 10).load()
    lateness = (
        F.when(F.col("value") % 10 == 0, F.lit(60))
        .otherwise((F.rand() * 5).cast("int"))
    )
    return rate.select(
        F.col("value").alias("payment_id"),
        (F.col("value") % 3).cast("int").alias("merchant_id"),
        F.round(F.rand() * 50 + 1, 2).alias("amount"),
        (F.col("timestamp") - F.make_interval(secs=lateness)).alias("event_time"),
    )


def tumbling_revenue(payments: DataFrame) -> DataFrame:
    """10s tumbling window per merchant, bounded by a watermark.

    withWatermark MUST be applied to the same column used in window(), and
    BEFORE the aggregation, otherwise Spark ignores it for state cleanup.
    """
    return (
        payments.withWatermark("event_time", WATERMARK)
        .groupBy(F.window("event_time", "10 seconds"), "merchant_id")
        .agg(F.count("*").alias("payments"), F.round(F.sum("amount"), 2).alias("revenue"))
        .select(F.col("window.start").alias("win_start"), F.col("window.end").alias("win_end"),
                "merchant_id", "payments", "revenue")
    )


def sliding_counts(payments: DataFrame) -> DataFrame:
    """20s windows sliding every 10s: a moving count, handy for alerting."""
    return (
        payments.withWatermark("event_time", WATERMARK)
        .groupBy(F.window("event_time", "20 seconds", "10 seconds"))
        .agg(F.count("*").alias("payments"))
        .select(F.col("window.start").alias("win_start"), F.col("window.end").alias("win_end"), "payments")
    )


def start_memory_query(df: DataFrame, name: str, mode: str, checkpoint: str):
    # The memory sink stores results in an in-driver table we can query with
    # SQL. Learning only - it is not fault tolerant and lives in driver RAM.
    return (
        df.writeStream.format("memory").queryName(name)
        .outputMode(mode)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="5 seconds")
        .start()
    )


def report_watermark(query) -> None:
    """Progress JSON tells you the current watermark and how much was dropped."""
    p = query.lastProgress
    if not p:
        return
    ops = p.get("stateOperators") or [{}]
    print("[%s] batch=%s watermark=%s stateRows=%s droppedLate=%s" % (
        query.name, p["batchId"], p.get("eventTime", {}).get("watermark"),
        ops[0].get("numRowsTotal"), ops[0].get("numRowsDroppedByWatermark")))


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson12_")
    try:
        payments = read_kafka(spark) if KAFKA_SERVERS else read_rate_demo(spark)
        tumbling = tumbling_revenue(payments)
        tumbling.explain()  # look for EventTimeWatermark and StateStoreSave

        # update: rows for a window re-appear each time the count changes.
        upd_q = start_memory_query(tumbling, "rev_update", "update", os.path.join(base, "chk_upd"))
        # append: a window appears exactly once, after watermark > window end.
        app_q = start_memory_query(tumbling, "rev_append", "append", os.path.join(base, "chk_app"))
        slide_q = start_memory_query(sliding_counts(payments), "slide", "update",
                                     os.path.join(base, "chk_slide"))

        deadline = time.time() + RUN_SECONDS
        while time.time() < deadline:
            time.sleep(10)
            for q in (upd_q, app_q, slide_q):
                report_watermark(q)

        print("UPDATE mode: same window may appear several times (revisions):")
        spark.sql("SELECT win_start, merchant_id, COUNT(*) AS versions, MAX(payments) AS final_payments "
                  "FROM rev_update GROUP BY win_start, merchant_id ORDER BY win_start, merchant_id").show(10, False)
        print("APPEND mode: one final row per window, only for closed windows:")
        spark.sql("SELECT * FROM rev_append ORDER BY win_start, merchant_id").show(10, False)
        print("Sliding windows overlap, so each event is counted in two rows:")
        spark.sql("SELECT win_start, win_end, MAX(payments) AS payments FROM slide "
                  "GROUP BY win_start, win_end ORDER BY win_start").show(10, False)
    finally:
        for q in spark.streams.active:
            q.stop()
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Set WATERMARK_DELAY to "90 seconds" and rerun. Does droppedLate go to
#      zero? What happens to the delay before rows show up in rev_append?
#   2. Add a session window (F.session_window("event_time", "15 seconds")) per
#      merchant and compare its output with the tumbling window.
#   3. Instead of losing late events, also write payments older than the
#      current watermark to a "late_payments" Parquet sink (hint: compare
#      event_time with max(event_time) inside foreachBatch) for a nightly
#      batch correction job.
#
# Interview tips
#   * The watermark is a threshold for state cleanup, not a hard guarantee:
#     Spark keeps data within the threshold and MAY still accept slightly
#     older data until state is purged. Never promise "drops exactly at Xs".
#   * Append mode with an aggregation requires a watermark; complete mode
#     ignores it and keeps all state (fine for tiny cardinality only).
#   * With several streams/watermarks Spark uses the MIN watermark by default
#     (spark.sql.streaming.multipleWatermarkPolicy=min) to avoid dropping data.
