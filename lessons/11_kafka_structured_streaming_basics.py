"""
Lesson 11 - Kafka to Spark Structured Streaming basics

What you'll learn
  * How readStream / writeStream differ from batch read / write
  * Reading a Kafka topic and parsing the binary "value" column as JSON
  * Why every production stream needs a checkpointLocation
  * Triggers: processingTime micro-batches vs availableNow (catch up then stop)
  * Monitoring a running query with lastProgress / recentProgress
  * Restarting from a checkpoint and continuing exactly where you left off

Key concepts (plain English)
  * Structured Streaming treats a stream as a table that keeps growing. You
    write the SAME DataFrame code as batch; Spark runs it again and again on
    each new slice of data (a "micro-batch").
  * Kafka gives Spark rows with key/value as BINARY plus topic, partition,
    offset, timestamp. You cast value to string and parse it with from_json.
  * The checkpoint directory stores which offsets were processed and the
    query state. Kill the job, restart it with the same checkpoint and it
    resumes - no reprocessing, no gaps. Lose the checkpoint = lose your place.
  * A trigger decides WHEN a micro-batch runs. processingTime="10 seconds" is
    the classic always-on job; availableNow=True drains what is there and
    exits (great for cheap "streaming as scheduled batch" jobs).

Real-world use case
  An e-commerce site publishes order events to a Kafka topic "orders". A
  Spark job reads them continuously, cleans and types the fields, and lands
  them as Parquet files in the bronze layer of the lake every few seconds.

Run
  pip install pyspark==3.5.1
  # Local demo mode (no Kafka needed - uses Spark's built-in "rate" source):
  python lessons/11_kafka_structured_streaming_basics.py
  # Kafka mode (connection settings come ONLY from environment variables):
  export KAFKA_BOOTSTRAP_SERVERS=localhost:9092 KAFKA_TOPIC=orders
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      lessons/11_kafka_structured_streaming_basics.py
  # Sample Kafka message value:
  #   {"order_id": 1, "store_id": 3, "amount": 25.5, "status": "NEW"}
"""

import os
import shutil
import tempfile

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")  # None -> demo mode
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "orders")
RUN_SECONDS = int(os.environ.get("DEMO_SECONDS", "15"))

# Always declare the JSON schema explicitly. Schema inference is NOT allowed
# on streams by default, and an explicit contract protects you from upstream
# surprises (unknown fields are ignored, bad values become NULL).
ORDER_SCHEMA = StructType([
    StructField("order_id", LongType()),
    StructField("store_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("status", StringType()),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("lesson11-streaming-basics").master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        # Graceful stop finishes the in-flight micro-batch before shutting down.
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def read_kafka(spark: SparkSession) -> DataFrame:
    """Kafka source -> typed order rows. Requires the spark-sql-kafka package."""
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        # Only used the FIRST time; afterwards the checkpoint decides offsets.
        .option("startingOffsets", "earliest")
        # Cap each micro-batch so a big backlog doesn't blow up memory.
        .option("maxOffsetsPerTrigger", "10000")
        .option("failOnDataLoss", "false")
        .load()
    )
    print("Raw Kafka schema (key/value are binary):")
    raw.printSchema()
    return (
        raw.select(
            F.from_json(F.col("value").cast("string"), ORDER_SCHEMA).alias("o"),
            F.col("timestamp").alias("event_time"),
            "partition", "offset",
        )
        .select("o.*", "event_time", "partition", "offset")
    )


def read_rate_demo(spark: SparkSession) -> DataFrame:
    """Self-contained fake 'orders' stream built from the rate source.

    rate emits (timestamp, value) rows at a fixed speed - perfect for learning
    because it behaves like an unbounded source but needs no infrastructure.
    """
    rate = spark.readStream.format("rate").option("rowsPerSecond", 5).load()
    statuses = F.array(F.lit("NEW"), F.lit("PAID"), F.lit("SHIPPED"), F.lit("bad"))
    return rate.select(
        F.col("value").alias("order_id"),
        (F.col("value") % 5).cast("int").alias("store_id"),
        F.round(F.rand() * 100, 2).alias("amount"),
        F.element_at(statuses, (F.col("value") % 4 + 1).cast("int")).alias("status"),
        F.col("timestamp").alias("event_time"),
        F.lit(0).alias("partition"),
        F.col("value").alias("offset"),
    )


def clean_orders(df: DataFrame) -> DataFrame:
    """Plain DataFrame logic - identical to what you would write for batch."""
    return (
        df.filter(F.col("order_id").isNotNull())
        .withColumn("status", F.upper(F.trim("status")))
        .filter(F.col("status").isin("NEW", "PAID", "SHIPPED"))
        .withColumn("order_date", F.to_date("event_time"))
        .withColumn("ingested_at", F.current_timestamp())
    )


def start_console_query(df: DataFrame, checkpoint: str):
    # Console sink is for learning/debugging only - never in production.
    return (
        df.writeStream.format("console")
        .outputMode("append")  # only new rows; no aggregation, so append is valid
        .option("truncate", "false").option("numRows", 5)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="5 seconds")
        .queryName("orders_console")
        .start()
    )


def start_parquet_query(df: DataFrame, out_path: str, checkpoint: str):
    # File sinks are exactly-once: Spark records committed files in a
    # _spark_metadata log next to the data, tied to the checkpoint.
    return (
        df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", out_path)
        .option("checkpointLocation", checkpoint)
        .partitionBy("order_date")
        .trigger(processingTime="5 seconds")
        .queryName("orders_bronze")
        .start()
    )


def print_progress(query) -> None:
    p = query.lastProgress
    if p:
        print("batchId=%s inputRows=%s rows/sec=%s durationMs=%s" % (
            p["batchId"], p["numInputRows"], p.get("processedRowsPerSecond"),
            p["durationMs"].get("triggerExecution")))


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson11_")
    out_path = os.path.join(base, "bronze_orders")
    ckpt_parquet = os.path.join(base, "chk", "orders_bronze")
    ckpt_console = os.path.join(base, "chk", "orders_console")
    try:
        source = read_kafka(spark) if KAFKA_SERVERS else read_rate_demo(spark)
        orders = clean_orders(source)
        print("isStreaming:", orders.isStreaming)
        orders.printSchema()
        orders.explain()  # look for StreamingRelation / MicroBatchScan in the plan

        # --- Run 1: two queries can share one source DataFrame -------------
        console_q = start_console_query(orders, ckpt_console)
        parquet_q = start_parquet_query(orders, out_path, ckpt_parquet)
        parquet_q.awaitTermination(RUN_SECONDS)
        print_progress(parquet_q)
        first_batch_count = len(parquet_q.recentProgress)
        console_q.stop()
        parquet_q.stop()

        # Batch read of the streaming output: the lake is just Parquet now.
        landed = spark.read.parquet(out_path)
        print("Rows landed after run 1:", landed.count())
        landed.groupBy("store_id").agg(F.count("*").alias("orders"),
                                       F.round(F.sum("amount"), 2).alias("revenue")) \
            .orderBy("store_id").show()

        # --- Run 2: restart with the SAME checkpoint -------------------------
        # Batch ids continue from where run 1 stopped; nothing is re-read.
        parquet_q = start_parquet_query(orders, out_path, ckpt_parquet)
        parquet_q.awaitTermination(RUN_SECONDS // 2)
        print("Progress events in run 1:", first_batch_count)
        print_progress(parquet_q)
        parquet_q.stop()
        print("Rows landed after restart:", spark.read.parquet(out_path).count())
        print("Checkpoint folders:", sorted(os.listdir(ckpt_parquet)))
    finally:
        for q in spark.streams.active:
            q.stop()
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Change the trigger to availableNow=True in Kafka mode and confirm the
#      job processes the backlog and exits by itself.
#   2. Route rows where from_json returned NULL (malformed JSON) to a separate
#      "quarantine" Parquet sink instead of silently dropping them.
#   3. Delete the checkpoint between runs, restart, and explain what happens to
#      offsets and output duplicates with startingOffsets earliest vs latest.
#
# Interview tips
#   * The checkpoint (offsets + commits + state) is what makes restarts safe;
#     one checkpoint per query, never shared between queries.
#   * startingOffsets applies only to a brand-new query with no checkpoint.
#   * Append mode is for rows that never change; aggregations without a
#     watermark need complete/update mode (lesson 12 covers watermarks).
