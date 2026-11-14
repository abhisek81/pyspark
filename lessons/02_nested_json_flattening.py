"""
Lesson 02 - Nested JSON ingestion and flattening (explode, struct access)

What you'll learn
  * Reading multi-line / nested JSON with an explicit schema
  * Accessing struct fields with dot notation and getField
  * explode vs explode_outer vs posexplode for arrays
  * Handling maps (key/value) with explode on MapType
  * Writing a generic "flatten any struct" helper used in real pipelines

Key concepts (plain English)
  * APIs and event streams send JSON objects inside objects (structs) and lists
    (arrays). Warehouses and BI tools want flat tables.
  * A struct is like a row inside a column: customer.address.city.
  * explode turns one row with an array of N items into N rows. explode_outer
    keeps the parent row (with null) when the array is empty or null, which is
    usually what you want so you don't silently drop orders with no items.

Real-world use case
  An e-commerce API returns orders with a nested customer object, an array of
  line items, and a free-form attributes map. You must produce two flat tables:
  orders (one row per order) and order_items (one row per line item).

Run
  pip install pyspark==3.5.1
  python lessons/02_nested_json_flattening.py
"""

import json
import os
import shutil
import tempfile

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    ArrayType, MapType,
)

SAMPLE_ORDERS = [
    {
        "order_id": "O-1",
        "customer": {"id": "C01", "name": "Asha",
                     "address": {"city": "Pune", "country": "IN"}},
        "items": [{"sku": "SKU-1", "qty": 2, "price": 10.0},
                  {"sku": "SKU-2", "qty": 1, "price": 99.5}],
        "attributes": {"channel": "web", "coupon": "DIWALI10"},
    },
    {
        "order_id": "O-2",
        "customer": {"id": "C02", "name": "Ben",
                     "address": {"city": "Austin", "country": "US"}},
        "items": [],                      # empty array -> explode would drop it!
        "attributes": {"channel": "app"},
    },
    {
        "order_id": "O-3",
        "customer": {"id": "C03", "name": "Chen", "address": None},
        "items": [{"sku": "SKU-3", "qty": 5, "price": 1.25}],
        "attributes": None,
    },
]

ITEM = StructType([
    StructField("sku", StringType()),
    StructField("qty", IntegerType()),
    StructField("price", DoubleType()),
])
ORDER_SCHEMA = StructType([
    StructField("order_id", StringType()),
    StructField("customer", StructType([
        StructField("id", StringType()),
        StructField("name", StringType()),
        StructField("address", StructType([
            StructField("city", StringType()),
            StructField("country", StringType()),
        ])),
    ])),
    StructField("items", ArrayType(ITEM)),
    StructField("attributes", MapType(StringType(), StringType())),
])


def build_spark() -> SparkSession:
    return (SparkSession.builder.appName("lesson02-nested-json")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


def write_sample(base_dir: str) -> str:
    path = os.path.join(base_dir, "orders.json")
    # JSON Lines: one object per line - the format Spark reads by default.
    with open(path, "w") as f:
        for o in SAMPLE_ORDERS:
            f.write(json.dumps(o) + "\n")
    return path


def read_orders(spark: SparkSession, path: str) -> DataFrame:
    # For a single pretty-printed JSON document use .option("multiLine", "true").
    return spark.read.schema(ORDER_SCHEMA).json(path)


def build_orders_table(df: DataFrame) -> DataFrame:
    """One row per order: pull struct fields up, pick known map keys."""
    return df.select(
        "order_id",
        F.col("customer.id").alias("customer_id"),         # dot notation
        F.col("customer.name").alias("customer_name"),
        # Null-safe: if address is null, city is simply null (no error).
        F.col("customer.address.city").alias("city"),
        F.col("customer").getField("address").getField("country").alias("country"),
        # Map lookup by key. Missing key -> null.
        F.col("attributes")["channel"].alias("channel"),
        F.col("attributes")["coupon"].alias("coupon"),
        F.size("items").alias("item_count"),               # -1 if items is null
    )


def build_items_table(df: DataFrame) -> DataFrame:
    """One row per line item. posexplode_outer also gives the line number."""
    return (
        df.select("order_id", F.posexplode_outer("items").alias("line_no", "item"))
        .select(
            "order_id",
            (F.col("line_no") + 1).alias("line_no"),
            "item.sku", "item.qty", "item.price",
            F.round(F.col("item.qty") * F.col("item.price"), 2).alias("line_total"),
        )
    )


def explode_vs_outer(df: DataFrame) -> None:
    print("explode       ->", df.select(F.explode("items")).count(), "rows (O-2 lost!)")
    print("explode_outer ->", df.select(F.explode_outer("items")).count(), "rows")


def flatten(df: DataFrame, sep: str = "_") -> DataFrame:
    """Generic helper: recursively flatten every struct column.

    Arrays and maps are left alone - exploding them changes the grain of the
    table, which must be a deliberate decision, never automatic.
    """
    while True:
        struct_cols = [f for f in df.schema.fields if isinstance(f.dataType, StructType)]
        if not struct_cols:
            return df
        cols = []
        for field in df.schema.fields:
            if isinstance(field.dataType, StructType):
                for child in field.dataType.fields:
                    cols.append(F.col(f"{field.name}.{child.name}")
                                .alias(f"{field.name}{sep}{child.name}"))
            else:
                cols.append(F.col(field.name))
        df = df.select(cols)


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    base = tempfile.mkdtemp(prefix="lesson02_")
    try:
        raw = read_orders(spark, write_sample(base))
        raw.printSchema()

        explode_vs_outer(raw)

        print("== orders ==")
        build_orders_table(raw).show(truncate=False)
        print("== order_items ==")
        build_items_table(raw).show(truncate=False)

        print("== generic flatten (structs only) ==")
        flat = flatten(raw)
        flat.printSchema()
        flat.show(truncate=False)

        # Map -> rows: useful when attribute keys are unknown in advance.
        print("== attributes as key/value rows ==")
        raw.select("order_id", F.explode_outer("attributes").alias("key", "value")).show()
    finally:
        spark.stop()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

# Exercises
#   1. Add a nested array inside each item (e.g. "discounts": [{"code","amount"}])
#      and produce an item_discounts table. Watch the row counts at each step.
#   2. Pivot the attributes map into columns for ALL keys found in the data
#      (hint: collect distinct keys first, then build columns dynamically).
#   3. Extend flatten() to also explode arrays but only when a flag is passed,
#      and log the row count before/after each explode.
#
# Interview tips
#   * explode drops rows with null/empty arrays; explode_outer keeps them.
#   * Exploding multiple arrays in one select is not allowed; chained explodes
#     create a cartesian product - explain grain before and after.
#   * Always supply a schema for JSON in production: inference scans everything
#     and a single odd record can change a column's type.
