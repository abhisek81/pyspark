# PySpark for Data Engineering — Learning Path

Hands-on, heavily commented PySpark scripts focused on real data engineering pipelines. A new lesson is added to the lessons/ folder every 12 hours. Each lesson explains the concepts, includes runnable code (most use self-generated sample data so you can run them locally), and ends with exercises and interview tips.

## How to run a lesson locally

    pip install pyspark==3.5.1
    python lessons/01_<topic>.py
    # or
    spark-submit lessons/01_<topic>.py

Lessons that use Kafka, Delta Lake, Snowflake, JDBC, or Iceberg list the exact spark-submit --packages needed in their docstring. Connection settings are always read from environment variables; no credentials are stored in this repo.

## Curriculum

- [ ] 01 Batch ETL: CSV to cleaned Parquet with explicit schema enforcement
- [ ] 02 Nested JSON ingestion and flattening (explode, struct access)
- [ ] 03 Incremental JDBC extract (Postgres) using a high-watermark column
- [ ] 04 Partitioned data lake writes (partitionBy, overwrite modes)
- [ ] 05 Data quality checks: nulls, ranges, uniqueness, quarantine bad records
- [ ] 06 Deduplication strategies (dropDuplicates vs window row_number)
- [ ] 07 SCD Type 1 upsert with Delta Lake MERGE
- [ ] 08 SCD Type 2 history tracking with Delta Lake MERGE
- [ ] 09 Medallion architecture: bronze to silver to gold pipeline
- [ ] 10 Schema evolution and mergeSchema handling
- [ ] 11 Kafka to Spark Structured Streaming basics (readStream, checkpoints, triggers)
- [ ] 12 Streaming windowed aggregations with watermarks and late data
- [ ] 13 Stream-static join for enrichment (dimension lookup)
- [ ] 14 Stream-stream join with watermarks
- [ ] 15 CDC processing: Debezium events from Kafka applied to a Delta table
- [ ] 16 Idempotent / exactly-once sinks with foreachBatch
- [ ] 17 Dead-letter queue pattern for bad streaming records
- [ ] 18 Snowflake read/write with the Spark connector and query pushdown
- [ ] 19 Sessionization of clickstream events
- [ ] 20 Broadcast joins and Adaptive Query Execution tuning
- [ ] 21 Handling data skew with salting
- [ ] 22 Small-file problem: compaction, coalesce/repartition, OPTIMIZE / Z-ORDER
- [ ] 23 Delta Lake time travel, VACUUM, table history
- [ ] 24 Apache Iceberg tables with Spark
- [ ] 25 Pandas UDFs vs regular UDFs vs built-in functions
- [ ] 26 Config-driven pipeline framework (YAML/JSON sources, transforms, sinks)
- [ ] 27 Logging, metrics, and error handling in production jobs
- [ ] 28 Unit testing PySpark transformations with pytest
- [ ] 29 Source-to-target data reconciliation job
- [ ] 30 Capstone: Kafka to bronze/silver/gold Delta to Snowflake, orchestrated by Airflow
