"""Day 1 — Dedupe a batch keeping the LATEST row per primary key.

Problem (from the airflow-metadata-driven-etl loader): if one extract
batch contains the same primary key twice, Delta MERGE fails with a
multiple-source-match error — it cannot know which source row should
win. Plain dropDuplicates(pk) stops the crash but keeps an ARBITRARY
row, which can resurrect a stale version of the record.

Fix: rank rows within each key by updated_at descending with a window
function and keep rank 1. Deterministic: the newest version always wins.

Run: spark-submit 2026-11-19-dedupe-keep-latest.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("dedupe-keep-latest").getOrCreate()

# The same order_id appears twice: a stale "shipped" row and the newer
# "delivered" row. A MERGE on order_id would fail on this batch as-is.
batch = spark.createDataFrame(
    [
        (101, "shipped",   "2026-11-19 09:15:00"),
        (102, "placed",    "2026-11-19 09:20:00"),
        (101, "delivered", "2026-11-19 11:40:00"),
        (103, "placed",    "2026-11-19 10:05:00"),
    ],
    ["order_id", "status", "updated_at"],
)

latest_first = Window.partitionBy("order_id").orderBy(
    F.col("updated_at").desc())

deduped = (
    batch.withColumn("rn", F.row_number().over(latest_first))
         .filter(F.col("rn") == 1)
         .drop("rn")
         .orderBy("order_id")
)

deduped.show()
# +--------+---------+-------------------+
# |order_id|status   |updated_at         |
# +--------+---------+-------------------+
# |101     |delivered|2026-11-19 11:40:00|
# |102     |placed   |2026-11-19 09:20:00|
# |103     |placed   |2026-11-19 10:05:00|
# +--------+---------+-------------------+

assert deduped.count() == 3
row101 = deduped.filter(F.col("order_id") == 101).first()
assert row101["status"] == "delivered", "the newest version must win"
print("OK: 4 batch rows -> 3 keys, newest version kept per key")

spark.stop()
