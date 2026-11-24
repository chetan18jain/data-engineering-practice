"""Late-arriving events: fix the strict-watermark boundary with an overlap window.

Lesson
------
An incremental loader that takes rows with `updated_at > watermark`
silently LOSES any row stamped exactly at the current max — the
watermark never moves backwards, so that row is never selected again.
It also loses genuinely late rows that arrive after their timestamp
has already passed the watermark. Both are the same root cause: the
watermark is a hard cut-off, and real data is never that tidy.

The production fix is an overlap (lookback) window: reprocess the
last N hours on every run instead of only rows past the watermark.
The watermark still decides what is *new*, but the batch window
reaches back before it, so boundary rows and late arrivals are
re-extracted and re-merged. The MERGE on primary keys makes the
reprocessing safe — re-merged rows update in place instead of
duplicating — and the dedupe keeps the latest version per key, so a
corrected late row wins over the stale one already in the target.

The cost is constant reprocessing: every run re-reads the last N
hours. Pick N from how late your data actually arrives (hours for
most sources, a day for flaky ones), not from gut feel.

Run:  python 2026-11-24-late-arriving-overlap-window.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (IntegerType, StringType, StructField,
                               StructType, TimestampType)
from pyspark.sql.window import Window


def main():
    spark = (SparkSession.builder.appName("late-arriving-overlap-window")
             .master("local[1]").getOrCreate())
    try:
        schema = StructType([
            StructField("order_id", StringType()),
            StructField("status", StringType()),
            StructField("amount", IntegerType()),
            StructField("updated_at", TimestampType()),
        ])
        # Target as of the last run: watermark = max(updated_at) = 10:00.
        # O2 was already loaded at 10:00 with status "paid".
        target = spark.createDataFrame([
            ("O1", "paid", 499, "2026-11-24 09:00:00"),
            ("O2", "paid", 1299, "2026-11-24 10:00:00"),
        ], schema=schema)

        # Source now: O2 corrected at exactly 10:00 (boundary row —
        # strict `>` would skip it forever), O3 arrived late with a
        # 09:30 stamp (already behind the watermark), O4 is genuinely new.
        source = spark.createDataFrame([
            ("O2", "refunded", 1299, "2026-11-24 10:00:00"),
            ("O3", "paid", 799, "2026-11-24 09:30:00"),
            ("O4", "paid", 199, "2026-11-24 11:00:00"),
        ], schema=schema)

        watermark = target.agg(F.max("updated_at")).collect()[0][0]
        overlap_hours = 2  # reprocess [watermark - 2h, now]

        # 1. Strict `>` batch: only O4. O2 (boundary) and O3 (late) lost.
        strict_batch = source.filter(F.col("updated_at") > F.lit(watermark))
        assert sorted(r["order_id"] for r in strict_batch.collect()) == ["O4"]

        # 2. Overlap batch: everything at/after watermark - 2h.
        from datetime import timedelta
        window_start = watermark - timedelta(hours=overlap_hours)
        overlap_batch = source.filter(F.col("updated_at") >= window_start)
        assert sorted(r["order_id"] for r in overlap_batch.collect()) == ["O2", "O3", "O4"]

        # 3. Dedupe keeping the LATEST row per key (a corrected late row
        #    must win over the stale version already in the target).
        #    Simulate the MERGE by unioning target + batch, then
        #    row_number() over the key ordered by updated_at desc.
        merged = (target.unionByName(overlap_batch)
                  .withColumn("rn", F.row_number().over(
                      Window.partitionBy("order_id")
                            .orderBy(F.col("updated_at").desc())))
                  .filter(F.col("rn") == 1).drop("rn"))
        by_id = {r["order_id"]: r for r in merged.collect()}

        assert by_id["O2"]["status"] == "refunded"  # boundary row recovered
        assert by_id["O3"]["amount"] == 799         # late row recovered
        assert by_id["O4"]["amount"] == 199         # new row loaded
        assert len(by_id) == 4
        merged.orderBy("order_id").show(truncate=False)
        print("OK: overlap window recovered the boundary + late rows; "
              "strict `>` would have lost O2 and O3")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
