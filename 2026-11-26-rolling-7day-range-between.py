"""Rolling 7-day revenue: rowsBetween is NOT a 7-day window.

Lesson
------
A "rolling 7 days" is a window over *values* (dates), not over *rows*.
``rowsBetween(-6, 0)`` takes the current row plus the 6 rows before it,
whatever dates they carry — so if a store had no sales for a few days,
or had a burst of old rows, the "7-day" total silently covers the
wrong period. ``rangeBetween`` on a numeric ordering key (here the
Unix timestamp of the date) takes every row whose value falls in
``[current - 6 days, current]``, which is the real calendar window
even when dates are missing.

The two frames agree only when there is exactly one row per day with
no gaps. Sparse data — closed days, no orders, late files — is the
normal case in production, so a rolling-days metric built on
``rowsBetween`` drifts a little more every time a date is missing,
and the drift is invisible in the output: the numbers still look
plausible, just wrong.

Rules of thumb:
  * Use ``rangeBetween`` (on a timestamp / epoch ordering key) for any
    window defined in time: rolling 7 days, trailing 30 days, and so on.
  * Use ``rowsBetween`` only for windows defined in row counts: "the
    last 10 events", "the previous 3 orders".
  * ``rangeBetween`` needs a numeric or timestamp ordering column; for
    a date column, order by ``unix_timestamp(date)`` and express the
    range in seconds (6 days = 6 * 86400).

Run:  python 2026-11-26-rolling-7day-range-between.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (IntegerType, StringType, StructField,
                               StructType)
from pyspark.sql.window import Window

SEVEN_DAYS_SECONDS = 6 * 86400


def main():
    spark = (SparkSession.builder.appName("rolling-7day-range-between")
             .master("local[1]").getOrCreate())
    try:
        schema = StructType([
            StructField("store", StringType()),
            StructField("sale_date", StringType()),
            StructField("revenue", IntegerType()),
        ])

        # Store A has gaps (no sales 2026-10-21..24, 2026-10-26..31,
        # 2026-11-03) and two old rows outside any 7-day window ending
        # in November. Store B sells every day, so the two frames
        # agree there — the control case.
        sales = spark.createDataFrame([
            ("A", "2026-10-20", 500), ("A", "2026-10-25", 300),
            ("A", "2026-11-01", 100), ("A", "2026-11-02", 50),
            ("A", "2026-11-04", 200), ("A", "2026-11-05", 80),
            ("A", "2026-11-06", 120),
            ("B", "2026-11-01", 10), ("B", "2026-11-02", 20),
            ("B", "2026-11-03", 30), ("B", "2026-11-04", 40),
            ("B", "2026-11-05", 50), ("B", "2026-11-06", 60),
            ("B", "2026-11-07", 70),
        ], schema=schema)

        rows_window = (Window.partitionBy("store")
                       .orderBy("sale_date").rowsBetween(-6, 0))
        # Order by the epoch seconds of the date so the range is a
        # real time interval, not a row count.
        range_window = (Window.partitionBy("store")
                        .orderBy(F.unix_timestamp("sale_date"))
                        .rangeBetween(-SEVEN_DAYS_SECONDS, 0))

        result = (sales
                  .withColumn("rolling_rows",
                              F.sum("revenue").over(rows_window))
                  .withColumn("rolling_7d",
                              F.sum("revenue").over(range_window)))

        # 1. On 2026-11-04, store A's last 7 rows reach back to
        #    2026-10-20 (500) and 2026-10-25 (300) — both outside the
        #    7 calendar days ending 2026-11-04.
        a_nov4 = result.filter(
            (F.col("store") == "A")
            & (F.col("sale_date") == "2026-11-04")).collect()[0]
        assert a_nov4["rolling_rows"] == 500 + 300 + 100 + 50 + 200, \
            a_nov4["rolling_rows"]
        assert a_nov4["rolling_7d"] == 100 + 50 + 200, a_nov4["rolling_7d"]
        assert a_nov4["rolling_rows"] != a_nov4["rolling_7d"], \
            "frames should disagree on gappy data"

        # 2. On 2026-11-06 the 7-day window is 2026-10-31..11-06, so
        #    the October rows drop out of the range frame — but store A
        #    has only 7 rows in total, so they are still inside the
        #    last-7-rows frame. The drift persists until enough new
        #    rows push the stale ones out of the row window.
        a_nov6 = result.filter(
            (F.col("store") == "A")
            & (F.col("sale_date") == "2026-11-06")).collect()[0]
        assert a_nov6["rolling_7d"] == 100 + 50 + 200 + 80 + 120, \
            a_nov6["rolling_7d"]
        assert a_nov6["rolling_rows"] == \
            500 + 300 + 100 + 50 + 200 + 80 + 120, a_nov6["rolling_rows"]
        assert a_nov6["rolling_rows"] != a_nov6["rolling_7d"], \
            "row frame still carries the stale October rows"

        # 3. Store B sells every day: one row per day, no gaps, so the
        #    two frames must match on every date.
        mismatch = result.filter(
            (F.col("store") == "B")
            & (F.col("rolling_rows") != F.col("rolling_7d"))).count()
        assert mismatch == 0, "frames must agree on gap-free data"

        result.orderBy("store", "sale_date").show(truncate=False)
        print("OK: rowsBetween over-counted store A on 2026-11-04 "
              "(1150 vs the true 7-day 350) and still on 2026-11-06 "
              "(1350 vs 550); rangeBetween on the epoch timestamp "
              "gives the real calendar window, and both frames agree "
              "where dates have no gaps")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
