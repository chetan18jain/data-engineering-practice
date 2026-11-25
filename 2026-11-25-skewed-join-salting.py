"""Skewed join: fix a hot-key join with salting.

Lesson
------
A join on a key with one value that dominates the fact table (here
`channel = "WEB"` is 80% of events) puts nearly all the work on a
single Spark task: the shuffle sends every "WEB" row to one
partition, that task runs for minutes while the others finish in
seconds, and the stage is only as fast as its slowest task. The
classic symptom is one task with a huge shuffle-read while the rest
sit idle.

The fix is salting: split the hot key into N artificial sub-keys on
the large side, and replicate the small side N times, once per salt
bucket. The join key becomes `(channel, salt)` instead of just
`channel`, so "WEB" rows spread evenly across N partitions, while
the replicated dimension rows match each salt value exactly once.
The join result is identical to the plain join — every fact row
still matches exactly one dimension row — because each fact row's
salt picks exactly one of the N copies.

Two rules decide whether salting is the right tool:
  * The small side must be small enough to replicate N times without
    blowing up memory. Salt a join against a 10-row lookup, not
    against another fact table.
  * N is a trade-off: bigger N spreads the skew thinner but copies
    the dimension more times. Start with N = number of cores per
    executor x a few, and watch the max task time drop.

A deterministic salt (`hash(key) % N`) keeps the exercise
reproducible; in production `rand() % N` (or a hash of the fact
row's own key) is more common and spreads slightly better when the
hot key itself is the only join column.

Run:  python 2026-11-25-skewed-join-salting.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (IntegerType, StringType, StructField,
                               StructType)

SALT_BUCKETS = 4


def main():
    spark = (SparkSession.builder.appName("skewed-join-salting")
             .master("local[1]").getOrCreate())
    try:
        events_schema = StructType([
            StructField("event_id", IntegerType()),
            StructField("channel", StringType()),
            StructField("amount", IntegerType()),
        ])
        channels_schema = StructType([
            StructField("channel", StringType()),
            StructField("channel_name", StringType()),
        ])

        # 12 events, 10 of them on the hot key "WEB" — a plain join
        # shuffles all 10 to one partition.
        events = spark.createDataFrame([
            (1, "WEB", 100), (2, "WEB", 150), (3, "WEB", 200),
            (4, "WEB", 120), (5, "WEB", 90), (6, "WEB", 300),
            (7, "WEB", 80), (8, "WEB", 110), (9, "WEB", 140),
            (10, "WEB", 160), (11, "APP", 500), (12, "STORE", 700),
        ], schema=events_schema)
        channels = spark.createDataFrame([
            ("WEB", "Website"), ("APP", "Mobile app"), ("STORE", "Retail"),
        ], schema=channels_schema)

        # 1. Plain join — correct, but skewed: verify what the answer
        #    should be before salting.
        plain = (events.join(channels, "channel")
                 .select("event_id", "channel", "channel_name", "amount"))
        assert plain.count() == 12

        # 2. Salt the large side: deterministic salt from the event's
        #    own key, so the same row always lands in the same bucket.
        salted_events = events.withColumn(
            "salt", F.abs(F.hash("event_id")) % SALT_BUCKETS)

        # 3. Replicate the small side once per salt bucket.
        salt_values = spark.createDataFrame(
            [(i,) for i in range(SALT_BUCKETS)], ["salt"])
        salted_channels = channels.crossJoin(salt_values)

        # 4. Join on (channel, salt): "WEB" rows now spread over 4
        #    partitions instead of 1, and each event still matches
        #    exactly one replicated channel row.
        salted = (salted_events
                  .join(salted_channels, ["channel", "salt"])
                  .select("event_id", "channel", "channel_name", "amount"))

        # 5. Same result as the plain join, row for row.
        plain_rows = sorted(
            (r["event_id"], r["channel_name"], r["amount"])
            for r in plain.collect())
        salted_rows = sorted(
            (r["event_id"], r["channel_name"], r["amount"])
            for r in salted.collect())
        assert salted_rows == plain_rows

        # 6. The skew is actually gone: the hot key is spread evenly.
        spread = (salted_events.filter(F.col("channel") == "WEB")
                  .groupBy("salt").count().collect())
        assert len(spread) == SALT_BUCKETS
        assert all(r["count"] > 0 for r in spread), \
            "a salt bucket got no WEB rows — skew not spread"
        salted.orderBy("event_id").show(truncate=False)
        print("OK: salted join matches the plain join exactly, and the "
              "10 WEB rows are spread across 4 salt buckets instead of "
              "one hot partition")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
