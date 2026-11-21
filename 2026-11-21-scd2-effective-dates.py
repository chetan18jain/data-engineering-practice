"""Day 3 — Slowly Changing Dimension Type 2 with lead().

Problem: a source system sends one row per change to a customer's
address, each stamped with the date it took effect. A reporting
dimension must keep every version: each row needs a valid_from,
a valid_to (the next change's date, or 9999-12-31 for the current
row), and an is_current flag. A self-join to find "the next change"
is the expensive way to do it.

Fix: lead() over a window partitioned by customer and ordered by
effective_date gives the next change's date in the same row, in one
pass. valid_to is that next date (or the open-ended sentinel), and
is_current is simply "valid_to is the sentinel". No join, no shuffle
beyond the window partition.

Run: spark-submit 2026-11-21-scd2-effective-dates.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("scd2-effective-dates").getOrCreate()

changes = spark.createDataFrame(
    [
        # (customer_id, city, effective_date)
        ("c1", "Hyderabad", "2024-01-15"),
        ("c1", "Bengaluru", "2025-03-01"),  # moved
        ("c1", "Pune",      "2026-06-10"),  # moved again
        ("c2", "Mumbai",    "2023-05-20"),
        ("c2", "Chennai",   "2026-01-05"),  # moved
        ("c3", "Delhi",     "2025-11-11"),  # never moved
    ],
    ["customer_id", "city", "effective_date"],
).withColumn("effective_date", F.to_date("effective_date"))

customer_window = Window.partitionBy("customer_id").orderBy("effective_date")

OPEN_ENDED = "9999-12-31"

scd2 = (
    changes
    .withColumn("next_effective",
                F.lead("effective_date").over(customer_window))
    .withColumn("valid_from", F.col("effective_date"))
    .withColumn(
        "valid_to",
        F.when(F.col("next_effective").isNull(), F.lit(OPEN_ENDED))
         .otherwise(F.col("next_effective")),
    )
    .withColumn(
        "is_current",
        F.when(F.col("next_effective").isNull(), F.lit(True))
         .otherwise(F.lit(False)),
    )
    .drop("effective_date", "next_effective")
    .orderBy("customer_id", "valid_from")
)

scd2.show(truncate=False)
# +-----------+---------+----------+----------+----------+
# |customer_id|city     |valid_from|valid_to  |is_current|
# +-----------+---------+----------+----------+----------+
# |c1         |Hyderabad|2024-01-15|2025-03-01|false     |
# |c1         |Bengaluru|2025-03-01|2026-06-10|false     |
# |c1         |Pune     |2026-06-10|9999-12-31|true      |
# |c2         |Mumbai   |2023-05-20|2026-01-05|false     |
# |c2         |Chennai  |2026-01-05|9999-12-31|true      |
# |c3         |Delhi    |2025-11-11|9999-12-31|true      |
# +-----------+---------+----------+----------+----------+

assert scd2.count() == 6
assert scd2.filter(F.col("is_current")).count() == 3, \
    "one current row per customer"
c1 = (scd2.filter(F.col("customer_id") == "c1")
      .orderBy("valid_from").collect())
assert str(c1[0]["valid_to"]) == "2025-03-01", \
    "Hyderabad row ends when the Bengaluru move takes effect"
assert str(c1[1]["valid_to"]) == "2026-06-10"
assert str(c1[2]["valid_to"]) == OPEN_ENDED and c1[2]["is_current"]
# Point-in-time check: where did c1 live on 2025-12-25? Bengaluru.
as_of = scd2.filter(
    (F.col("customer_id") == "c1")
    & (F.col("valid_from") <= F.lit("2025-12-25"))
    & (F.col("valid_to") > F.lit("2025-12-25"))
).collect()
assert len(as_of) == 1 and as_of[0]["city"] == "Bengaluru"
print("OK: 6 change rows -> 6 SCD2 rows, 3 current, point-in-time works")

spark.stop()
