"""As-of join: attach the latest FX rate on or before each transaction.

Lesson
------
A transaction dated 2026-11-15 must use the FX rate effective on
2026-11-15 — or the most recent rate BEFORE it if none was published
that day. An equality join on the date drops those rows, and a plain
range join (`rate_date <= txn_date`) fans each transaction out to
every older rate. The fix: range-join, then rank the matches with
row_number() ordered by rate_date DESC and keep rank 1. That is an
"as-of" join — join to the value as of a point in time.

Same pattern powers slowly-changing reference data everywhere:
exchange rates, tax tables, price lists, product categories.

Run:  python 2026-11-22-asof-join-fx-rates.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    spark = (SparkSession.builder.appName("asof-join-fx-rates")
             .master("local[1]").getOrCreate())
    try:
        # Transactions in USD; rates published only on some days.
        txns = spark.createDataFrame([
            ("T1", "2026-11-14", 100.0),
            ("T2", "2026-11-15", 250.0),   # no rate on the 15th -> use 14th
            ("T3", "2026-11-17", 400.0),   # no rate on 16th/17th -> use 14th
            ("T4", "2026-11-18", 50.0),    # rate published on the 18th
            ("T5", "2026-11-10", 75.0),    # before any rate -> stays NULL
        ], ["txn_id", "txn_date", "amount_usd"])
        rates = spark.createDataFrame([
            ("2026-11-11", 83.10),
            ("2026-11-14", 83.25),
            ("2026-11-18", 83.40),
        ], ["rate_date", "usd_to_inr"])

        txns = txns.withColumn("txn_date", F.to_date("txn_date"))
        rates = rates.withColumn("rate_date", F.to_date("rate_date"))

        # 1. Range join: every rate on or before the transaction date.
        matched = txns.join(
            rates, F.col("rate_date") <= F.col("txn_date"), "left")
        # 2. Rank matches newest-first per transaction; keep rank 1.
        latest_first = Window.partitionBy("txn_id").orderBy(
            F.col("rate_date").desc_nulls_last())
        asof = (matched
                .withColumn("_rn", F.row_number().over(latest_first))
                .filter(F.col("_rn") == 1)
                .drop("_rn")
                .withColumn("amount_inr",
                            F.col("amount_usd") * F.col("usd_to_inr")))
        asof.orderBy("txn_id").show(truncate=False)
        # T2 uses the 14th's rate (83.25), T5 has no rate yet (NULL INR).
        assert asof.filter("txn_id = 'T2'").first()["usd_to_inr"] == 83.25
        assert asof.filter("txn_id = 'T5'").first()["usd_to_inr"] is None
        print("OK: as-of join attached the latest rate on/before each date")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
