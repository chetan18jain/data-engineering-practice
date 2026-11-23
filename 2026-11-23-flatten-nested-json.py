"""Flatten nested JSON: explode an array of structs into line items.

Lesson
------
Source systems love nested JSON — one order row carrying an `items`
array of structs (sku, qty, price). Downstream tables want one row
per line item, so the array has to be flattened. `explode()` does
that, but it silently DROPS rows whose array is empty or NULL — an
order with no items vanishes from the output, and a NULL array does
the same. `explode_outer()` keeps those rows with NULL item fields,
so no order is lost; filter or coalesce afterwards, deliberately.

Once exploded, each element is a struct: pull fields out with
`col("item").sku` (or `getField`), compute the line total, then
aggregate back per order. The explode-then-aggregate round trip is
the standard shape for nested order/event payloads.

Run:  python 2026-11-23-flatten-nested-json.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, IntegerType, StringType,
                               StructField, StructType)


def main():
    spark = (SparkSession.builder.appName("flatten-nested-json")
             .master("local[1]").getOrCreate())
    try:
        item_schema = StructType([
            StructField("sku", StringType()),
            StructField("qty", IntegerType()),
            StructField("price", IntegerType()),
        ])
        orders_schema = StructType([
            StructField("order_id", StringType()),
            StructField("customer", StringType()),
            StructField("items", ArrayType(item_schema)),
        ])
        # O3 has an empty items array, O4 a NULL one — both must survive.
        orders = spark.createDataFrame([
            ("O1", "Asha", [("SKU-1", 2, 499), ("SKU-2", 1, 1299)]),
            ("O2", "Ravi", [("SKU-1", 1, 499)]),
            ("O3", "Meera", []),
            ("O4", "Kiran", None),
        ], schema=orders_schema)

        # 1. explode_outer: one row per line item; O3/O4 kept, item NULL.
        lines = (orders
                 .withColumn("item", F.explode_outer("items"))
                 .withColumn("sku", F.col("item").getField("sku"))
                 .withColumn("qty", F.col("item").getField("qty"))
                 .withColumn("price", F.col("item").getField("price"))
                 .withColumn("line_total", F.col("qty") * F.col("price"))
                 .drop("items", "item"))
        lines.orderBy("order_id", "sku").show(truncate=False)

        # 2. Aggregate back: order totals, NULL-safe for item-less orders.
        totals = (lines.groupBy("order_id", "customer")
                  .agg(F.coalesce(F.sum("line_total"), F.lit(0))
                       .alias("order_total"),
                       F.count("sku").alias("line_count")))
        totals.orderBy("order_id").show(truncate=False)

        by_id = {r["order_id"]: r for r in totals.collect()}
        assert by_id["O1"]["order_total"] == 2 * 499 + 1299
        assert by_id["O1"]["line_count"] == 2
        assert by_id["O3"]["order_total"] == 0   # empty array, kept
        assert by_id["O4"]["order_total"] == 0   # NULL array, kept
        assert by_id["O4"]["line_count"] == 0
        assert lines.count() == 5  # 2 + 1 + 1 (empty) + 1 (NULL)
        print("OK: explode_outer flattened items without losing orders")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
