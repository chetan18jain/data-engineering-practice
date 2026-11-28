"""Incremental load: which batch rows are inserts, which are updates?

Lesson
------
An incremental load is not "overwrite the table with the batch" —
that reprocesses everything and loses nothing only by luck. It is
"split the batch into three sets, and touch only the rows that
changed":

  * inserts  — keys in the batch that are not in the target yet.
  * updates  — keys in both, where a compared column differs.
  * unchanged — keys in both, nothing differs: skip them entirely.

The insert set is one join: ``batch.join(target, "key", "left_anti")``
keeps exactly the batch rows with no matching target key. No
``exceptAll``, no full outer join, no collecting keys to Python —
one pass, and it reads the target only for its keys.

The update set needs a join on the key plus a difference test on the
compared columns — and the test must be NULL-safe. Plain ``!=``
returns NULL (not TRUE) when either side is NULL, so a row whose
amount went from NULL to 500 is silently classified as unchanged.
Spark's null-safe equality — ``eqNullSafe`` in Python, ``<=>`` in
SQL — returns TRUE when both sides are NULL and FALSE when only one
is, so ``NOT eqNullSafe(...)`` is TRUE exactly when the values
differ, NULLs included. Change detection is one of the places
null-safe equality is not a nicety but the correct operator: with
``!=`` a NULL-to-value fix never lands, and the target keeps the
stale NULL forever.

Rules of thumb:
  * Inserts = ``left_anti`` on the key. It is the cheapest way to
    ask "what is new" and it never touches a matching row.
  * Updates = inner join on the key + ``NOT eqNullSafe(col, col)``
    per compared column (``<=>`` in SQL). Compare only the columns
    the load owns — comparing a derived column re-flags every row.
  * A key in the target but absent from the batch is untouched, not
    deleted: an incremental load never deletes what the batch did
    not mention. Deletes are a separate, explicit step.
  * Do the split once, then MERGE (or write) inserts + updates and
    skip unchanged — the unchanged set is usually the largest, and
    skipping it is the whole point of incremental.

Run:  python 2026-11-28-incremental-load-anti-join.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType)


def main():
    spark = (SparkSession.builder.appName("incremental-load-anti-join")
             .master("local[1]").getOrCreate())
    try:
        schema = StructType([
            StructField("order_id", IntegerType()),
            StructField("customer_id", StringType()),
            StructField("amount", DoubleType()),
            StructField("updated_at", StringType()),
        ])

        # Target as it stands today. Order 107 has a NULL amount —
        # a gap the next batch fills.
        target = spark.createDataFrame([
            (101, "C1", 250.0, "2026-11-20 10:00:00"),
            (102, "C2", 400.0, "2026-11-21 09:00:00"),
            (103, "C3", 150.0, "2026-11-22 08:00:00"),
            (104, "C4", 900.0, "2026-11-23 07:00:00"),
            (107, "C7", None, "2026-11-24 06:00:00"),
        ], schema=schema)

        # Today's batch. Deliberately mixed:
        #  - 101: identical to target — unchanged, must be skipped.
        #  - 102: amount changed 400 -> 450 — update.
        #  - 103: amount the same, updated_at re-stamped — update
        #    (the row changed, even though the money did not).
        #  - 105, 106: brand-new keys — inserts.
        #  - 107: amount NULL -> 500 with updated_at unchanged —
        #    update, and the trap: plain != sees NULL != 500 as
        #    NULL, not TRUE, and the timestamp test is FALSE, so
        #    NULL OR FALSE is NULL and filter() drops the row.
        #  - 104: in the target, absent from the batch — untouched.
        batch = spark.createDataFrame([
            (101, "C1", 250.0, "2026-11-20 10:00:00"),
            (102, "C2", 450.0, "2026-11-25 09:00:00"),
            (103, "C3", 150.0, "2026-11-25 08:00:00"),
            (105, "C5", 300.0, "2026-11-26 10:00:00"),
            (106, "C6", 120.0, "2026-11-26 11:00:00"),
            (107, "C7", 500.0, "2026-11-24 06:00:00"),
        ], schema=schema)

        # Inserts: batch rows whose key has no match in the target.
        inserts = batch.join(target, "order_id", "left_anti")

        # Changed rows: same key in both, any compared column
        # differs. eqNullSafe is null-safe: NULL eqNullSafe NULL is
        # TRUE, and NULL eqNullSafe 500 is FALSE, so NOT eqNullSafe
        # catches the NULL fix.
        joined = batch.alias("b").join(target.alias("t"), "order_id")
        changed = joined.filter(
            ~F.col("b.amount").eqNullSafe(F.col("t.amount"))
            | ~F.col("b.updated_at").eqNullSafe(F.col("t.updated_at"))
        )
        # The unchanged set, for the record: same key, all compared
        # columns equal.
        unchanged = joined.filter(
            F.col("b.amount").eqNullSafe(F.col("t.amount"))
            & F.col("b.updated_at").eqNullSafe(F.col("t.updated_at"))
        )

        insert_ids = sorted(r["order_id"] for r in inserts.collect())
        changed_ids = sorted(r["order_id"] for r in changed.collect())
        unchanged_ids = sorted(r["order_id"]
                               for r in unchanged.collect())
        assert insert_ids == [105, 106], insert_ids
        assert changed_ids == [102, 103, 107], changed_ids
        assert unchanged_ids == [101], unchanged_ids

        # The trap, shown: with plain != the NULL-to-500 fix on 107
        # is not flagged, because NULL != 500 evaluates to NULL,
        # the updated_at test is FALSE, and NULL OR FALSE is NULL —
        # filter() drops NULL rows.
        naive_changed = joined.filter(
            (F.col("b.amount") != F.col("t.amount"))
            | (F.col("b.updated_at") != F.col("t.updated_at"))
        )
        naive_ids = sorted(r["order_id"]
                           for r in naive_changed.collect())
        assert naive_ids == [102, 103], naive_ids
        assert 107 not in naive_ids, \
            "plain != must miss the NULL fix — that is the bug"

        # The upsert set is inserts + updates: 5 of the 6 batch rows
        # need writing, and only the 1 unchanged row is skipped.
        # Order 104 is in neither set: absent from the batch, it
        # stays in the target untouched.
        print(f"inserts:   {insert_ids}")
        print(f"updates:   {changed_ids}")
        print(f"unchanged: {unchanged_ids} (skipped)")
        print("OK: batch of 6 splits into 2 inserts, 3 updates, "
              "1 unchanged; plain != would have lost the NULL->500 "
              "fix on order 107, and order 104 stays untouched")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
