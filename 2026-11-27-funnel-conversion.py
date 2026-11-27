"""Funnel conversion: how many users actually finish each step?

Lesson
------
A funnel is not "count the events per step" — that counts repeats,
out-of-order events, and users who skipped a step, and it inflates
every downstream number. A funnel is "how many users reached step k
*having done every step before it, in order*".

The trick is one aggregation per user, not one filter per step:
``min(when(event == step, ts))`` gives the first time each user hit
each step (NULL when they never did). From those four timestamps you
can test the whole funnel in plain comparisons — reached step 2 means
``ts1 IS NOT NULL AND ts2 IS NOT NULL AND ts1 <= ts2`` — and the
conversion rate at each step is just ``reached_k / reached_(k-1)``.

Why ``min`` and not ``count``: a user who retries onboarding three
times is still one user at that step. ``count`` would let one noisy
user push a step's total above the number of users who entered the
funnel — a funnel where step 2 has more users than step 1 is always a
bug, and counting events instead of users is the usual cause.

Rules of thumb:
  * Build the funnel from per-user first-event times, then test the
    sequence with comparisons — never filter the raw events per step.
  * Treat NULL as "never reached": a missing timestamp fails the
    comparison, so skipped steps drop out of every later step
    automatically. A user who orders without signing up is a data
    problem to surface, not a conversion to count.
  * ``first(event_ts)`` needs a window and an order; ``min(when(...))``
    inside a plain ``groupBy`` is cheaper and reads the same.
  * Conversion is always a ratio of *users*, computed on the
    per-user table — dividing event counts gives a different (wrong)
    number whenever anyone repeats a step.

Run:  python 2026-11-27-funnel-conversion.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (IntegerType, StringType, StructField,
                               StructType)

STEPS = ["signup", "profile_complete", "first_order", "second_order"]


def main():
    spark = (SparkSession.builder.appName("funnel-conversion")
             .master("local[1]").getOrCreate())
    try:
        schema = StructType([
            StructField("user_id", IntegerType()),
            StructField("event", StringType()),
            StructField("event_ts", StringType()),
        ])

        # 10 users, deliberately messy:
        #  - user 1: clean run through all four steps, in order.
        #  - user 2: stops after first_order.
        #  - user 3: profile_complete twice (retry) — still one user.
        #  - user 4: first_order BEFORE profile_complete — out of
        #    order, so the funnel stops at profile_complete for them.
        #  - user 5: second_order without first_order — skipped step.
        #  - user 6: signup only.
        #  - user 7: no signup at all, but a first_order — a data
        #    problem, not a conversion.
        #  - user 8: all four steps, second_order repeated.
        #  - user 9: profile_complete but never signed up.
        #  - user 10: signup and profile_complete, then nothing.
        events = spark.createDataFrame([
            (1, "signup", "2026-11-01 09:00:00"),
            (1, "profile_complete", "2026-11-01 09:05:00"),
            (1, "first_order", "2026-11-02 10:00:00"),
            (1, "second_order", "2026-11-05 11:00:00"),
            (2, "signup", "2026-11-01 09:00:00"),
            (2, "profile_complete", "2026-11-01 09:10:00"),
            (2, "first_order", "2026-11-03 12:00:00"),
            (3, "signup", "2026-11-01 09:00:00"),
            (3, "profile_complete", "2026-11-01 09:02:00"),
            (3, "profile_complete", "2026-11-01 09:03:00"),
            (3, "first_order", "2026-11-02 08:00:00"),
            (3, "second_order", "2026-11-04 08:00:00"),
            (4, "signup", "2026-11-01 09:00:00"),
            (4, "first_order", "2026-11-01 09:30:00"),
            (4, "profile_complete", "2026-11-01 09:45:00"),
            (5, "signup", "2026-11-01 09:00:00"),
            (5, "profile_complete", "2026-11-01 09:05:00"),
            (5, "second_order", "2026-11-03 10:00:00"),
            (6, "signup", "2026-11-01 09:00:00"),
            (7, "first_order", "2026-11-02 10:00:00"),
            (8, "signup", "2026-11-01 09:00:00"),
            (8, "profile_complete", "2026-11-01 09:06:00"),
            (8, "first_order", "2026-11-02 09:00:00"),
            (8, "second_order", "2026-11-03 09:00:00"),
            (8, "second_order", "2026-11-04 09:00:00"),
            (9, "profile_complete", "2026-11-01 09:00:00"),
            (10, "signup", "2026-11-01 09:00:00"),
            (10, "profile_complete", "2026-11-01 09:20:00"),
        ], schema=schema)

        # One row per user: the FIRST timestamp of each step, NULL
        # when the user never did that step.
        per_user = events.groupBy("user_id").agg(*[
            F.min(F.when(F.col("event") == step,
                         F.col("event_ts"))).alias(f"ts_{step}")
            for step in STEPS
        ])

        # Reached step k = did every step up to k, in order. The
        # comparisons chain: each step's timestamp must exist and be
        # no earlier than the previous step's.
        reached = per_user.withColumn(
            "reached_signup", F.col("ts_signup").isNotNull()
        ).withColumn(
            "reached_profile",
            F.col("reached_signup")
            & F.col("ts_profile_complete").isNotNull()
            & (F.col("ts_signup") <= F.col("ts_profile_complete"))
        ).withColumn(
            "reached_first_order",
            F.col("reached_profile")
            & F.col("ts_first_order").isNotNull()
            & (F.col("ts_profile_complete") <= F.col("ts_first_order"))
        ).withColumn(
            "reached_second_order",
            F.col("reached_first_order")
            & F.col("ts_second_order").isNotNull()
            & (F.col("ts_first_order") <= F.col("ts_second_order"))
        )

        funnel = reached.agg(*[
            F.sum(F.col(f"reached_{step}").cast("int")).alias(step)
            for step in STEPS
        ]).collect()[0]
        counts = [int(funnel[step]) for step in STEPS]

        # Users 1, 2, 3, 4, 5, 8, 10 reach profile; 1, 2, 3, 8 reach
        # first_order; 1, 3, 8 reach second_order. User 4 (out of
        # order) and user 5 (skipped first_order) stop at profile;
        # user 7 never signed up, user 9 never signed up either.
        assert counts == [8, 7, 4, 3], counts
        assert counts == sorted(counts, reverse=True), \
            "funnel must shrink monotonically — a step can never " \
            "have more users than the one before it"

        # Naive event counts would say otherwise: first_order has 6
        # events and second_order 5, both above the true funnel
        # numbers, because they count repeats and out-of-order users.
        naive = events.groupBy("event").count().collect()
        naive_counts = {row["event"]: row["count"] for row in naive}
        assert naive_counts["second_order"] == 5 > counts[3], \
            "event counts must overstate the funnel here"

        for i, step in enumerate(STEPS):
            prev = counts[i - 1] if i else counts[0]
            rate = counts[i] / prev if prev else 0.0
            print(f"{step:17s} users={counts[i]}  "
                  f"step conversion={rate:.1%}")

        print("OK: funnel is 8 -> 7 -> 4 -> 3 users; event counts "
              "would have claimed 6 first_orders and 5 second_orders, "
              "counting repeats and users who skipped or reordered "
              "steps")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
