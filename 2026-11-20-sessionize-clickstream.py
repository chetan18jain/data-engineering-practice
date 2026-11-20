"""Day 2 — Sessionize clickstream events with a 30-minute inactivity gap.

Problem: raw click events arrive as one row per click. For funnel and
engagement analysis you need sessions: a new session starts when a user
is idle for 30 minutes or more. There is no session_id column to group
by — it has to be derived from the timestamps.

Fix: lag() the previous event time per user, flag a row as a session
start when the gap is >= 30 minutes (or it is the user's first event),
then a running sum() of that flag over the window assigns a session
number to every event. Two window functions, one pass, no self-join.

Run: spark-submit 2026-11-20-sessionize-clickstream.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("sessionize-clickstream").getOrCreate()

events = spark.createDataFrame(
    [
        # (user_id, event, event_time)
        ("u1", "page_view", "2026-11-20 09:00:00"),
        ("u1", "click",     "2026-11-20 09:05:00"),
        ("u1", "purchase",  "2026-11-20 09:50:00"),  # 45 min gap -> new session
        ("u1", "page_view", "2026-11-20 10:10:00"),
        ("u2", "page_view", "2026-11-20 09:02:00"),
        ("u2", "click",     "2026-11-20 09:29:00"),  # 27 min gap -> same session
        ("u2", "click",     "2026-11-20 10:05:00"),  # 36 min gap -> new session
    ],
    ["user_id", "event", "event_time"],
).withColumn("event_time", F.to_timestamp("event_time"))

user_window = Window.partitionBy("user_id").orderBy("event_time")

sessionized = (
    events
    .withColumn("prev_time", F.lag("event_time").over(user_window))
    .withColumn(
        "gap_min",
        (F.unix_timestamp("event_time")
         - F.unix_timestamp("prev_time")) / 60,
    )
    .withColumn(
        "is_new_session",
        F.when(F.col("prev_time").isNull(), 1)
         .when(F.col("gap_min") >= 30, 1)
         .otherwise(0),
    )
    .withColumn(
        "session_no",
        F.sum("is_new_session").over(
            user_window.rowsBetween(Window.unboundedPreceding,
                                    Window.currentRow)),
    )
    .drop("prev_time", "gap_min", "is_new_session")
    .orderBy("user_id", "event_time")
)

sessionized.show(truncate=False)
# +-------+----------+-------------------+----------+
# |user_id|event     |event_time         |session_no|
# +-------+----------+-------------------+----------+
# |u1     |page_view |2026-11-20 09:00:00|1         |
# |u1     |click     |2026-11-20 09:05:00|1         |
# |u1     |purchase |2026-11-20 09:50:00|2         |
# |u1     |page_view |2026-11-20 10:10:00|2         |
# |u2     |page_view |2026-11-20 09:02:00|1         |
# |u2     |click     |2026-11-20 09:29:00|1         |
# |u2     |click     |2026-11-20 10:05:00|2         |
# +-------+----------+-------------------+----------+

assert sessionized.count() == 7
u1_sessions = (sessionized.filter(F.col("user_id") == "u1")
               .select("session_no").distinct().count())
assert u1_sessions == 2, "u1's 45-min gap must start a second session"
purchase = sessionized.filter(F.col("event") == "purchase").first()
assert purchase["session_no"] == 2
u2_clicks = (sessionized.filter(F.col("user_id") == "u2")
             .orderBy("event_time").collect())
assert u2_clicks[1]["session_no"] == 1, "27-min gap stays in session 1"
assert u2_clicks[2]["session_no"] == 2, "36-min gap starts session 2"
print("OK: 7 events -> 2 sessions per user, 30-min inactivity gap")

spark.stop()
