# Data Engineering Practice

One small, real problem per day — PySpark, SQL, Delta Lake, Airflow.
Each exercise is a single self-contained file: sample data inline, the
technique in the middle, and the lesson written at the top.

## Exercises

| Date | File | Problem | Technique |
|------|------|---------|-----------|
| 2026-11-19 | `2026-11-19-dedupe-keep-latest.py` | A batch contains the same order twice — keep only the newest version per key before a Delta MERGE | Window `row_number()` partitioned by key, ordered by `updated_at` desc |
| 2026-11-20 | `2026-11-20-sessionize-clickstream.py` | Raw clicks have no session id — derive sessions from a 30-minute inactivity gap | Window `lag()` for the gap, running `sum()` of a session-start flag |
| 2026-11-21 | `2026-11-21-scd2-effective-dates.py` | Address changes arrive as one row per change — build a Type 2 dimension with valid_from / valid_to / is_current | Window `lead()` for the next change date, no self-join |
| 2026-11-22 | `2026-11-22-asof-join-fx-rates.py` | Transactions need the FX rate effective on their date — or the most recent one before it | Range join on `rate_date <= txn_date`, then `row_number()` newest-first, keep rank 1 |
| 2026-11-23 | `2026-11-23-flatten-nested-json.py` | One order row carries an `items` array — flatten it to one row per line item without losing orders with empty or NULL arrays | `explode_outer()` keeps empty/NULL arrays, struct fields via `getField`, then aggregate back per order |
| 2026-11-24 | `2026-11-24-late-arriving-overlap-window.py` | A strict `updated_at > watermark` filter loses rows stamped exactly at the watermark and genuinely late rows — recover them with a reprocessing overlap window | Overlap window `[watermark - 2h, now]`, MERGE + `row_number()` dedupe keeping the latest version per key |
| 2026-11-25 | `2026-11-25-skewed-join-salting.py` | One join key ("WEB") holds 80% of the fact rows, so a plain join funnels everything to a single task — spread it without changing the result | Salt the large side (`hash(event_id) % N`), replicate the small side N times via cross join, join on `(channel, salt)` |
| 2026-11-26 | `2026-11-26-rolling-7day-range-between.py` | A rolling 7-day total built with `rowsBetween(-6, 0)` over-counts whenever dates are missing — it counts rows, not days | `rangeBetween` on `unix_timestamp(date)`, `[-6*86400, 0]`, for a true calendar window |
