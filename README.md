# Data Engineering Practice

One small, real problem per day — PySpark, SQL, Delta Lake, Airflow.
Each exercise is a single self-contained file: sample data inline, the
technique in the middle, and the lesson written at the top.

## Exercises

| Date | File | Problem | Technique |
|------|------|---------|-----------|
| 2026-11-19 | `2026-11-19-dedupe-keep-latest.py` | A batch contains the same order twice — keep only the newest version per key before a Delta MERGE | Window `row_number()` partitioned by key, ordered by `updated_at` desc |
| 2026-11-20 | `2026-11-20-sessionize-clickstream.py` | Raw clicks have no session id — derive sessions from a 30-minute inactivity gap | Window `lag()` for the gap, running `sum()` of a session-start flag |
