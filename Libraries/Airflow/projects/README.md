# Airflow DAG contract

Validate scheduling, timezone, ownership, retries, timeouts, dependencies, and
idempotent side effects without starting Airflow. Run `python -W error -m
unittest -v`; extend with DAG-bag parsing and a backfill integration test.
