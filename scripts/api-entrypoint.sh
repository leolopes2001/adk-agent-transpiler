#!/bin/sh
# Container entrypoint for the scheduler API: apply migrations, then serve.
# Running migrations at startup is fine for a single replica; with several
# replicas they would run concurrently, so production would use a separate job.
set -eu

alembic upgrade head

# --no-access-log: the app already logs one structured `request` event per call.
exec uvicorn scheduler_api.main:create_app --factory \
    --host 0.0.0.0 --port "${PORT:-8000}" --no-access-log
