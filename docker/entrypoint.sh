#!/usr/bin/env bash
set -euo pipefail

# Wait for Postgres (pgvector) if a DATABASE_URL is configured and we use it.
if [ "${VECTOR_STORE:-pgvector}" = "pgvector" ]; then
  echo "Waiting for Postgres..."
  python - <<'PY'
import os, time, sys
import psycopg
url = os.environ.get("DATABASE_URL", "postgresql://rag:rag@db:5432/rag")
for _ in range(60):
    try:
        psycopg.connect(url, connect_timeout=2).close()
        print("Postgres is ready.")
        sys.exit(0)
    except Exception:
        time.sleep(1)
print("Postgres not reachable; continuing anyway.", file=sys.stderr)
PY
fi

# Build the index (idempotent — resets and re-ingests the corpus).
echo "Ingesting corpus..."
python -m rag.ingest --corpus data/corpus || echo "Ingestion failed; starting API anyway."

exec "$@"
