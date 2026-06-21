#!/usr/bin/env bash
# One-command local launcher.
#   ./run_local.sh
# Starts the API (auto-builds the index on first request) and the Streamlit UI.
# Press Ctrl+C once to stop both. Uses the home venv created during setup so the
# colon in the project path doesn't matter.
set -euo pipefail
cd "$(dirname "$0")"

source ~/rag-guardrails-venv/bin/activate
export PYTHONPATH="$(pwd)/src"

echo "Starting API on http://localhost:8000 (docs at /docs) ..."
uvicorn rag.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

echo "Waiting for the API to come up (first run downloads the embedding model)..."
for _ in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "API is up."
    break
  fi
  sleep 1
done

echo "Opening the UI on http://localhost:8501 ..."
exec streamlit run ui/streamlit_app.py
