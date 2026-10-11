FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

# Unprivileged runtime user. The embedding-model cache (HF_HOME) lives in its
# home directory so model downloads work without root.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ui]"

COPY data ./data
COPY eval ./eval
COPY ui ./ui
COPY .streamlit ./.streamlit
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh

# App code stays root-owned (read-only to the app user); the eval harness
# writes its results under eval/, so only that directory is writable.
RUN chown -R app:app /app/eval

USER app

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
