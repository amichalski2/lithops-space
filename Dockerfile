# Lithops evidence API for Cloud Run.
#
# The container serves the read-only product API and the weekly execution spine.
# It deliberately does not ship the CEO-Bench CLI: that binary is an external
# environment, so a cloud deployment reads persisted run evidence from Supabase and
# leaves the benchmark adapter on the fake backend unless a real one is provided.

FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /web
COPY package.json package-lock.json ./
COPY frontend/package.json ./frontend/package.json
RUN npm ci
COPY frontend ./frontend
ENV VITE_API_URL=""
RUN npm run frontend:build

FROM ghcr.io/astral-sh/uv:0.8.13-python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Resolve the locked dependency set before copying source, so ordinary code edits
# do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra agents

COPY backend/src ./backend/src
# --no-editable matters: the default editable install only links back to /app, which
# the runtime stage does not carry, so the package would be missing at import time.
RUN uv sync --frozen --no-dev --extra agents --no-editable

FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LITHOPS_STORAGE_BACKEND=supabase \
    LITHOPS_BENCHMARK_BACKEND=fake \
    LITHOPS_MODEL_PROVIDER=static \
    LITHOPS_FRONTEND_DIST=/opt/lithops-frontend \
    PORT=8080

RUN useradd --create-home --uid 10001 lithops
COPY --from=builder /opt/venv /opt/venv
COPY --from=frontend-builder /web/frontend/dist /opt/lithops-frontend

USER lithops
WORKDIR /home/lithops

EXPOSE 8080

# Cloud Run injects PORT; the shell form lets it expand.
CMD exec uvicorn lithops.api.main:app --host 0.0.0.0 --port "${PORT}"
