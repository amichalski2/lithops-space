# Lithops on Cloud Run Jobs — the autonomous CEO and the benchmark it operates.
#
# The CEO-Bench distribution is proprietary and is NOT vendored in this
# repository. `deploy.sh` stages what the image needs — the `novamind-operation`
# zipapp, its requirements, and the published `docs/` SDK — into `vendor/ceobench/`
# of a temporary build context, then submits that to Cloud Build. Nothing
# proprietary is committed here, and no local Docker daemon is required.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# sqlcipher3-binary ships a manylinux wheel; the rest are pure wheels. gcloud
# storage is used by the entrypoint to checkpoint state outside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- benchmark --
# The simulated company runs in its own interpreter, exactly as it does locally:
# Lithops drives it through the documented public CLI and never imports it.
WORKDIR /opt/ceobench
COPY vendor/ceobench/requirements.txt ./requirements.txt
RUN python -m venv /opt/ceobench/venv \
    && /opt/ceobench/venv/bin/pip install --no-cache-dir -r requirements.txt
COPY vendor/ceobench/novamind-operation ./novamind-operation
RUN chmod +x ./novamind-operation
# The benchmark exposes its SDK as source at <base>/docs/novamind_api and puts
# <base>/docs on PYTHONPATH for agent code; without it every tool call fails on
# `import novamind_api`. The zipapp itself stays off that path by design.
COPY vendor/ceobench/docs ./docs

# ------------------------------------------------------------------ lithops --
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
# `agents` carries google-adk: the Executive reasons through ADK, so it is not
# optional for a run even though it is an optional extra for a library consumer.
RUN pip install --no-cache-dir -e ".[agents]"

COPY scripts/ ./scripts/
COPY deploy/cloudrun/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Sessions are written next to the executable; the entrypoint mirrors this
# directory to Cloud Storage so an interrupted run can be resumed.
ENV CEOBENCH_EXECUTABLE=/opt/ceobench/novamind-operation \
    CEOBENCH_PYTHON=/opt/ceobench/venv/bin/python \
    LITHOPS_BENCHMARK_BACKEND=ceobench \
    LITHOPS_STORAGE_BACKEND=supabase \
    LITHOPS_MODEL_PROVIDER=gemini \
    LITHOPS_EXECUTABLE_MODEL_PLANNING=true \
    LITHOPS_EXECUTIVE_AUTHORITY_V2=true

ENTRYPOINT ["/app/entrypoint.sh"]
