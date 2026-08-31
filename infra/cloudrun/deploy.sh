#!/usr/bin/env bash
# Deploy the Lithops evidence API to Cloud Run.
#
# Persistence secrets are held in Secret Manager and mounted as environment variables.
# Participant Gemini keys are request-scoped BYOK credentials and are never installed
# as Cloud Run secrets or application environment variables.
#
# Read/replay endpoints are public for the demo. A weekly step requires the caller's
# X-Gemini-API-Key, so the service cannot spend project-owned provider quota.
#
# Usage: infra/cloudrun/deploy.sh [region]

set -euo pipefail

: "${LITHOPS_GCP_PROJECT:?Set LITHOPS_GCP_PROJECT to your Google Cloud project ID}"
PROJECT="${LITHOPS_GCP_PROJECT}"
REGION="${1:-${LITHOPS_GCP_REGION:-europe-central2}}"
# Overridable: the public site runs as `lithops` in europe-west1 (the name the
# lithops.space domain mapping targets); the API-only deployment keeps its own.
SERVICE="${LITHOPS_SERVICE:-lithops-api}"
RUNTIME_SA="lithops-api-run"
SA_EMAIL="${RUNTIME_SA}@${PROJECT}.iam.gserviceaccount.com"

if [[ ! -f .env ]]; then
    echo "error: .env with SUPABASE_URL and SUPABASE_SECRET_KEY is required" >&2
    exit 1
fi

# shellcheck disable=SC1091
set -a; source .env; set +a

# The repository lives on a Windows-mounted filesystem, so .env carries CRLF line
# endings. An unnoticed trailing carriage return travels into Secret Manager and makes
# the Supabase URL unparseable at runtime, so strip it here.
strip_cr() { printf '%s' "${1//$'\r'/}"; }
SUPABASE_URL="$(strip_cr "${SUPABASE_URL:-}")"
SUPABASE_SECRET_KEY="$(strip_cr "${SUPABASE_SECRET_KEY:-}")"
GEMINI_MODEL="$(strip_cr "${GEMINI_MODEL:-}")"
LITHOPS_DEMO_RUN_ID="$(strip_cr "${LITHOPS_DEMO_RUN_ID:-}")"

for name in SUPABASE_URL SUPABASE_SECRET_KEY; do
    if [[ -z "${!name:-}" ]]; then
        echo "error: ${name} is not set in .env" >&2
        exit 1
    fi
done

echo "==> project ${PROJECT}, region ${REGION}"
gcloud config set project "${PROJECT}" --quiet

echo "==> enabling required services"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    --quiet

echo "==> runtime service account"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --quiet >/dev/null 2>&1; then
    gcloud iam service-accounts create "${RUNTIME_SA}" \
        --display-name="Lithops evidence API runtime" --quiet
fi

put_secret() {
    local name="$1" value="$2"
    if ! gcloud secrets describe "${name}" --quiet >/dev/null 2>&1; then
        gcloud secrets create "${name}" --replication-policy=automatic --quiet
    fi
    printf '%s' "${value}" | gcloud secrets versions add "${name}" --data-file=- --quiet
    gcloud secrets add-iam-policy-binding "${name}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role=roles/secretmanager.secretAccessor --quiet >/dev/null
}

echo "==> secrets"
put_secret lithops-supabase-url "${SUPABASE_URL}"
put_secret lithops-supabase-secret-key "${SUPABASE_SECRET_KEY}"
echo "==> building and deploying from source"
gcloud run deploy "${SERVICE}" \
    --source . \
    --region "${REGION}" \
    --service-account "${SA_EMAIL}" \
    --allow-unauthenticated \
    --cpu 1 --memory 1Gi \
    --min-instances 0 --max-instances 1 \
    --timeout 600 \
    --set-env-vars "LITHOPS_STORAGE_BACKEND=supabase,LITHOPS_BENCHMARK_BACKEND=fake,LITHOPS_MODEL_PROVIDER=static,GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.7-flash},LITHOPS_DEMO_RUN_ID=${LITHOPS_DEMO_RUN_ID}" \
    --set-secrets "SUPABASE_URL=lithops-supabase-url:latest,SUPABASE_SECRET_KEY=lithops-supabase-secret-key:latest" \
    --quiet

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
echo
echo "==> deployed: ${URL}"
echo "==> health check"
curl -fsS "${URL}/health" && echo
echo "==> recent logs"
gcloud logging read \
    "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE}" \
    --limit 10 --format='value(timestamp,textPayload)' --project "${PROJECT}"
