#!/usr/bin/env bash
# Build the Lithops image and (re)create the Cloud Run Job that runs CEO-Bench.
#
# Idempotent: safe to re-run after a code change. It never starts a run — see
# `execute.sh` for that, so building and spending are separate decisions.
set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${CEOBENCH_PUBLIC_DIR:?CEOBENCH_PUBLIC_DIR must point at ceobench-src/public}"

REGION="${REGION:-europe-central2}"
REPOSITORY="${REPOSITORY:-lithops}"
JOB_NAME="${JOB_NAME:-lithops-ceobench}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-${PROJECT_ID}-lithops-runs}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/lithops:$(date -u +%Y%m%d-%H%M%S)"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-lithops-runner@${PROJECT_ID}.iam.gserviceaccount.com}"
# Secret names already in use by the deployed API service; the job reads the
# same values rather than keeping a second copy of the same credentials.
GEMINI_SECRET="${GEMINI_SECRET:-lithops-gemini-api-key}"
SUPABASE_URL_SECRET="${SUPABASE_URL_SECRET:-lithops-supabase-url}"
SUPABASE_KEY_SECRET="${SUPABASE_KEY_SECRET:-lithops-supabase-secret-key}"
# The simulated company runs its own customer model on Anthropic; that is the
# benchmark's credential, not the agent's, and it refuses to start without it.
ANTHROPIC_SECRET="${ANTHROPIC_SECRET:-lithops-anthropic-api-key}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> enabling APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> ensuring Artifact Registry repository"
gcloud artifacts repositories describe "${REPOSITORY}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format docker --location "${REGION}" --project "${PROJECT_ID}" \
  --description "Lithops autonomous CEO images"

echo "==> ensuring state bucket gs://${ARTIFACT_BUCKET}"
gcloud storage buckets describe "gs://${ARTIFACT_BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud storage buckets create "gs://${ARTIFACT_BUCKET}" \
  --location "${REGION}" --project "${PROJECT_ID}" --uniform-bucket-level-access

echo "==> ensuring service account"
gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create lithops-runner \
  --display-name "Lithops autonomous run" --project "${PROJECT_ID}"
gcloud storage buckets add-iam-policy-binding "gs://${ARTIFACT_BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" --role roles/storage.objectAdmin \
  --project "${PROJECT_ID}" >/dev/null
for secret in "${GEMINI_SECRET}" "${SUPABASE_URL_SECRET}" "${SUPABASE_KEY_SECRET}" "${ANTHROPIC_SECRET}"; do
  gcloud secrets add-iam-policy-binding "${secret}" \
    --member "serviceAccount:${SERVICE_ACCOUNT}" --role roles/secretmanager.secretAccessor \
    --project "${PROJECT_ID}" >/dev/null
done

# The benchmark distribution is proprietary and lives outside this repository,
# so the build context is staged: the repository plus exactly the two files the
# image needs from it. Cloud Build compiles and pushes, so no local Docker
# daemon is required.
echo "==> staging build context"
BUILD_CONTEXT="$(mktemp -d)"
trap 'rm -rf "${BUILD_CONTEXT}"' EXIT
tar -C "${REPO_ROOT}" \
  --exclude=.git --exclude=.venv --exclude=.venv-win --exclude=artifacts \
  --exclude=__pycache__ --exclude='*.pyc' --exclude=node_modules \
  -cf - pyproject.toml README.md backend scripts deploy | tar -C "${BUILD_CONTEXT}" -xf -
# `gcloud builds submit --tag` expects the Dockerfile at the context root.
cp "${REPO_ROOT}/deploy/cloudrun/Dockerfile" "${BUILD_CONTEXT}/Dockerfile"
mkdir -p "${BUILD_CONTEXT}/vendor/ceobench"
cp "${CEOBENCH_PUBLIC_DIR}/novamind-operation" \
   "${CEOBENCH_PUBLIC_DIR}/requirements.txt" \
   "${BUILD_CONTEXT}/vendor/ceobench/"
cp -r "${CEOBENCH_PUBLIC_DIR}/docs" "${BUILD_CONTEXT}/vendor/ceobench/docs"

echo "==> building ${IMAGE} on Cloud Build"
gcloud builds submit "${BUILD_CONTEXT}" \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"

echo "==> deploying Cloud Run Job ${JOB_NAME}"
ACTION=update
gcloud run jobs describe "${JOB_NAME}" --region "${REGION}" --project "${PROJECT_ID}" \
  >/dev/null 2>&1 || ACTION=create

gcloud run jobs "${ACTION}" "${JOB_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --cpu 2 \
  --memory 4Gi \
  --max-retries 1 \
  --task-timeout 24h \
  --set-env-vars "ARTIFACT_BUCKET=${ARTIFACT_BUCKET}" \
  --set-secrets "GEMINI_API_KEY=${GEMINI_SECRET}:latest,SUPABASE_URL=${SUPABASE_URL_SECRET}:latest,SUPABASE_SECRET_KEY=${SUPABASE_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_SECRET}:latest"

echo
echo "deployed ${IMAGE}"
echo "run it with: RUN_NAME=<name> SEED=<seed> ./deploy/cloudrun/execute.sh"
