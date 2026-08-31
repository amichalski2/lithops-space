#!/usr/bin/env bash
# Start one execution of the Lithops run job.
#
# Each execution advances the same run: state is restored from Cloud Storage at
# start and mirrored back as weeks commit, so a 500-day run can be carried by
# several executions and an interrupted one resumes at the week after the last
# committed decision.
set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${RUN_NAME:?RUN_NAME must be set (for example driftaware-500d-cloud)}"

REGION="${REGION:-europe-central2}"
JOB_NAME="${JOB_NAME:-lithops-ceobench}"
SEED="${SEED:-83}"
WEEKS="${WEEKS:-72}"

ENV_VARS="RUN_NAME=${RUN_NAME},SEED=${SEED},WEEKS=${WEEKS}"
if [[ -n "${MAX_WEEKS_THIS_PROCESS:-}" ]]; then
  ENV_VARS="${ENV_VARS},MAX_WEEKS_THIS_PROCESS=${MAX_WEEKS_THIS_PROCESS}"
fi
if [[ -n "${RETRY_FAILED:-}" ]]; then
  ENV_VARS="${ENV_VARS},RETRY_FAILED=${RETRY_FAILED}"
fi

gcloud run jobs execute "${JOB_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --update-env-vars "${ENV_VARS}" \
  "${@}"

cat <<EOF

Follow the run:
  gcloud beta run jobs logs tail ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}

Week-by-week decisions only:
  gcloud logging read \\
    'resource.type="cloud_run_job" AND jsonPayload.component="lithops.week"' \\
    --project ${PROJECT_ID} --format='value(jsonPayload.week, jsonPayload.strategy, jsonPayload.binding_constraint)' --limit 50
EOF
