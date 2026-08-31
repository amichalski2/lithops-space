#!/usr/bin/env bash
# Run one leg of a CEO-Bench run inside a Cloud Run Job.
#
# A 500-day run is a long-lived autonomous operation, so the container is not
# the unit of durability: Supabase holds the decision record, Cloud Storage
# holds the artifacts and the benchmark session, and the checkpoint lets the
# next execution pick up the week after the last committed one. Killing this
# job and starting it again is a supported operation, not a failure path.
set -euo pipefail

: "${RUN_NAME:?RUN_NAME must be set: it names the state prefix in GCS}"
: "${ARTIFACT_BUCKET:?ARTIFACT_BUCKET must be set: the bucket holding run state}"

SEED="${SEED:-83}"
WEEKS="${WEEKS:-72}"
ROLLOUTS="${ROLLOUTS:-200}"
MAX_WEEKS_THIS_PROCESS="${MAX_WEEKS_THIS_PROCESS:-}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"

STATE_URI="gs://${ARTIFACT_BUCKET}/runs/${RUN_NAME}"
ARTIFACT_DIR="/app/artifacts/experiments/${RUN_NAME}"
SESSION_DIR="/opt/ceobench/sessions"

mkdir -p "${ARTIFACT_DIR}" "${SESSION_DIR}"

log() {
  # Cloud Logging parses a JSON object on stdout into structured fields.
  printf '{"severity":"%s","component":"lithops.cloudrun","message":%s}\n' \
    "$1" "$(printf '%s' "$2" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
}

log INFO "restoring run state from ${STATE_URI}"
gcloud storage rsync --recursive "${STATE_URI}/artifacts" "${ARTIFACT_DIR}" 2>/dev/null || \
  log INFO "no prior artifacts for ${RUN_NAME}; starting a fresh run"
gcloud storage rsync --recursive "${STATE_URI}/sessions" "${SESSION_DIR}" 2>/dev/null || \
  log INFO "no prior benchmark session for ${RUN_NAME}"

sync_state() {
  # A sync failure must be visible: silence here once hid an empty state bucket.
  gcloud storage rsync --recursive "${ARTIFACT_DIR}" "${STATE_URI}/artifacts" >/dev/null \
    || log WARNING "artifact sync to ${STATE_URI}/artifacts failed"
  gcloud storage rsync --recursive "${SESSION_DIR}" "${STATE_URI}/sessions" >/dev/null \
    || log WARNING "session sync to ${STATE_URI}/sessions failed"
}

# Mirror state while the run advances, so an interrupted execution loses at most
# one sync interval of artifacts and never the committed decision record.
(
  while true; do
    sleep "${SYNC_INTERVAL_SECONDS}"
    sync_state
  done
) &
SYNC_PID=$!

finish() {
  local status=$?
  kill "${SYNC_PID}" 2>/dev/null || true
  log INFO "final state sync to ${STATE_URI}"
  sync_state
  exit "${status}"
}
trap finish EXIT

ARGS=(
  --provider gemini
  --python "${CEOBENCH_PYTHON}"
  --executable "${CEOBENCH_EXECUTABLE}"
  --checkpoint "${ARTIFACT_DIR}/checkpoint.json"
  --report "${ARTIFACT_DIR}/report.json"
  --weeks "${WEEKS}"
  --seed "${SEED}"
  --rollouts "${ROLLOUTS}"
  --worker-id "${RUN_NAME}"
  --executive-authority-v2
)
if [[ -n "${MAX_WEEKS_THIS_PROCESS}" ]]; then
  ARGS+=(--max-weeks-this-process "${MAX_WEEKS_THIS_PROCESS}")
fi
if [[ "${RETRY_FAILED:-false}" == "true" ]]; then
  ARGS+=(--retry-failed)
fi

log INFO "starting CEO-Bench run ${RUN_NAME}: seed ${SEED}, ${WEEKS} weeks"
# Not `exec`: replacing this shell would orphan the sync loop and skip the EXIT
# trap, so the final state would never reach Cloud Storage.
python /app/scripts/run_ceobench_experiment.py "${ARGS[@]}"
