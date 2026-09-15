#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
DIAG_ROUNDS="${DIAG_ROUNDS:-0,20,40,59}"
ANALYZE_AFTER_TRAIN="${ANALYZE_AFTER_TRAIN:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/server_experiments/mos_sign_shadow_diagnostics/${TIMESTAMP}"

[[ "${GPU}" =~ ^-?[0-9]+$ ]] || { echo "GPU must be an integer" >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative" >&2; exit 2; }
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { echo "ROUNDS must be positive" >&2; exit 2; }
[[ "${DIAG_ROUNDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "DIAG_ROUNDS must be comma-separated non-negative integers" >&2; exit 2; }
[[ "${ANALYZE_AFTER_TRAIN}" =~ ^[01]$ ]] || { echo "ANALYZE_AFTER_TRAIN must be 0 or 1" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to overwrite ${RUN_DIR}" >&2; exit 2; }

mkdir -p "${RUN_DIR}/config" "${ROOT_DIR}/data"
# Cover setup failures too; the richer training trap below replaces this once
# command/environment capture has completed.
trap 'code=$?; if (( code != 0 )); then printf "state=failed\nexit_code=%s\n" "${code}" > "${RUN_DIR}/status.txt"; fi' EXIT
cp -R "${ROOT_DIR}/config/." "${RUN_DIR}/config/"
CONFIG_FILE="${RUN_DIR}/config/attack/cifar/basee.yaml"
sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" "${CONFIG_FILE}"
ln -s "${ROOT_DIR}/data" "${RUN_DIR}/data"

command=(
  python -u "${ROOT_DIR}/main.py" --dataset cifar --num_attackers 20
  --attack mos_attack --defend1 multi_krum --seed "${SEED}" --gpu "${GPU}" --repeat 1
  --mos_constraint_mode strict --mos_objective_mode dual --mos_adaptive_guided_init 1
  --mos_inject_attack_ray_diagnostics 0 --mos_diagnostics 0
  --mos_sign_shadow_diagnostics 1 --mos_diag_rounds "${DIAG_ROUNDS}"
  --mos_diagnostics_dir .
)

{
  printf 'cd %q\n' "${RUN_DIR}"
  printf 'PYTHONPATH=%q ' "${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "${RUN_DIR}/command.txt"

{
  printf 'GPU=%s\nSEED=%s\nROUNDS=%s\nDIAG_ROUNDS=%s\nANALYZE_AFTER_TRAIN=%s\n' \
    "${GPU}" "${SEED}" "${ROUNDS}" "${DIAG_ROUNDS}" "${ANALYZE_AFTER_TRAIN}"
  printf 'FORMAL_MOS=strict+dual+adaptive_guided_init\n'
  printf 'GIT_COMMIT=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'GIT_STATUS_BEGIN\n'
  git -C "${ROOT_DIR}" status --short 2>/dev/null || true
  printf 'GIT_STATUS_END\n'
  python --version 2>&1
  python -c 'import torch; print("TORCH=" + torch.__version__); print("CUDA=" + str(torch.version.cuda))' 2>&1
} > "${RUN_DIR}/environment.txt"

heartbeat_pid=''
cleanup() {
  local exit_code=$?
  [[ -z "${heartbeat_pid}" ]] || kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true
  printf 'exit_code=%s\nfinished_at=%s\n' "${exit_code}" "$(date --iso-8601=seconds)" >> "${RUN_DIR}/status.txt"
  if (( exit_code != 0 )); then
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "${RUN_DIR}/train.log"; then
      printf 'hint=CUDA_OOM\n' >> "${RUN_DIR}/status.txt"
    elif grep -Eqi 'non-finite|NaN|Inf' "${RUN_DIR}/train.log"; then
      printf 'hint=NON_FINITE\n' >> "${RUN_DIR}/status.txt"
    elif grep -Eqi 'RuntimeError' "${RUN_DIR}/train.log"; then
      printf 'hint=RUNTIME_ERROR\n' >> "${RUN_DIR}/status.txt"
    fi
  fi
  exit "${exit_code}"
}
trap cleanup EXIT
printf 'state=running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${RUN_DIR}/status.txt"
(
  while true; do
    printf '%s alive\n' "$(date --iso-8601=seconds)" >> "${RUN_DIR}/heartbeat.log"
    sleep 60
  done
) &
heartbeat_pid=$!

set +e
(
  cd "${RUN_DIR}"
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${command[@]}"
) 2>&1 | tee "${RUN_DIR}/train.log"
status=${PIPESTATUS[0]}
set -e

awk '
BEGIN { print "round,train_loss,test_acc" }
/^t[[:space:]]+[0-9]+: train_loss =/ {
  round=$2; sub(/:$/, "", round); loss=$5; sub(/,$/, "", loss); acc=$8
  print round "," loss "," acc
}' "${RUN_DIR}/train.log" > "${RUN_DIR}/metrics.csv"

awk -F, '
BEGIN { print "observed_rounds,last_test_acc,mean_test_acc" }
NR > 1 { n++; last=$3; sum+=$3 }
END { print n "," last "," (n ? sum/n : "") }
' "${RUN_DIR}/metrics.csv" > "${RUN_DIR}/summary.csv"

if (( status == 0 && ANALYZE_AFTER_TRAIN == 1 )); then
  set +e
  (
    cd "${RUN_DIR}"
    PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      python -u "${ROOT_DIR}/analyze_mos_sign_snapshots.py" \
      --snapshot_dir . --output_dir .
  ) 2>&1 | tee "${RUN_DIR}/analysis.log"
  analysis_status=${PIPESTATUS[0]}
  set -e
  if (( analysis_status != 0 )); then
    printf 'analysis_exit_code=%s\n' "${analysis_status}" >> "${RUN_DIR}/status.txt"
    status=${analysis_status}
  fi
fi

printf 'state=%s\n' "$([[ ${status} -eq 0 ]] && printf completed || printf failed)" >> "${RUN_DIR}/status.txt"
exit "${status}"
