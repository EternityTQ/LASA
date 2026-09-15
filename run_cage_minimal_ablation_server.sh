#!/usr/bin/env bash
set -Eeuo pipefail

# Stage-two runner only. This script is never invoked by the primary matrix.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
DEFENSES="${DEFENSES:-multi_krum}"
RESUME="${RESUME:-0}"
CLEAN_RESULTS_DIR="${CLEAN_RESULTS_DIR:-}"
OUTPUT_BASE="${ROOT_DIR}/server_experiments/cage_minimal_ablation"
VALID_DEFENSES=(multi_krum tr_mean signguard dnc)

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
contains() { local wanted="$1" item; shift; for item in "$@"; do [[ "${item}" == "${wanted}" ]] && return 0; done; return 1; }
[[ "${GPU}" =~ ^-?[0-9]+$ ]] || die "GPU must be an integer"
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED must be non-negative"
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || die "ROUNDS must be positive"

IFS=',' read -r -a raw_defenses <<< "${DEFENSES}"
RUN_DEFENSES=()
declare -A seen=()
for raw in "${raw_defenses[@]}"; do
  defense="${raw//[[:space:]]/}"
  contains "${defense}" "${VALID_DEFENSES[@]}" || die "unsupported defense '${defense}'"
  if [[ -z "${seen[${defense}]:-}" ]]; then RUN_DEFENSES+=("${defense}"); seen["${defense}"]=1; fi
done
(( ${#RUN_DEFENSES[@]} >= 1 && ${#RUN_DEFENSES[@]} <= 2 )) || die "DEFENSES must contain one or two defenses"
DEFENSE_CSV="$(IFS=,; printf '%s' "${RUN_DEFENSES[*]}")"

mkdir -p "${OUTPUT_BASE}"
case "${RESUME}" in
  ''|0)
    OUTPUT_ROOT="${OUTPUT_BASE}/$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite ${OUTPUT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}"
    ;;
  1)
    OUTPUT_ROOT="$(find "${OUTPUT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    [[ -n "${OUTPUT_ROOT}" && -d "${OUTPUT_ROOT}" ]] || die "RESUME=1 requested but no prior ablation directory exists"
    ;;
  *)
    [[ -d "${RESUME}" ]] || die "RESUME directory does not exist: ${RESUME}"
    OUTPUT_ROOT="$(cd "${RESUME}" && pwd)"
    ;;
esac

run_mode() {
  local label="$1" objective="$2" boundary="$3" mode_root="${OUTPUT_ROOT}/$1" status=0
  mkdir -p "${mode_root}"
  printf '\nRunning ablation mode=%s objective=%s boundary_only=%s\n' "${label}" "${objective}" "${boundary}"
  env GPU="${GPU}" ROUNDS="${ROUNDS}" SEEDS="${SEED}" \
    ATTACKS=mos_attack DEFENSES="${DEFENSE_CSV}" RESUME_DIR="${mode_root}" \
    MOS_OBJECTIVE_MODE="${objective}" MOS_BOUNDARY_ONLY="${boundary}" \
    bash "${ROOT_DIR}/run_mos_baselines_server.sh" || status=$?
  return "${status}"
}

overall_status=0
run_mode full_dual dual 0 || overall_status=1
run_mode strict_a_only a_only 0 || overall_status=1
run_mode boundary_only dual 1 || overall_status=1

# D: run only clean cells that are not already complete in CLEAN_RESULTS_DIR.
# Point CLEAN_RESULTS_DIR at a primary cage_effect_matrix timestamp directory to
# reuse its same-defense clean controls without rerunning them.
MISSING_CLEAN=()
for defense in "${RUN_DEFENSES[@]}"; do
  source_status="${CLEAN_RESULTS_DIR}/non_attack/${defense}/seed_${SEED}/status.txt"
  if [[ -n "${CLEAN_RESULTS_DIR}" && -f "${source_status}" ]] && grep -qx 'status=COMPLETED' "${source_status}"; then
    printf 'Reusing completed clean result: %s\n' "${source_status}"
  else
    MISSING_CLEAN+=("${defense}")
  fi
done
if (( ${#MISSING_CLEAN[@]} )); then
  clean_csv="$(IFS=,; printf '%s' "${MISSING_CLEAN[*]}")"
  mkdir -p "${OUTPUT_ROOT}/clean_baseline"
  env GPU="${GPU}" ROUNDS="${ROUNDS}" SEEDS="${SEED}" \
    ATTACKS=non_attack DEFENSES="${clean_csv}" RESUME_DIR="${OUTPUT_ROOT}/clean_baseline" \
    bash "${ROOT_DIR}/run_mos_baselines_server.sh" || overall_status=1
fi

printf '\nAblation directory: %s\n' "${OUTPUT_ROOT}"
exit "${overall_status}"
