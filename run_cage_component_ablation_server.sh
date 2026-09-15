#!/usr/bin/env bash
set -Eeuo pipefail

# CAGE component ablation: CIFAR, seed=1, 60 rounds, repeat=1.
# Each cell is delegated to the hardened one-process-per-cell server runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
RESUME="${RESUME:-0}"
ONLY_DEFENSE="${ONLY_DEFENSE:-}"
ONLY_VARIANT="${ONLY_VARIANT:-}"
SKIP_FULL="${SKIP_FULL:-0}"
FULL_RESULTS_DIR="${FULL_RESULTS_DIR:-}"
OUTPUT_BASE="${ROOT_DIR}/server_experiments/cage_component_ablation"
ROUNDS=60
SEED=1

ALL_DEFENSES=(tr_mean signguard)
ALL_VARIANTS=(full a_only boundary_only fixed_init)
RUN_DEFENSES=("${ALL_DEFENSES[@]}")
RUN_VARIANTS=("${ALL_VARIANTS[@]}")

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
contains() { local wanted="$1" item; shift; for item in "$@"; do [[ "${item}" == "${wanted}" ]] && return 0; done; return 1; }
join_by_comma() { local IFS=,; printf '%s' "$*"; }

[[ "${GPU}" =~ ^-?[0-9]+$ ]] || die "GPU must be an integer"
[[ "${SKIP_FULL}" == 0 || "${SKIP_FULL}" == 1 ]] || die "SKIP_FULL must be 0 or 1"
if [[ -n "${ONLY_DEFENSE}" ]]; then
  contains "${ONLY_DEFENSE}" "${ALL_DEFENSES[@]}" || die "ONLY_DEFENSE must be tr_mean or signguard"
  RUN_DEFENSES=("${ONLY_DEFENSE}")
fi
if [[ -n "${ONLY_VARIANT}" ]]; then
  contains "${ONLY_VARIANT}" "${ALL_VARIANTS[@]}" || die "ONLY_VARIANT must be one of: ${ALL_VARIANTS[*]}"
  RUN_VARIANTS=("${ONLY_VARIANT}")
fi
if [[ -n "${FULL_RESULTS_DIR}" && ! -d "${FULL_RESULTS_DIR}" ]]; then
  die "FULL_RESULTS_DIR does not exist: ${FULL_RESULTS_DIR}"
fi

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

is_complete() {
  local cell="$1"
  [[ -f "${cell}/status.txt" ]] \
    && grep -qx 'status=COMPLETED' "${cell}/status.txt" \
    && grep -qx "target_rounds=${ROUNDS}" "${cell}/status.txt" \
    && awk -F= -v rounds="${ROUNDS}" '$1=="observed_rounds" && $2+0>=rounds{ok=1}END{exit !ok}' "${cell}/status.txt"
}

is_formal_full() {
  local cell="$1" command="${1}/command.txt"
  is_complete "${cell}" \
    && [[ -f "${cell}/metrics.csv" && -f "${command}" \
          && -f "${cell}/environment.txt" && -f "${cell}/train.log" \
          && -f "${cell}/heartbeat.log" ]] \
    && awk -F, -v rounds="${ROUNDS}" 'END{exit !((NR-1)>=rounds)}' "${cell}/metrics.csv" \
    || return 1
  grep -Eq -- '--dataset[[:space:]]+cifar([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--num_attackers[[:space:]]+20([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--attack[[:space:]]+mos_attack([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--defend1[[:space:]]+'"${2}"'([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--seed[[:space:]]+1([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--repeat[[:space:]]+1([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--mos_constraint_mode[[:space:]]+strict([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--mos_objective_mode[[:space:]]+dual([[:space:]]|$)' "${command}" \
    && grep -Eq -- '--mos_adaptive_guided_init[[:space:]]+1([[:space:]]|$)' "${command}" \
    && ! grep -Eq -- '--mos_boundary_only[[:space:]]+1([[:space:]]|$)' "${command}"
}

find_full_source() {
  local defense="$1" base status cell
  local -a bases=()
  [[ -z "${FULL_RESULTS_DIR}" ]] || bases+=("${FULL_RESULTS_DIR}")
  bases+=("${ROOT_DIR}/server_experiments/cage_effect_matrix" "${ROOT_DIR}/server_experiments/mos_baseline_matrix")
  for base in "${bases[@]}"; do
    [[ -d "${base}" ]] || continue
    while IFS= read -r status; do
      cell="${status%/status.txt}"
      if is_formal_full "${cell}" "${defense}"; then printf '%s\n' "${cell}"; return 0; fi
    done < <(find "${base}" -type f -path "*/mos_attack/${defense}/seed_${SEED}/status.txt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
  done
  return 1
}

reuse_full() {
  local defense="$1" target="${OUTPUT_ROOT}/full/mos_attack/${1}/seed_${SEED}" source
  if is_complete "${target}"; then
    printf 'Skipping completed full cell: defense=%s\n' "${defense}"
    return 0
  fi
  # Preserve an existing incomplete attempt; the baseline runner will create
  # the next attempt instead of mixing it with artifacts from another run.
  [[ ! -d "${target}" ]] || return 1
  source="$(find_full_source "${defense}" || true)"
  [[ -n "${source}" ]] || return 1
  mkdir -p "${target}"
  cp -a "${source}/." "${target}/"
  printf 'reused_from=%s\n' "${source}" >> "${target}/status.txt"
  printf 'REUSED_FROM=%s\n' "${source}" >> "${target}/environment.txt"
  printf 'Reused formal full result: defense=%s source=%s\n' "${defense}" "${source}"
}

run_variant() {
  local variant="$1" objective=dual boundary=0 adaptive=1 defense_csv status=0 defense
  case "${variant}" in
    full) ;;
    a_only) objective=a_only ;;
    boundary_only) boundary=1 ;;
    fixed_init) adaptive=0 ;;
  esac

  local -a pending=()
  for defense in "${RUN_DEFENSES[@]}"; do
    if [[ "${variant}" == full ]] && reuse_full "${defense}"; then continue; fi
    if [[ "${variant}" == full && "${SKIP_FULL}" == 1 ]]; then
      printf 'SKIP_FULL=1: no reusable full result for defense=%s; leaving it unrun.\n' "${defense}"
      continue
    fi
    pending+=("${defense}")
  done
  (( ${#pending[@]} )) || return 0

  defense_csv="$(join_by_comma "${pending[@]}")"
  mkdir -p "${OUTPUT_ROOT}/${variant}"
  printf '\nRunning variant=%s defenses=%s objective=%s boundary_only=%s adaptive_init=%s\n' \
    "${variant}" "${defense_csv}" "${objective}" "${boundary}" "${adaptive}"
  env GPU="${GPU}" ROUNDS="${ROUNDS}" SEEDS="${SEED}" ATTACKS=mos_attack \
    DEFENSES="${defense_csv}" RESUME_DIR="${OUTPUT_ROOT}/${variant}" STOP_ON_ERROR=0 \
    MOS_OBJECTIVE_MODE="${objective}" MOS_BOUNDARY_ONLY="${boundary}" \
    MOS_ADAPTIVE_GUIDED_INIT="${adaptive}" \
    bash "${ROOT_DIR}/run_mos_baselines_server.sh" || status=$?
  return "${status}"
}

write_ablation_summary() {
  python - "${OUTPUT_ROOT}" "${ROUNDS}" "${SEED}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
target_rounds = int(sys.argv[2])
seed = sys.argv[3]
defenses = ('tr_mean', 'signguard')
variants = ('full', 'a_only', 'boundary_only', 'fixed_init')
fields = ('defense', 'variant', 'observed_rounds', 'final_acc', 'last10_mean_acc',
          'mean_acc', 'min_acc', 'completed', 'exit_code', 'mean_A',
          'last10_mean_A', 'mean_R', 'mean_CV', 'mean_alpha_feasible')

def status_values(path):
    values = {}
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                values[key] = value
    return values

def numbers(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ''))
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values

def mean(values):
    return sum(values) / len(values) if values else None

def fmt(value):
    return '' if value is None else f'{value:.8f}'

summary = []
for defense in defenses:
    for variant in variants:
        cell = root / variant / 'mos_attack' / defense / f'seed_{seed}'
        status = status_values(cell / 'status.txt')
        rows = []
        metrics = cell / 'metrics.csv'
        if metrics.exists():
            with metrics.open(newline='', errors='replace') as handle:
                rows = list(csv.DictReader(handle))
        acc = numbers(rows, 'test_acc')
        avals = numbers(rows, 'selected_A')
        rvals = numbers(rows, 'selected_R')
        cvvals = numbers(rows, 'selected_CV')
        alphas = numbers(rows, 'alpha_feasible')
        observed = len(rows)
        completed = status.get('status', '').upper() == 'COMPLETED' and observed >= target_rounds
        summary.append({
            'defense': defense, 'variant': variant, 'observed_rounds': observed,
            'final_acc': fmt(acc[-1]) if acc else '',
            'last10_mean_acc': fmt(mean(acc[-10:])), 'mean_acc': fmt(mean(acc)),
            'min_acc': fmt(min(acc)) if acc else '',
            'completed': '1' if completed else '0', 'exit_code': status.get('exit_code', ''),
            'mean_A': fmt(mean(avals)), 'last10_mean_A': fmt(mean(avals[-10:])),
            'mean_R': fmt(mean(rvals)), 'mean_CV': fmt(mean(cvvals)),
            'mean_alpha_feasible': fmt(mean(alphas)),
        })
with (root / 'ablation_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)
PY
}

printf 'CAGE component ablation directory: %s\n' "${OUTPUT_ROOT}"
printf 'Defenses: %s\nVariants: %s\nCIFAR seed=%s rounds=%s repeat=1 GPU=%s\n' \
  "${RUN_DEFENSES[*]}" "${RUN_VARIANTS[*]}" "${SEED}" "${ROUNDS}" "${GPU}"

overall_status=0
for variant in "${RUN_VARIANTS[@]}"; do
  run_variant "${variant}" || overall_status=1
  write_ablation_summary
done
write_ablation_summary
printf 'Summary: %s/ablation_summary.csv\n' "${OUTPUT_ROOT}"
exit "${overall_status}"
