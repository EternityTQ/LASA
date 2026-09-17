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
AUDIT_ONLY="${AUDIT_ONLY:-0}"
OUTPUT_BASE="${ROOT_DIR}/server_experiments/cage_component_ablation"
ROUNDS=60
SEED=1

ALL_DEFENSES=(tr_mean signguard)
ALL_VARIANTS=(full a_only boundary_only fixed_init no_radial)
RUN_DEFENSES=("${ALL_DEFENSES[@]}")
RUN_VARIANTS=("${ALL_VARIANTS[@]}")

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
contains() { local wanted="$1" item; shift; for item in "$@"; do [[ "${item}" == "${wanted}" ]] && return 0; done; return 1; }
join_by_comma() { local IFS=,; printf '%s' "$*"; }

[[ "${GPU}" =~ ^-?[0-9]+$ ]] || die "GPU must be an integer"
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || die "ROUNDS must be positive"
[[ "${SKIP_FULL}" == 0 || "${SKIP_FULL}" == 1 ]] || die "SKIP_FULL must be 0 or 1"
[[ "${AUDIT_ONLY}" == 0 || "${AUDIT_ONLY}" == 1 ]] || die "AUDIT_ONLY must be 0 or 1"
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
    && ! grep -Eq -- '--mos_boundary_only[[:space:]]+1([[:space:]]|$)' "${command}" \
    && ! grep -Eq -- '--mos_use_radial_constraint[[:space:]]+0([[:space:]]|$)' "${command}"
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
  local variant="$1" objective=dual boundary=0 adaptive=1 radial=1 defense_csv status=0 defense
  case "${variant}" in
    full) ;;
    a_only) objective=a_only ;;
    boundary_only) objective=a_only; boundary=1 ;;
    fixed_init) adaptive=0 ;;
    no_radial) radial=0 ;;
  esac

  local -a pending=()
  for defense in "${RUN_DEFENSES[@]}"; do
    # Smoke tests must exercise the live full path.  Historical reuse is only
    # eligible for the paper-facing 60-round run.
    if [[ "${variant}" == full && "${ROUNDS}" == 60 ]] && reuse_full "${defense}"; then continue; fi
    if [[ "${variant}" == full && "${SKIP_FULL}" == 1 ]]; then
      printf 'SKIP_FULL=1: no reusable full result for defense=%s; leaving it unrun.\n' "${defense}"
      continue
    fi
    pending+=("${defense}")
  done
  (( ${#pending[@]} )) || return 0

  defense_csv="$(join_by_comma "${pending[@]}")"
  mkdir -p "${OUTPUT_ROOT}/${variant}"
  printf '\nRunning variant=%s defenses=%s objective=%s boundary_only=%s adaptive_init=%s radial=%s\n' \
    "${variant}" "${defense_csv}" "${objective}" "${boundary}" "${adaptive}" "${radial}"
  env GPU="${GPU}" ROUNDS="${ROUNDS}" SEEDS="${SEED}" ATTACKS=mos_attack \
    DEFENSES="${defense_csv}" RESUME_DIR="${OUTPUT_ROOT}/${variant}" STOP_ON_ERROR=0 \
    MOS_OBJECTIVE_MODE="${objective}" MOS_BOUNDARY_ONLY="${boundary}" \
    MOS_ADAPTIVE_GUIDED_INIT="${adaptive}" MOS_USE_RADIAL_CONSTRAINT="${radial}" \
    MOS_RUN_VARIANT="${variant}" \
    bash "${ROOT_DIR}/run_mos_baselines_server.sh" || status=$?
  return "${status}"
}

write_ablation_outputs() {
  local strict="${1:-0}"
  python - "${OUTPUT_ROOT}" "${ROUNDS}" "${SEED}" \
    "$(join_by_comma "${RUN_DEFENSES[@]}")" "$(join_by_comma "${RUN_VARIANTS[@]}")" "${strict}" <<'PY'
import csv
import hashlib
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
target_rounds = int(sys.argv[2])
seed = sys.argv[3]
selected_defenses = set(sys.argv[4].split(','))
selected_variants = set(sys.argv[5].split(','))
strict = sys.argv[6] == '1'
defenses = ('tr_mean', 'signguard')
variants = ('full', 'a_only', 'boundary_only', 'fixed_init', 'no_radial')
summary_fields = [
    'defense', 'variant', 'observed_rounds', 'final_acc', 'last10_mean_acc',
    'last20_mean_acc', 'last20_std_acc', 'mean_acc', 'min_acc', 'completed',
    'exit_code', 'mean_A', 'last10_mean_A', 'last20_mean_A', 'mean_R',
    'mean_CV', 'mean_alpha_feasible', 'nonfinite_count', 'rollback_count',
    'config_verified', 'metrics_path', 'train_log_sha256']
audit_fields = (
    'defense', 'variant', 'cell_dir', 'command_path', 'metrics_path',
    'train_log_path', 'status', 'observed_rounds', 'command_sha256',
    'metrics_sha256', 'train_log_sha256', 'expected_config',
    'observed_configs', 'command_matches', 'config_matches',
    'evolutionary_iterations', 'reused_from', 'duplicate_train_log_of',
    'first_error_path', 'first_error')

expected = {
    'full': ('dual', '1', '0', '1', 'evolutionary', 'radial,sign', '1'),
    'a_only': ('a_only', '1', '0', '1', 'evolutionary', 'radial,sign', '1'),
    'boundary_only': ('a_only', '1', '1', '1', 'boundary_only', 'radial,sign', '0'),
    'fixed_init': ('dual', '0', '0', '1', 'evolutionary', 'radial,sign', '1'),
    'no_radial': ('dual', '1', '0', '0', 'evolutionary', 'sign', '1'),
}

def status_values(path):
    values = {}
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                values[key] = value
    return values

def sha256(path):
    if not path.exists():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def flag(command, name):
    match = re.search(r'(?:^|\s)--' + re.escape(name) + r'\s+([^\s]+)', command)
    return match.group(1).replace('\\ ', ' ') if match else None

def numbers(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ''))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values

def mean(values):
    return sum(values) / len(values) if values else None

def std(values):
    if not values:
        return None
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))

def fmt(value):
    return '' if value is None else f'{value:.8f}'

def finite_audit_counts(log_text):
    nonfinite_rounds = set()
    rollback_rounds = set()
    for line in log_text.splitlines():
        if not line.startswith('[FiniteAudit]'):
            continue
        match = re.search(r'\bround=(\d+)\b', line)
        if not match:
            continue
        round_number = int(match.group(1))
        if re.search(r'\bnonfinite_count=[1-9]\d*\b', line) or re.search(
                r'\bfirst_bad_stage=(?!none\b)\S+', line):
            nonfinite_rounds.add(round_number)
        if re.search(r'\brollback=True\b', line):
            rollback_rounds.add(round_number)
    return len(nonfinite_rounds), len(rollback_rounds)

metric_fields = set()
for defense in defenses:
    for variant in variants:
        metrics = root / variant / 'mos_attack' / defense / f'seed_{seed}' / 'metrics.csv'
        if metrics.exists():
            with metrics.open(newline='', errors='replace') as handle:
                metric_fields.update(csv.DictReader(handle).fieldnames or ())
has_selected_norm = 'selected_norm' in metric_fields
alignment_field = next((field for field in (
    'selected_alignment', 'selected_guidance_alignment') if field in metric_fields), None)
if has_selected_norm:
    summary_fields[summary_fields.index('nonfinite_count'):summary_fields.index('nonfinite_count')] = [
        'mean_selected_norm', 'last20_mean_selected_norm']
if alignment_field:
    summary_fields[summary_fields.index('nonfinite_count'):summary_fields.index('nonfinite_count')] = [
        'mean_selected_alignment', 'last20_mean_selected_alignment']

summary = []
audit = []
for defense in defenses:
    for variant in variants:
        cell = root / variant / 'mos_attack' / defense / f'seed_{seed}'
        status_path = cell / 'status.txt'
        command_path = cell / 'command.txt'
        environment_path = cell / 'environment.txt'
        metrics = cell / 'metrics.csv'
        train_log = cell / 'train.log'
        status = status_values(status_path)
        environment = status_values(environment_path)
        command = command_path.read_text(errors='replace') if command_path.exists() else ''
        log_text = train_log.read_text(errors='replace') if train_log.exists() else ''
        rows = []
        if metrics.exists():
            with metrics.open(newline='', errors='replace') as handle:
                rows = list(csv.DictReader(handle))
        acc = numbers(rows, 'test_acc')
        avals = numbers(rows, 'selected_A')
        rvals = numbers(rows, 'selected_R')
        cvvals = numbers(rows, 'selected_CV')
        alphas = numbers(rows, 'alpha_feasible')
        selected_norms = numbers(rows, 'selected_norm') if has_selected_norm else []
        selected_alignments = numbers(rows, alignment_field) if alignment_field else []
        observed = len(rows)
        objective, adaptive, boundary, radial, search_mode, constraints, search_enabled = expected[variant]
        # Existing boundary-only runs used objective=dual, but this path selects
        # its sole directional boundary candidate directly.  Accept those runs
        # while all future invocations above use the paper-facing a_only label.
        allowed_objectives = (objective, 'dual') if variant == 'boundary_only' else (objective,)
        expected_configs = tuple(
            f'[CAGE_CONFIG] objective={allowed_objective} adaptive_init={adaptive} '
            f'search_mode={search_mode} constraints={constraints} '
            f'evolutionary_search_enabled={search_enabled}'
            for allowed_objective in allowed_objectives)
        expected_config = ' | '.join(expected_configs)
        observed_configs = sorted(set(
            line.strip() for line in log_text.splitlines()
            if line.startswith('[CAGE_CONFIG]')))
        reused_from = status.get('reused_from', '')
        variant_tag_ok = environment.get('MOS_RUN_VARIANT') == variant
        command_matches = bool(command) and all((
            flag(command, 'dataset') == 'cifar',
            flag(command, 'attack') == 'mos_attack',
            flag(command, 'defend1') == defense,
            flag(command, 'seed') == seed,
            flag(command, 'repeat') == '1',
            flag(command, 'mos_constraint_mode') == 'strict',
            flag(command, 'mos_objective_mode') in allowed_objectives,
            flag(command, 'mos_adaptive_guided_init') == adaptive,
            flag(command, 'mos_boundary_only') == boundary,
            (flag(command, 'mos_use_radial_constraint') == radial
             or (variant == 'full' and reused_from and
                 flag(command, 'mos_use_radial_constraint') is None and radial == '1')),
            variant_tag_ok or (variant == 'full' and bool(reused_from)),
        ))
        evolutionary_iterations = len(re.findall(r'^\[MOS-Core\] Gen ', log_text, re.MULTILINE))
        runtime_matches = any(config in observed_configs for config in expected_configs)
        if variant == 'boundary_only' and evolutionary_iterations:
            runtime_matches = False
        config_matches = command_matches and (
            runtime_matches or (variant == 'full' and bool(reused_from) and not observed_configs))
        completed = (status.get('status', '').upper() == 'COMPLETED'
                     and observed >= target_rounds and config_matches)

        first_error = ''
        first_error_path = ''
        if log_text and status.get('status', '').upper() == 'FAILED':
            lines = log_text.splitlines()
            start = next((i for i, line in enumerate(lines) if 'Traceback (most recent call last)' in line), None)
            if start is None:
                pattern = re.compile(r'RuntimeError|CUDA out of memory|OutOfMemoryError|non-finite|\bError\b|Exception|Killed')
                start = next((i for i, line in enumerate(lines) if pattern.search(line)), None)
            if start is not None:
                excerpt = '\n'.join(lines[start:start + 50]) + '\n'
                error_file = cell / 'first_error.txt'
                error_file.write_text(excerpt, errors='replace')
                first_error_path = str(error_file.resolve())
                first_error = lines[start]

        nonfinite_count, rollback_count = finite_audit_counts(log_text)
        summary_row = {
            'defense': defense, 'variant': variant, 'observed_rounds': observed,
            'final_acc': fmt(acc[-1]) if acc else '',
            'last10_mean_acc': fmt(mean(acc[-10:])), 'mean_acc': fmt(mean(acc)),
            'last20_mean_acc': fmt(mean(acc[-20:])),
            'last20_std_acc': fmt(std(acc[-20:])),
            'min_acc': fmt(min(acc)) if acc else '',
            'completed': '1' if completed else '0', 'exit_code': status.get('exit_code', ''),
            'mean_A': fmt(mean(avals)), 'last10_mean_A': fmt(mean(avals[-10:])),
            'last20_mean_A': fmt(mean(avals[-20:])),
            'mean_R': fmt(mean(rvals)), 'mean_CV': fmt(mean(cvvals)),
            'mean_alpha_feasible': fmt(mean(alphas)),
            'nonfinite_count': nonfinite_count, 'rollback_count': rollback_count,
            'config_verified': '1' if config_matches else '0',
            'metrics_path': str(metrics.resolve()),
            'train_log_sha256': sha256(train_log),
        }
        if has_selected_norm:
            summary_row.update({
                'mean_selected_norm': fmt(mean(selected_norms)),
                'last20_mean_selected_norm': fmt(mean(selected_norms[-20:])),
            })
        if alignment_field:
            summary_row.update({
                'mean_selected_alignment': fmt(mean(selected_alignments)),
                'last20_mean_selected_alignment': fmt(mean(selected_alignments[-20:])),
            })
        summary.append(summary_row)
        audit.append({
            'defense': defense, 'variant': variant, 'cell_dir': str(cell.resolve()),
            'command_path': str(command_path.resolve()), 'metrics_path': str(metrics.resolve()),
            'train_log_path': str(train_log.resolve()), 'status': status.get('status', ''),
            'observed_rounds': observed, 'command_sha256': sha256(command_path),
            'metrics_sha256': sha256(metrics), 'train_log_sha256': sha256(train_log),
            'expected_config': expected_config, 'observed_configs': ' | '.join(observed_configs),
            'command_matches': '1' if command_matches else '0',
            'config_matches': '1' if config_matches else '0',
            'evolutionary_iterations': evolutionary_iterations,
            'reused_from': reused_from, 'duplicate_train_log_of': '',
            'first_error_path': first_error_path, 'first_error': first_error,
        })

first_by_hash = {}
duplicates = []
for row in audit:
    digest = row['train_log_sha256']
    if not digest:
        continue
    key = (row['defense'], digest)
    if key in first_by_hash:
        row['duplicate_train_log_of'] = first_by_hash[key]
        duplicates.append(row)
    else:
        first_by_hash[key] = row['variant']

with (root / 'ablation_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary)
with (root / 'ablation_audit.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=audit_fields)
    writer.writeheader()
    writer.writerows(audit)

for row in duplicates:
    print(f"WARNING: identical train.log defense={row['defense']} "
          f"variant={row['variant']} duplicate_of={row['duplicate_train_log_of']} "
          f"sha256={row['train_log_sha256']}", file=sys.stderr)

if strict:
    failures = []
    for row in audit:
        if row['defense'] not in selected_defenses or row['variant'] not in selected_variants:
            continue
        if row['status'] and (row['command_matches'] != '1' or row['config_matches'] != '1'):
            failures.append(f"{row['defense']}/{row['variant']}: configuration provenance failed")
        if row['duplicate_train_log_of'] and row['duplicate_train_log_of'] in selected_variants:
            failures.append(f"{row['defense']}/{row['variant']}: train.log duplicates {row['duplicate_train_log_of']}")
    if failures:
        for failure in failures:
            print('AUDIT FAILURE: ' + failure, file=sys.stderr)
        raise SystemExit(1)
PY
}

printf 'CAGE component ablation directory: %s\n' "${OUTPUT_ROOT}"
printf 'Defenses: %s\nVariants: %s\nCIFAR seed=%s rounds=%s repeat=1 GPU=%s\n' \
  "${RUN_DEFENSES[*]}" "${RUN_VARIANTS[*]}" "${SEED}" "${ROUNDS}" "${GPU}"

overall_status=0
if [[ "${AUDIT_ONLY}" == 0 ]]; then
  for variant in "${RUN_VARIANTS[@]}"; do
    run_variant "${variant}" || overall_status=1
    write_ablation_outputs 0 || overall_status=1
  done
else
  printf 'AUDIT_ONLY=1: no experiments will be started.\n'
fi
write_ablation_outputs 1 || overall_status=1
printf 'Summary: %s/ablation_summary.csv\n' "${OUTPUT_ROOT}"
printf 'Audit: %s/ablation_audit.csv\n' "${OUTPUT_ROOT}"
exit "${overall_status}"
