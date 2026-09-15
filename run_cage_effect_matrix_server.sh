#!/usr/bin/env bash
set -Eeuo pipefail

# Primary 5 x 4 CAGE effectiveness matrix. Each cell is delegated to the
# repository's hardened, one-process-per-cell baseline runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
RESUME="${RESUME:-0}"
ONLY_ATTACK="${ONLY_ATTACK:-}"
ONLY_DEFENSE="${ONLY_DEFENSE:-}"
OUTPUT_BASE="${ROOT_DIR}/server_experiments/cage_effect_matrix"

ALL_ATTACKS=(non_attack signflip_attack lie_attack agrAgnosticMinMax mos_attack)
ALL_DEFENSES=(multi_krum tr_mean signguard dnc)
RUN_ATTACKS=("${ALL_ATTACKS[@]}")
RUN_DEFENSES=("${ALL_DEFENSES[@]}")

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
contains() { local wanted="$1" item; shift; for item in "$@"; do [[ "${item}" == "${wanted}" ]] && return 0; done; return 1; }
join_by_comma() { local IFS=,; printf '%s' "$*"; }

[[ "${GPU}" =~ ^-?[0-9]+$ ]] || die "GPU must be an integer"
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED must be non-negative"
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || die "ROUNDS must be positive"
if [[ -n "${ONLY_ATTACK}" ]]; then
  contains "${ONLY_ATTACK}" "${ALL_ATTACKS[@]}" || die "ONLY_ATTACK must be one of: ${ALL_ATTACKS[*]}"
  RUN_ATTACKS=("${ONLY_ATTACK}")
fi
if [[ -n "${ONLY_DEFENSE}" ]]; then
  contains "${ONLY_DEFENSE}" "${ALL_DEFENSES[@]}" || die "ONLY_DEFENSE must be one of: ${ALL_DEFENSES[*]}"
  RUN_DEFENSES=("${ONLY_DEFENSE}")
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
    [[ -n "${OUTPUT_ROOT}" && -d "${OUTPUT_ROOT}" ]] || die "RESUME=1 requested but no prior matrix directory exists"
    ;;
  *)
    [[ -d "${RESUME}" ]] || die "RESUME directory does not exist: ${RESUME}"
    OUTPUT_ROOT="$(cd "${RESUME}" && pwd)"
    ;;
esac

printf 'CAGE effect matrix directory: %s\n' "${OUTPUT_ROOT}"
printf 'Attacks: %s\nDefenses: %s\nSeed: %s  Rounds: %s  GPU: %s\n' \
  "${RUN_ATTACKS[*]}" "${RUN_DEFENSES[*]}" "${SEED}" "${ROUNDS}" "${GPU}"

set +e
env \
  GPU="${GPU}" ROUNDS="${ROUNDS}" SEEDS="${SEED}" \
  ATTACKS="$(join_by_comma "${RUN_ATTACKS[@]}")" \
  DEFENSES="$(join_by_comma "${RUN_DEFENSES[@]}")" \
  RESUME_DIR="${OUTPUT_ROOT}" \
  bash "${ROOT_DIR}/run_mos_baselines_server.sh"
runner_status=$?
set -e

# Build the paper-facing summary from per-cell status.txt and metrics.csv.
# Metrics and clean-relative drops are intentionally blank unless the cell and
# its same-defense clean reference are both formally complete.
python - "${OUTPUT_ROOT}" "${ROUNDS}" \
  "$(join_by_comma "${ALL_ATTACKS[@]}")" "$(join_by_comma "${ALL_DEFENSES[@]}")" "${SEED}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
target_rounds = int(sys.argv[2])
attacks = sys.argv[3].split(',')
defenses = sys.argv[4].split(',')
seed = sys.argv[5]

def read_status(path):
    values = {}
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                values[key] = value
    return values

def read_acc(path):
    values = []
    if path.exists():
        with path.open(newline='', errors='replace') as handle:
            for row in csv.DictReader(handle):
                try:
                    values.append(float(row['test_acc']))
                except (KeyError, TypeError, ValueError):
                    pass
    return values

def mean_tail(values, n):
    tail = values[-n:]
    return sum(tail) / len(tail) if tail else None

def fmt(value):
    return '' if value is None else f'{value:.8f}'

rows = []
for defense in defenses:
    for attack in attacks:
        cell = root / attack / defense / f'seed_{seed}'
        status = read_status(cell / 'status.txt')
        acc = read_acc(cell / 'metrics.csv')
        exit_code = status.get('exit_code', '')
        recorded = status.get('status', '').upper()
        completed = recorded == 'COMPLETED' and len(acc) >= target_rounds
        if completed:
            state = 'completed'
        elif recorded == 'FAILED' and exit_code not in ('', '0'):
            state = 'failed'
        else:
            state = 'incomplete'
        row = {
            'attack': attack, 'defense': defense, 'seed': seed,
            'status': state, 'completed': '1' if completed else '0',
            'observed_rounds': len(acc),
            'final_test_acc': fmt(acc[-1]) if completed else '',
            'last10_mean_acc': fmt(mean_tail(acc, 10)) if completed else '',
            'last20_mean_acc': fmt(mean_tail(acc, 20)) if completed else '',
            'min_test_acc': fmt(min(acc)) if completed else '',
            'runtime_seconds': status.get('elapsed_seconds', ''),
            'exit_code': exit_code,
            'final_clean_relative_drop': '',
            'last10_clean_relative_drop': '',
            'last20_clean_relative_drop': '',
        }
        rows.append(row)

by_key = {(row['attack'], row['defense']): row for row in rows}
for row in rows:
    clean = by_key.get(('non_attack', row['defense']))
    if row['completed'] != '1' or not clean or clean['completed'] != '1':
        continue
    for metric, drop_name in (
        ('final_test_acc', 'final_clean_relative_drop'),
        ('last10_mean_acc', 'last10_clean_relative_drop'),
        ('last20_mean_acc', 'last20_clean_relative_drop'),
    ):
        clean_value = float(clean[metric])
        attacked_value = float(row[metric])
        if clean_value != 0:
            row[drop_name] = fmt((clean_value - attacked_value) / clean_value)

fields = [
    'attack', 'defense', 'seed', 'status', 'completed', 'observed_rounds',
    'final_test_acc', 'last10_mean_acc', 'last20_mean_acc', 'min_test_acc',
    'runtime_seconds', 'exit_code', 'final_clean_relative_drop',
    'last10_clean_relative_drop', 'last20_clean_relative_drop',
]
with (root / 'matrix_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

printf 'Summary: %s/matrix_summary.csv\n' "${OUTPUT_ROOT}"
exit "${runner_status}"
