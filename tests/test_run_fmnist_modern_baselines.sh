#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="${ROOT_DIR}/tests/test_run_fmnist_modern_baselines.sh"
# Stand-in process: no Python imports, datasets, model or training.
if [[ "${1:-}" == -u ]]; then
    [[ " $* " == *' --dataset fmnist '* && " $* " == *' --freeze_datasplit 1 '* ]] || exit 8
    grep -Eq '^num_users: 100' config/attack/fmnist/basee.yaml
    grep -Eq '^num_selected_users: 25' config/attack/fmnist/basee.yaml
    for ((r=0;r<ROUNDS;r++)); do printf 't %s: train_loss = 1.0, test_acc = 0.5\n' "${r}"; done
    exit "${MOCK_EXIT:-0}"
fi
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fmnist-prep.XXXXXX")"
quick="${TEST_ROOT}/quick"; mkdir -p "${quick}"
env RESUME_DIR="${quick}" MOS_BASELINE_PYTHON_BIN="${SELF}" MOS_BASELINE_HEARTBEAT_SECONDS=1 GPU=-1 SMOKE=1 bash "${ROOT_DIR}/run_fmnist_modern_baselines_server.sh"
[[ -f "${quick}/poisonedfl_attack/signguard/seed_1/status.txt" ]]
[[ "$(find "${quick}" -mindepth 3 -maxdepth 3 -type d -name 'seed_*' | wc -l)" == 1 ]]
out="${TEST_ROOT}/matrix"; mkdir -p "${out}"
run() { env RESUME_DIR="${out}" MOS_BASELINE_PYTHON_BIN="${SELF}" MOS_BASELINE_HEARTBEAT_SECONDS=1 GPU=-1 SMOKE=1 "$@" bash "${ROOT_DIR}/run_fmnist_modern_baselines_server.sh"; }
run ATTACKS=non_attack,poisonedfl_attack DEFENSES=multi_krum,tr_mean,signguard
for attack in non_attack poisonedfl_attack; do for defense in multi_krum tr_mean signguard; do
    cell="${out}/${attack}/${defense}/seed_1"
    for f in command.txt environment.txt train.log status.txt heartbeat.log summary.csv; do [[ -f "${cell}/${f}" ]]; done
    grep -qx status=COMPLETED "${cell}/status.txt"
    grep -q fmnist_cnn_u100_s25_m20_iid_splitv2_legacyff10 "${cell}/summary.csv"
    [[ "${attack}" != poisonedfl_attack ]] || grep -q -- '--poisonedfl_scale_factor 8 --poisonedfl_feedback_interval 50' "${cell}/command.txt"
done; done
run ATTACKS=poisonedfl_attack DEFENSES=multi_krum
[[ ! -d "${out}/poisonedfl_attack/multi_krum/seed_1/attempt_2" ]]
[[ "$(wc -l < "${out}/summary.csv")" == 7 ]]
if run ATTACKS=poisonedfl_attack DEFENSES=multi_krum POISONEDFL_SCALE_FACTOR=4; then echo 'FAIL: accepted incompatible resume'; exit 1; fi
out="${TEST_ROOT}/failed"; mkdir -p "${out}"
if run ATTACKS=poisonedfl_attack DEFENSES=signguard MOCK_EXIT=7; then echo 'FAIL: lost process exit code'; exit 1; fi
grep -qx status=FAILED "${out}/poisonedfl_attack/signguard/seed_1/status.txt"
run ATTACKS=poisonedfl_attack DEFENSES=signguard
[[ -f "${out}/poisonedfl_attack/signguard/seed_1/attempt_2/status.txt" ]]
echo "PASS: FMNIST mock matrix, artifacts, filtered resume, mismatch rejection and failed retry (${TEST_ROOT})"
