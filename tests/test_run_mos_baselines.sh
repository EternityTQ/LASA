#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/run_mos_baselines_server.sh"
SELF="${ROOT_DIR}/tests/test_run_mos_baselines.sh"

# Stand-in Python process: orchestration smoke tests never train a model.
if [[ "${1:-}" == */main.py ]]; then
    attack="" defense="" seed=""
    while (( $# )); do
        case "$1" in --attack)attack="$2";shift 2;;--defend1)defense="$2";shift 2;;--seed)seed="$2";shift 2;;*)shift;;esac
    done
    acc="$(awk -v s="${seed}" -v a="${attack}" 'BEGIN{printf "%.3f",0.50+s/100+(a=="signflip_attack"?-0.10:0)}')"
    printf '[FiniteAudit] round=0 global_post_finite=True first_bad_stage=none rollback=False\n'
    printf 't 0: train_loss = 1.2500, test_acc = %s\n' "${acc}"
    [[ "${MOCK_SLEEP:-0}" == 0 ]] || sleep "${MOCK_SLEEP}"
    if [[ "${attack}" == "${MOCK_FAIL_ATTACK:-}" && "${defense}" == "${MOCK_FAIL_DEFENSE:-${defense}}" ]]; then
        printf 'Traceback (most recent call last):\nRuntimeError: injected smoke-test failure\n' >&2;exit "${MOCK_EXIT_CODE:-7}"
    fi
    exit 0
fi

fail(){ printf 'FAIL: %s\n' "$*" >&2;exit 1; }
assert_file(){ [[ -f "$1" ]]||fail "missing file $1"; }
assert_status(){ grep -qx "status=$2" "$1/status.txt"||fail "expected $1 status $2"; }
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mos-matrix-smoke.XXXXXX")"
printf 'Smoke workspace: %s\n' "${TEST_ROOT}"
run_launcher(){ local out="$1";shift;mkdir -p "${out}";env RESUME_DIR="${out}" GPU=-1 ROUNDS=1 MOS_BASELINE_PYTHON_BIN="${SELF}" MOS_BASELINE_HEARTBEAT_SECONDS=1 "$@" bash "${SCRIPT}"; }

# Required 2 attack x 2 defense, 1-round matrix. Each cell is a process.
basic="${TEST_ROOT}/basic"
run_launcher "${basic}" ATTACKS=non_attack,mos_attack DEFENSES=fedavg,multi_krum SEEDS=1
for attack in non_attack mos_attack;do for defense in fedavg multi_krum;do
    cell="${basic}/${attack}/${defense}/seed_1";assert_status "${cell}" COMPLETED
    for artifact in train.log command.txt environment.txt status.txt metrics.csv summary.csv heartbeat.log;do assert_file "${cell}/${artifact}";done
    grep -q -- '--defend1' "${cell}/command.txt"||fail '--defend1 missing'
    grep -q -- '--repeat 1' "${cell}/command.txt"||fail '--repeat 1 missing'
    if grep -Eq -- '--defend2|--defend3|--defend([ =]|$)' "${cell}/command.txt";then fail 'multi-defense/obsolete flag used';fi
done;done
grep -q -- '--mos_adaptive_guided_init 1' "${basic}/mos_attack/fedavg/seed_1/command.txt"||fail 'MOS adaptive init missing'
grep -q -- '--mos_constraint_mode strict' "${basic}/mos_attack/fedavg/seed_1/command.txt"||fail 'MOS strict mode missing'
grep -q -- '--mos_objective_mode dual' "${basic}/mos_attack/fedavg/seed_1/command.txt"||fail 'MOS dual mode missing'
if grep -q -- '--mos_' "${basic}/non_attack/fedavg/seed_1/command.txt";then fail 'baseline received MOS-only arguments';fi
for f in results_long.csv matrix_last10_acc.csv matrix_final_acc.csv matrix_acc_drop_vs_clean.csv;do assert_file "${basic}/${f}";done
awk -F, '$1=="mos_attack"{if(($2=="fedavg"&&$3!="0.100000")||($2=="multi_krum"&&$3!="0.100000"))exit 1;seen++}END{if(seen!=1)exit 1}' "${basic}/matrix_acc_drop_vs_clean.csv"||fail 'clean-relative drop is wrong'

# A failed cell is diagnosed and does not prevent the next cell.
continued="${TEST_ROOT}/continued";set +e
run_launcher "${continued}" ATTACKS=signflip_attack,non_attack DEFENSES=fedavg SEEDS=1 MOCK_FAIL_ATTACK=signflip_attack MOCK_FAIL_DEFENSE=fedavg MOCK_EXIT_CODE=7
rc=$?;set -e;[[ "${rc}" == 1 ]]||fail "failure run returned ${rc}"
assert_status "${continued}/signflip_attack/fedavg/seed_1" FAILED
assert_file "${continued}/signflip_attack/fedavg/seed_1/failure_diagnostics.txt"
grep -qx 'exit_code=7' "${continued}/signflip_attack/fedavg/seed_1/status.txt"||fail 'exit code missing'
grep -q 'RuntimeError: injected' "${continued}/signflip_attack/fedavg/seed_1/status.txt"||fail 'error hint missing'
assert_status "${continued}/non_attack/fedavg/seed_1" COMPLETED

# STOP_ON_ERROR leaves subsequent cells untouched.
stopped="${TEST_ROOT}/stopped";set +e
run_launcher "${stopped}" ATTACKS=signflip_attack,non_attack DEFENSES=fedavg SEEDS=1 STOP_ON_ERROR=1 MOCK_FAIL_ATTACK=signflip_attack MOCK_FAIL_DEFENSE=fedavg
rc=$?;set -e;[[ "${rc}" == 1 ]]||fail "STOP_ON_ERROR run returned ${rc}"
[[ ! -d "${stopped}/non_attack" ]]||fail 'STOP_ON_ERROR started a later cell'

# Resume keeps attempt_1, retries failure, and skips the completed cell.
run_launcher "${continued}" ATTACKS=signflip_attack,non_attack DEFENSES=fedavg SEEDS=1
assert_status "${continued}/signflip_attack/fedavg/seed_1" COMPLETED
grep -qx status=FAILED "${continued}/signflip_attack/fedavg/seed_1/attempt_1/status.txt"||fail 'failed attempt overwritten'
assert_file "${continued}/signflip_attack/fedavg/seed_1/attempt_2/status.txt"
[[ ! -d "${continued}/non_attack/fedavg/seed_1/attempt_2" ]]||fail 'completed cell reran'

# Multiple seeds produce numeric mean/std and paired clean drops.
multi="${TEST_ROOT}/multi";run_launcher "${multi}" ATTACKS=non_attack,signflip_attack DEFENSES=fedavg SEEDS=1,2
for f in matrix_last10_mean.csv matrix_last10_std.csv matrix_drop_mean.csv matrix_drop_std.csv matrix_paper_preview.csv;do assert_file "${multi}/${f}";done
awk -F, '$1=="non_attack"{if($2!="0.515000")exit 1;seen=1}END{if(!seen)exit 1}' "${multi}/matrix_last10_mean.csv"||fail 'multi-seed mean wrong'
awk -F, '$1=="non_attack"{if($2!="0.005000")exit 1;seen=1}END{if(!seen)exit 1}' "${multi}/matrix_last10_std.csv"||fail 'multi-seed std wrong'
awk -F, '$1=="signflip_attack"{if($2!="0.100000")exit 1;seen=1}END{if(!seen)exit 1}' "${multi}/matrix_drop_mean.csv"||fail 'multi-seed drop wrong'

# Heartbeat stops with its child.
heartbeat="${TEST_ROOT}/heartbeat";run_launcher "${heartbeat}" ATTACKS=lie_attack DEFENSES=dnc SEEDS=1 MOCK_SLEEP=2
hb="${heartbeat}/lie_attack/dnc/seed_1/heartbeat.log";[[ -s "${hb}" ]]||fail 'heartbeat empty';lines="$(wc -l < "${hb}")";sleep 2;[[ "$(wc -l < "${hb}")" == "${lines}" ]]||fail 'heartbeat did not stop'
printf 'PASS: MOS baseline matrix smoke tests\n'
