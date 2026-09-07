#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
DIAG_ROUNDS="${DIAG_ROUNDS:-0,20,40,59}"
DIAG_CANDIDATES="${DIAG_CANDIDATES:-16}"

[[ "${GPU}" =~ ^-?[0-9]+$ ]] || { echo "GPU must be an integer" >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative" >&2; exit 2; }
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { echo "ROUNDS must be positive" >&2; exit 2; }
[[ "${DIAG_CANDIDATES}" =~ ^[1-9][0-9]*$ ]] || { echo "DIAG_CANDIDATES must be positive" >&2; exit 2; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/server_experiments/mos_first_batch_diagnostics/${TIMESTAMP}"
CONFIG_FILE="${RUN_DIR}/config/attack/cifar/basee.yaml"
mkdir -p "${RUN_DIR}/config" "${ROOT_DIR}/data"
cp -R "${ROOT_DIR}/config/." "${RUN_DIR}/config/"
sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" "${CONFIG_FILE}"
ln -s "${ROOT_DIR}/data" "${RUN_DIR}/data"

command=(
    python "${ROOT_DIR}/main.py" --dataset cifar --num_attackers 20
    --attack mos_attack --defend1 multi_krum --seed "${SEED}" --gpu "${GPU}" --repeat 1
    --mos_constraint_mode strict --mos_objective_mode dual --mos_adaptive_guided_init 1
    --mos_inject_attack_ray_diagnostics 0 --mos_diagnostics 1
    --mos_diag_rounds "${DIAG_ROUNDS}" --mos_diag_candidates "${DIAG_CANDIDATES}"
    --mos_diagnostics_dir .
)

{
    printf 'cd %q\n' "${RUN_DIR}"
    printf 'PYTHONPATH=%q ' "${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "${RUN_DIR}/command.txt"

{
    printf 'GPU=%s\nSEED=%s\nROUNDS=%s\nDIAG_ROUNDS=%s\nDIAG_CANDIDATES=%s\n' \
        "${GPU}" "${SEED}" "${ROUNDS}" "${DIAG_ROUNDS}" "${DIAG_CANDIDATES}"
    printf 'GIT_COMMIT=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'GIT_STATUS_BEGIN\n'
    git -C "${ROOT_DIR}" status --short 2>/dev/null || true
    printf 'GIT_STATUS_END\n'
} > "${RUN_DIR}/environment.txt"

set +e
printf 'round,candidate_id,selected,effect_loss_before,effect_loss_after,delta_loss,effect_acc_before,effect_acc_after,delta_acc,A,norm,alignment,R,CV,radial_ratio,sign_ratio,feasible\n' > "${RUN_DIR}/proxy_candidates.csv"
printf 'round,constraint_set,alpha_feasible,max_feasible_A,radial_ratio,sign_ratio,limiting_constraint,boundary_left,boundary_right,last_feasible_A,first_infeasible_A\n' > "${RUN_DIR}/sign_bottleneck.csv"
printf 'round,objective_mode,selection_mode,alpha_feasible,delta_loss,delta_acc,A,norm,alignment,R,CV,radial_ratio,sign_ratio,feasible,initial_population_equal\n' > "${RUN_DIR}/same_checkpoint_objectives.csv"
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

(( status == 0 )) || exit "${status}"
printf 'Completed. Results: %s\n' "${RUN_DIR}"
