#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-200}"
MODE="${MODE:-both}"

DATASET="cifar"
ATTACK_RATIO="20"
OUTPUT_ROOT="${ROOT_DIR}/adaptive_guided_init_ablation"

case "${MODE}" in
    baseline) RUNS=("baseline:0") ;;
    adaptive) RUNS=("adaptive:1") ;;
    both) RUNS=("baseline:0" "adaptive:1") ;;
    *) echo "MODE must be baseline, adaptive, or both" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/data"

run_one() {
    local label="$1"
    local adaptive_flag="$2"
    local run_dir="${OUTPUT_ROOT}/${label}"
    local log_file="${run_dir}/train.log"

    mkdir -p "${run_dir}/config"
    cp -R "${ROOT_DIR}/config/." "${run_dir}/config/"
    sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" \
        "${run_dir}/config/attack/${DATASET}/basee.yaml"
    if [[ ! -e "${run_dir}/data" ]]; then
        ln -s "${ROOT_DIR}/data" "${run_dir}/data"
    fi

    echo "Running ${label}: GPU=${GPU} SEED=${SEED} ROUNDS=${ROUNDS}" | tee "${log_file}"
    (
        cd "${run_dir}"
        PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python "${ROOT_DIR}/main.py" \
            --dataset "${DATASET}" \
            --num_attackers "${ATTACK_RATIO}" \
            --attack mos_attack \
            --defend1 multi_krum \
            --seed "${SEED}" \
            --gpu "${GPU}" \
            --repeat 1 \
            --mos_adaptive_guided_init "${adaptive_flag}" \
            --mos_inject_attack_ray_diagnostics 0
    ) 2>&1 | tee -a "${log_file}"
}

for run in "${RUNS[@]}"; do
    run_one "${run%%:*}" "${run##*:}"
done
