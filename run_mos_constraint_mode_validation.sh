#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
MODE="${MODE:-all}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${ROOT_DIR}/mos_constraint_mode_validation/${RUN_ID}"

case "${MODE}" in
    strict|soft_select|soft_full) RUNS=("${MODE}") ;;
    all) RUNS=(strict soft_select soft_full) ;;
    *) echo "MODE must be strict, soft_select, soft_full, or all" >&2; exit 2 ;;
esac

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to overwrite existing run: ${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/data"

for name in ${!MOS_EVIDENCE@}; do unset "${name}"; done

write_csv() {
    local log_file="$1"
    local csv_file="$2"
    awk '
        BEGIN { print "round,test_acc,train_loss,alpha_feasible,alpha_init,limiting_constraint,selected_A,selected_R,selected_CV,selected_guidance_alignment,boundary_left,boundary_right" }
        function value(prefix,    i) {
            for (i = 1; i <= NF; i++) if (index($i, prefix) == 1) return substr($i, length(prefix) + 1)
            return ""
        }
        /^\[MOS-Boundary\]/ {
            alpha = value("alpha_feasible="); limiting = value("limiting_constraint=")
            left = value("left="); right = value("right=")
        }
        /^\[MOS-Core\] adaptive_guided_init=True/ { alpha_init = value("alpha_init=") }
        /^\[MOS-Core\] selected_A=/ { selected_a = value("selected_A=") }
        /^\[MOS-Core\] selected_R=/ { selected_r = value("selected_R=") }
        /^\[MOS-Core\] selected_cv=/ { selected_cv = value("selected_cv=") }
        /^\[MOS-Core\] selected_guidance_alignment=/ { alignment = value("selected_guidance_alignment=") }
        /^t[[:space:]]+[0-9]+: train_loss =/ {
            round = $2; sub(":", "", round)
            loss = $5; sub(",", "", loss)
            acc = $8
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\"%s\",\"%s\"\n", round, acc, loss, alpha, alpha_init, limiting, selected_a, selected_r, selected_cv, alignment, left, right
            alpha=alpha_init=limiting=selected_a=selected_r=selected_cv=alignment=left=right=""
        }
    ' "${log_file}" > "${csv_file}"
}

run_one() {
    local constraint_mode="$1"
    local run_dir="${OUTPUT_ROOT}/${constraint_mode}_seed${SEED}_rounds${ROUNDS}"
    local log_file="${run_dir}/train.log"
    mkdir -p "${run_dir}/config"
    cp -R "${ROOT_DIR}/config/." "${run_dir}/config/"
    sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" \
        "${run_dir}/config/attack/cifar/basee.yaml"
    ln -s "${ROOT_DIR}/data" "${run_dir}/data"

    local command="python ${ROOT_DIR}/main.py --dataset cifar --num_attackers 20 --attack mos_attack --defend1 multi_krum --seed ${SEED} --gpu ${GPU} --repeat 1 --mos_adaptive_guided_init 1 --mos_constraint_mode ${constraint_mode} --mos_inject_attack_ray_diagnostics 0"
    printf '%s\n' "${command}" > "${run_dir}/command.txt"
    {
        printf 'GPU=%s\nSEED=%s\nROUNDS=%s\nMODE=%s\nRUN_ID=%s\n' "${GPU}" "${SEED}" "${ROUNDS}" "${constraint_mode}" "${RUN_ID}"
        if command -v git >/dev/null 2>&1 && \
                git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            printf 'GIT_COMMIT=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
            printf 'GIT_STATUS_BEGIN\n'
            git -C "${ROOT_DIR}" status --short
            printf 'GIT_STATUS_END\n'
        else
            printf 'GIT_COMMIT=unavailable\nGIT_STATUS=unavailable\n'
        fi
    } > "${run_dir}/environment.txt"

    printf 'Running %s: GPU=%s SEED=%s ROUNDS=%s\n' "${constraint_mode}" "${GPU}" "${SEED}" "${ROUNDS}" | tee "${log_file}"
    (
        cd "${run_dir}"
        PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python "${ROOT_DIR}/main.py" \
            --dataset cifar --num_attackers 20 --attack mos_attack --defend1 multi_krum \
            --seed "${SEED}" --gpu "${GPU}" --repeat 1 \
            --mos_adaptive_guided_init 1 --mos_constraint_mode "${constraint_mode}" \
            --mos_inject_attack_ray_diagnostics 0
    ) 2>&1 | tee -a "${log_file}"
    write_csv "${log_file}" "${run_dir}/metrics.csv"
}

for run in "${RUNS[@]}"; do run_one "${run}"; done
