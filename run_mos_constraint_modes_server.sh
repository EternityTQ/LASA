#!/usr/bin/env bash
set -euo pipefail

# Server-side MOS constraint/objective ablations. By default this runs the
# current strict-constraint dual vs A-only comparison. Use MODE=all and
# OBJECTIVE_MODE=dual to reproduce the original three constraint-mode runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-2}"
SEED="${SEED:-1}"
ROUNDS="${ROUNDS:-60}"
MODE="${MODE:-strict}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-all}"

case "${MODE}" in
    strict|soft_select|soft_full) MODES=("${MODE}") ;;
    all) MODES=(strict soft_select soft_full) ;;
    *)
        echo "MODE must be strict, soft_select, soft_full, or all" >&2
        exit 2
        ;;
esac

case "${OBJECTIVE_MODE}" in
    dual|a_only) OBJECTIVE_MODES=("${OBJECTIVE_MODE}") ;;
    all) OBJECTIVE_MODES=(dual a_only) ;;
    *)
        echo "OBJECTIVE_MODE must be dual, a_only, or all" >&2
        exit 2
        ;;
esac

if ! [[ "${GPU}" =~ ^-?[0-9]+$ ]]; then
    echo "GPU must be an integer" >&2
    exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROUNDS must be a positive integer" >&2
    exit 2
fi

# Do not allow a shell inherited from an older Evidence Gate experiment to
# affect these runs.
while IFS= read -r evidence_var; do
    [[ -n "${evidence_var}" ]] && unset "${evidence_var}"
done < <(compgen -A variable MOS_EVIDENCE || true)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ROOT_DIR}/server_experiments/mos_constraint_modes/${TIMESTAMP}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to overwrite existing experiment directory: ${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/data"

write_metrics() {
    local constraint_mode="$1"
    local objective_mode="$2"
    local log_file="$3"
    local csv_file="$4"

    awk -v constraint_mode="${constraint_mode}" -v objective_mode="${objective_mode}" '
        BEGIN {
            print "constraint_mode,objective_mode,round,train_loss,test_acc,alpha_feasible,alpha_init,selected_A,selected_R,selected_CV,selected_guidance_alignment,limiting_constraint,boundary_left,boundary_right,selection_mode,selected_feasible,radial_score,radial_ratio,radial_violation,sign_score,sign_ratio,sign_violation"
            reset_round_fields()
        }

        function reset_round_fields() {
            alpha_feasible = ""
            alpha_init = ""
            selected_a = ""
            selected_r = ""
            selected_cv = ""
            selected_alignment = ""
            limiting_constraint = ""
            boundary_left = ""
            boundary_right = ""
            selection_mode = selected_feasible = ""
            radial_score = radial_ratio = radial_violation = ""
            sign_score = sign_ratio = sign_violation = ""
        }

        function token_value(prefix,    i, value) {
            for (i = 1; i <= NF; i++) {
                if (index($i, prefix) == 1) {
                    value = substr($i, length(prefix) + 1)
                    sub(/,$/, "", value)
                    return value
                }
            }
            return ""
        }

        function csv(value,    escaped) {
            escaped = value
            gsub(/"/, "\"\"", escaped)
            if (escaped ~ /[,"]/ || escaped ~ /[\r\n]/) {
                return "\"" escaped "\""
            }
            return escaped
        }

        /^\[MOS-Boundary\]/ {
            alpha_feasible = token_value("alpha_feasible=")
            limiting_constraint = token_value("limiting_constraint=")
            boundary_left = token_value("left=")
            boundary_right = token_value("right=")
            next
        }

        /^\[MOS-Core\] adaptive_guided_init=True/ {
            alpha_feasible = token_value("alpha_feasible=")
            alpha_init = token_value("alpha_init=")
            next
        }

        /^\[MOS-Core\] selected_A=/ {
            selected_a = token_value("selected_A=")
            next
        }
        /^\[MOS-Core\] selected_R=/ {
            selected_r = token_value("selected_R=")
            next
        }
        /^\[MOS-Core\] selected_cv=/ {
            selected_cv = token_value("selected_cv=")
            next
        }
        /^\[MOS-Core\] selected_guidance_alignment=/ {
            selected_alignment = token_value("selected_guidance_alignment=")
            next
        }
        /^\[MOS-Core\] selection_mode=/ { selection_mode = token_value("selection_mode="); next }
        /^\[MOS-Core\] selected_feasible=/ { selected_feasible = token_value("selected_feasible="); next }
        /^\[MOS-Core\]   Radial:/ {
            radial_score = token_value("score="); radial_ratio = token_value("ratio=")
            radial_violation = token_value("violation="); next
        }
        /^\[MOS-Core\]   Sign:/ {
            sign_score = token_value("score="); sign_ratio = token_value("ratio=")
            sign_violation = token_value("violation="); next
        }

        /^t[[:space:]]+[0-9]+: train_loss =/ {
            round = $2
            sub(/:$/, "", round)
            train_loss = ""
            test_acc = ""
            for (i = 1; i <= NF; i++) {
                if ($i == "train_loss" && i + 2 <= NF) {
                    train_loss = $(i + 2)
                    sub(/,$/, "", train_loss)
                }
                if ($i == "test_acc" && i + 2 <= NF) {
                    test_acc = $(i + 2)
                    sub(/,$/, "", test_acc)
                }
            }

            print csv(constraint_mode) "," csv(objective_mode) "," csv(round) "," csv(train_loss) "," csv(test_acc) "," \
                  csv(alpha_feasible) "," csv(alpha_init) "," csv(selected_a) "," \
                  csv(selected_r) "," csv(selected_cv) "," csv(selected_alignment) "," \
                  csv(limiting_constraint) "," csv(boundary_left) "," csv(boundary_right) "," \
                  csv(selection_mode) "," csv(selected_feasible) "," \
                  csv(radial_score) "," csv(radial_ratio) "," csv(radial_violation) "," \
                  csv(sign_score) "," csv(sign_ratio) "," csv(sign_violation)
            reset_round_fields()
        }
    ' "${log_file}" > "${csv_file}"
}

write_summary() {
    local metrics_file="$1"
    local summary_file="$2"

    awk -F, '
        BEGIN {
            OFS = ","
            print "constraint_mode,objective_mode,observed_rounds,last_test_acc,last10_mean_acc,mean_test_acc,min_test_acc,mean_A,last10_mean_A,mean_R,mean_CV,mean_alpha_feasible"
        }
        NR == 1 { next }
        {
            mode = $1 "," $2
            if (!(mode in seen_mode)) {
                seen_mode[mode] = 1
                order[++mode_count] = mode
            }
            observed[mode]++

            if ($5 != "") {
                acc_count[mode]++
                acc[mode, acc_count[mode]] = $5 + 0
                acc_sum[mode] += $5
                last_acc[mode] = $5
                if (!(mode in has_min_acc) || ($5 + 0) < min_acc[mode]) {
                    min_acc[mode] = $5 + 0
                    has_min_acc[mode] = 1
                }
            }
            if ($8 != "") {
                a_count[mode]++
                attack[mode, a_count[mode]] = $8 + 0
                a_sum[mode] += $8
            }
            if ($9 != "") {
                r_count[mode]++
                r_sum[mode] += $9
            }
            if ($10 != "") {
                cv_count[mode]++
                cv_sum[mode] += $10
            }
            if ($6 != "") {
                alpha_count[mode]++
                alpha_sum[mode] += $6
            }
        }

        function mean_or_blank(sum, count) {
            return count ? sprintf("%.6f", sum / count) : ""
        }

        END {
            for (m = 1; m <= mode_count; m++) {
                mode = order[m]

                last10_acc_sum = 0
                last10_acc_count = 0
                start = acc_count[mode] - 9
                if (start < 1) start = 1
                for (i = start; i <= acc_count[mode]; i++) {
                    last10_acc_sum += acc[mode, i]
                    last10_acc_count++
                }

                last10_a_sum = 0
                last10_a_count = 0
                start = a_count[mode] - 9
                if (start < 1) start = 1
                for (i = start; i <= a_count[mode]; i++) {
                    last10_a_sum += attack[mode, i]
                    last10_a_count++
                }

                print mode, observed[mode], \
                      (acc_count[mode] ? last_acc[mode] : ""), \
                      mean_or_blank(last10_acc_sum, last10_acc_count), \
                      mean_or_blank(acc_sum[mode], acc_count[mode]), \
                      (has_min_acc[mode] ? sprintf("%.6f", min_acc[mode]) : ""), \
                      mean_or_blank(a_sum[mode], a_count[mode]), \
                      mean_or_blank(last10_a_sum, last10_a_count), \
                      mean_or_blank(r_sum[mode], r_count[mode]), \
                      mean_or_blank(cv_sum[mode], cv_count[mode]), \
                      mean_or_blank(alpha_sum[mode], alpha_count[mode])
            }
        }
    ' "${metrics_file}" > "${summary_file}"
}

run_one() {
    local constraint_mode="$1"
    local objective_mode="$2"
    local run_name="${constraint_mode}__${objective_mode}"
    local run_dir="${OUTPUT_ROOT}/${run_name}"
    local log_file="${run_dir}/train.log"
    local metrics_file="${run_dir}/metrics.csv"
    local summary_file="${run_dir}/summary.csv"
    local config_file="${run_dir}/config/attack/cifar/basee.yaml"

    mkdir -p "${run_dir}/config"
    cp -R "${ROOT_DIR}/config/." "${run_dir}/config/"
    sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" "${config_file}"
    ln -s "${ROOT_DIR}/data" "${run_dir}/data"

    local -a command=(
        python "${ROOT_DIR}/main.py"
        --dataset cifar
        --num_attackers 20
        --attack mos_attack
        --defend1 multi_krum
        --seed "${SEED}"
        --gpu "${GPU}"
        --repeat 1
        --mos_adaptive_guided_init 1
        --mos_constraint_mode "${constraint_mode}"
        --mos_objective_mode "${objective_mode}"
        --mos_inject_attack_ray_diagnostics 0
    )

    {
        printf 'cd %q\n' "${run_dir}"
        printf 'PYTHONPATH=%q ' "${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${run_dir}/command.txt"

    {
        printf 'GPU=%s\nSEED=%s\nROUNDS=%s\nCONSTRAINT_MODE=%s\nOBJECTIVE_MODE=%s\n' \
            "${GPU}" "${SEED}" "${ROUNDS}" "${constraint_mode}" "${objective_mode}"
        printf 'GIT_COMMIT=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unavailable)"
        printf 'GIT_STATUS_BEGIN\n'
        git -C "${ROOT_DIR}" status --short 2>/dev/null || true
        printf 'GIT_STATUS_END\n'
    } > "${run_dir}/environment.txt"

    printf 'Running constraint_mode=%s objective_mode=%s GPU=%s SEED=%s ROUNDS=%s\n' \
        "${constraint_mode}" "${objective_mode}" "${GPU}" "${SEED}" "${ROUNDS}" | tee "${log_file}"

    local train_status
    set +e
    (
        cd "${run_dir}"
        PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${command[@]}"
    ) 2>&1 | tee -a "${log_file}"
    train_status=${PIPESTATUS[0]}
    set -e

    write_metrics "${constraint_mode}" "${objective_mode}" "${log_file}" "${metrics_file}"
    write_summary "${metrics_file}" "${summary_file}"

    if (( train_status != 0 )); then
        echo "Training failed for mode=${constraint_mode} with status ${train_status}" >&2
        return "${train_status}"
    fi
}

for constraint_mode in "${MODES[@]}"; do
    for objective_mode in "${OBJECTIVE_MODES[@]}"; do
        run_one "${constraint_mode}" "${objective_mode}"
    done
done

COMBINED_METRICS="${OUTPUT_ROOT}/metrics.csv"
first_metrics=1
for constraint_mode in "${MODES[@]}"; do
    for objective_mode in "${OBJECTIVE_MODES[@]}"; do
        mode_metrics="${OUTPUT_ROOT}/${constraint_mode}__${objective_mode}/metrics.csv"
        if (( first_metrics )); then
            cp "${mode_metrics}" "${COMBINED_METRICS}"
            first_metrics=0
        else
            tail -n +2 "${mode_metrics}" >> "${COMBINED_METRICS}"
        fi
    done
done
write_summary "${COMBINED_METRICS}" "${OUTPUT_ROOT}/summary.csv"

printf 'Completed. Results: %s\n' "${OUTPUT_ROOT}"
