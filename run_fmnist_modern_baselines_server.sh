#!/usr/bin/env bash
set -Eeuo pipefail
# Preparation runner: invoked explicitly only after the experiment freeze.
# Reuse the established cell lifecycle, logs, heartbeat and resume machinery.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Refuse a misleading setting label if the checked-in config later changes.
for setting in num_users:100 num_selected_users:25 iid:1 model:cnnfmnist tau:3 batch_size:64 local_lr:0.1 local_momentum:0.9 decay_weight:0.99 global_momentum:0.9 clip:2.0; do
    key="${setting%%:*}"; expected="${setting#*:}"
    actual="$(awk -v key="${key}:" '$1==key{print $2}' "${ROOT_DIR}/config/attack/fmnist/basee.yaml")"
    [[ "${actual}" == "${expected}" ]] || { echo "FMNIST setting mismatch: ${key}=${actual}, expected ${expected}" >&2; exit 2; }
done
export BASELINE_DATASET=fmnist
export BASELINE_PROTOCOL=fmnist_cnn_u100_s25_m20_iid_splitv2_legacyff10
export BASELINE_OUTPUT_GROUP="modern_baselines/${BASELINE_PROTOCOL}"
attacks_were_set="${ATTACKS+x}"
defenses_were_set="${DEFENSES+x}"
export ATTACKS="${ONLY_ATTACK:-${ATTACKS:-non_attack,agrAgnosticMinMax,poisonedfl_attack,mos_attack}}"
export DEFENSES="${ONLY_DEFENSE:-${DEFENSES:-multi_krum,tr_mean,signguard}}"
export SEEDS="${SEEDS:-1}"
export ROUNDS="${ROUNDS:-200}"
if [[ "${SMOKE:-0}" == 1 ]]; then
    export ROUNDS="${SMOKE_ROUNDS:-2}"
    export BASELINE_OUTPUT_GROUP="${BASELINE_OUTPUT_GROUP}/smoke"
    [[ -n "${attacks_were_set}" || -n "${ONLY_ATTACK:-}" ]] || export ATTACKS=poisonedfl_attack
    [[ -n "${defenses_were_set}" || -n "${ONLY_DEFENSE:-}" ]] || export DEFENSES=signguard
fi
[[ "${SMOKE:-0}" == 0 || "${SMOKE:-0}" == 1 ]] || { echo 'SMOKE must be 0 or 1' >&2; exit 2; }
[[ "${SEEDS}" =~ ^[0-9]+$ ]] || { echo 'This preparation runner accepts one seed only' >&2; exit 2; }
IFS=',' read -r -a requested_attacks <<< "${ATTACKS}"
for value in "${requested_attacks[@]}"; do
    case "${value//[[:space:]]/}" in non_attack|agrAgnosticMinMax|poisonedfl_attack|mos_attack);; *) echo "Unsupported attack: ${value}" >&2; exit 2;; esac
done
IFS=',' read -r -a requested_defenses <<< "${DEFENSES}"
for value in "${requested_defenses[@]}"; do
    case "${value//[[:space:]]/}" in multi_krum|tr_mean|signguard);; *) echo "Unsupported defense: ${value}" >&2; exit 2;; esac
done
exec bash "${ROOT_DIR}/run_mos_baselines_server.sh"
