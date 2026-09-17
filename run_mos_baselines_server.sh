#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable CIFAR attack/defense baseline matrix. Does not daemonize itself.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${BASELINE_DATASET:-cifar}"
PROTOCOL="${BASELINE_PROTOCOL:-legacy_cifar}"
POISONEDFL_SCALE_FACTOR="${POISONEDFL_SCALE_FACTOR:-8}"
POISONEDFL_FEEDBACK_INTERVAL="${POISONEDFL_FEEDBACK_INTERVAL:-50}"
GPU="${GPU:-2}"
ROUNDS="${ROUNDS:-200}"
ATTACKS="${ATTACKS:-non_attack,signflip_attack,lie_attack,agrAgnosticMinMax,mos_attack}"
DEFENSES="${DEFENSES:-fedavg,multi_krum,tr_mean,signguard,dnc,lasa}"
SEEDS="${SEEDS:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
RESUME_DIR="${RESUME_DIR:-}"
PYTHON_BIN="${MOS_BASELINE_PYTHON_BIN:-python}"       # smoke-test hook
HEARTBEAT_SECONDS="${MOS_BASELINE_HEARTBEAT_SECONDS:-60}"
MOS_OBJECTIVE_MODE="${MOS_OBJECTIVE_MODE:-dual}"
MOS_BOUNDARY_ONLY="${MOS_BOUNDARY_ONLY:-0}"
MOS_ADAPTIVE_GUIDED_INIT="${MOS_ADAPTIVE_GUIDED_INIT:-1}"
MOS_USE_RADIAL_CONSTRAINT="${MOS_USE_RADIAL_CONSTRAINT:-1}"
MOS_RUN_VARIANT="${MOS_RUN_VARIANT:-}"

VALID_ATTACKS=(agrTailoredTrmean agrAgnosticMinMax agrAgnosticMinSum signflip_attack noise_attack random_attack lie_attack byzmean_attack non_attack mos_attack skew_attack poisonedfl_attack)
VALID_DEFENSES=(fedavg signguard dnc lasa bulyan tr_mean multi_krum sparsefed geomed rlr lfd)
CURRENT_ATTACK="" CURRENT_DEFENSE="" CURRENT_SEED="" CURRENT_CELL_DIR="" CURRENT_ATTEMPT_DIR=""
CURRENT_START_EPOCH="" CURRENT_START_TIME="" CURRENT_CHILD_PID="" CURRENT_HEARTBEAT_PID="" CURRENT_TEE_PID=""
CURRENT_SIGNAL="" CURRENT_FINALIZED=1 RUN_ONE_FAILED=0 LAST_SHELL_ERROR=""
GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unavailable)"
HOST_NAME="$(hostname 2>/dev/null || printf unavailable)"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
contains() { local wanted="$1" item; shift; for item in "$@"; do [[ "${item}" == "${wanted}" ]] && return 0; done; return 1; }
[[ "${DATASET}" == cifar || "${DATASET}" == fmnist ]] || die "unsupported dataset"
[[ "${PROTOCOL}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid protocol label"
[[ "${POISONEDFL_SCALE_FACTOR}" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v x="${POISONEDFL_SCALE_FACTOR}" 'BEGIN{exit !(x>0)}' || die "PoisonedFL scale must be positive"
[[ "${POISONEDFL_FEEDBACK_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "PoisonedFL interval must be positive"
[[ "${GPU}" =~ ^-?[0-9]+$ ]] || die "GPU must be an integer"
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || die "ROUNDS must be positive"
[[ "${STOP_ON_ERROR}" == 0 || "${STOP_ON_ERROR}" == 1 ]] || die "STOP_ON_ERROR must be 0 or 1"
[[ "${HEARTBEAT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || die "heartbeat interval must be positive"
[[ "${MOS_OBJECTIVE_MODE}" == dual || "${MOS_OBJECTIVE_MODE}" == a_only ]] || die "MOS_OBJECTIVE_MODE must be dual or a_only"
[[ "${MOS_BOUNDARY_ONLY}" == 0 || "${MOS_BOUNDARY_ONLY}" == 1 ]] || die "MOS_BOUNDARY_ONLY must be 0 or 1"
[[ "${MOS_ADAPTIVE_GUIDED_INIT}" == 0 || "${MOS_ADAPTIVE_GUIDED_INIT}" == 1 ]] || die "MOS_ADAPTIVE_GUIDED_INIT must be 0 or 1"
[[ "${MOS_USE_RADIAL_CONSTRAINT}" == 0 || "${MOS_USE_RADIAL_CONSTRAINT}" == 1 ]] || die "MOS_USE_RADIAL_CONSTRAINT must be 0 or 1"

parse_list() {
    local source="$1" kind="$2" valid_name="$3" output_name="$4" raw value
    local -n valid_ref="${valid_name}" output_ref="${output_name}"
    local -A seen=()
    IFS=',' read -r -a raw_values <<< "${source}"; output_ref=()
    for raw in "${raw_values[@]}"; do
        value="${raw//[[:space:]]/}"; [[ -n "${value}" ]] || die "${kind} list contains an empty entry"
        contains "${value}" "${valid_ref[@]}" || die "unsupported ${kind} '${value}' (not in current main.py choices)"
        if [[ -z "${seen[${value}]:-}" ]]; then output_ref+=("${value}"); seen["${value}"]=1; fi
    done
    (( ${#output_ref[@]} )) || die "${kind} list must not be empty"
}
declare -a RUN_ATTACKS RUN_DEFENSES RUN_SEEDS
parse_list "${ATTACKS}" attack VALID_ATTACKS RUN_ATTACKS
parse_list "${DEFENSES}" defense VALID_DEFENSES RUN_DEFENSES
IFS=',' read -r -a raw_seeds <<< "${SEEDS}"; declare -A seen_seeds=()
for raw_seed in "${raw_seeds[@]}"; do
    seed="${raw_seed//[[:space:]]/}"; [[ "${seed}" =~ ^[0-9]+$ ]] || die "invalid seed '${seed}'"
    if [[ -z "${seen_seeds[${seed}]:-}" ]]; then RUN_SEEDS+=("${seed}"); seen_seeds["${seed}"]=1; fi
done
(( ${#RUN_SEEDS[@]} )) || die "SEEDS must not be empty"

if [[ -n "${RESUME_DIR}" ]]; then
    [[ -d "${RESUME_DIR}" ]] || die "RESUME_DIR does not exist: ${RESUME_DIR}"
    OUTPUT_ROOT="$(cd "${RESUME_DIR}" && pwd)"
else
    OUTPUT_ROOT="${ROOT_DIR}/server_experiments/${BASELINE_OUTPUT_GROUP:-mos_baseline_matrix}/$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite: ${OUTPUT_ROOT}"; mkdir -p "${OUTPUT_ROOT}"
fi
mkdir -p "${ROOT_DIR}/data"
# The FMNIST wrapper writes a frozen protocol manifest before any cell starts.
# Resume is at cell level: failed attempts restart at round zero with fresh
# attack state. No claim of model/optimizer checkpoint continuation is made.
if [[ "${DATASET}" == fmnist ]]; then
    manifest="$(printf 'dataset=%s\nprotocol=%s\nrounds=%s\npoisonedfl_scale_factor=%s\npoisonedfl_feedback_interval=%s\nmos=%s,%s,%s,%s\n' "${DATASET}" "${PROTOCOL}" "${ROUNDS}" "${POISONEDFL_SCALE_FACTOR}" "${POISONEDFL_FEEDBACK_INTERVAL}" "${MOS_OBJECTIVE_MODE}" "${MOS_ADAPTIVE_GUIDED_INIT}" "${MOS_BOUNDARY_ONLY}" "${MOS_USE_RADIAL_CONSTRAINT}"; cd "${ROOT_DIR}"; find config algorithms model utils -type f \( -name '*.yaml' -o -name '*.py' \) -print0 | sort -z | xargs -0 sha256sum; sha256sum main.py test.py run_mos_baselines_server.sh run_fmnist_modern_baselines_server.sh)"
    if [[ -f "${OUTPUT_ROOT}/protocol.txt" ]]; then
        [[ "$(cat "${OUTPUT_ROOT}/protocol.txt")" == "${manifest}" ]] || die "resume protocol/config mismatch; use a new output directory"
    else
        [[ -z "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "unlabeled resume directory is not empty"
        printf '%s\n' "${manifest}" > "${OUTPUT_ROOT}/protocol.txt"
    fi
fi

last_round_from_log() {
    [[ -f "$1" ]] || { printf ''; return; }
    awk '/^t[[:space:]]+[0-9]+: train_loss =/{r=$2;sub(/:$/,"",r)}END{if(r!="")print r}' "$1"
}

write_metrics() {
    local attack="$1" defense="$2" seed="$3" log="$4" output="$5"
    awk -v attack="${attack}" -v defense="${defense}" -v seed="${seed}" '
      BEGIN{OFS=",";print "attack,defense,seed,round,train_loss,test_acc,alpha_feasible,alpha_init,selected_A,selected_R,selected_CV,selected_guidance_alignment,limiting_constraint,boundary_left,boundary_right,selection_mode,selected_feasible,radial_score,radial_ratio,radial_violation,sign_score,sign_ratio,sign_violation";reset()}
      function reset(){af=ai=a=r=cv=align=limit=left=right=sm=sf=rs=rr=rv=ss=sr=sv=""}
      function tok(p,i,v){for(i=1;i<=NF;i++)if(index($i,p)==1){v=substr($i,length(p)+1);sub(/,$/,"",v);return v}return ""}
      function csv(v,e){e=v;gsub(/"/,"\"\"",e);return(e~/[,"\r\n]/)?"\""e"\"":e}
      /^\[MOS-Boundary\]/{af=tok("alpha_feasible=");limit=tok("limiting_constraint=");left=tok("left=");right=tok("right=");next}
      /^\[MOS-Core\] adaptive_guided_init=True/{af=tok("alpha_feasible=");ai=tok("alpha_init=");next}
      /^\[MOS-Core\] selected_A=/{a=tok("selected_A=");next} /^\[MOS-Core\] selected_R=/{r=tok("selected_R=");next}
      /^\[MOS-Core\] selected_cv=/{cv=tok("selected_cv=");next} /^\[MOS-Core\] selected_guidance_alignment=/{align=tok("selected_guidance_alignment=");next}
      /^\[MOS-Core\] selection_mode=/{sm=tok("selection_mode=");next} /^\[MOS-Core\] selected_feasible=/{sf=tok("selected_feasible=");next}
      /^\[MOS-Core\]   Radial:/{rs=tok("score=");rr=tok("ratio=");rv=tok("violation=");next}
      /^\[MOS-Core\]   Sign:/{ss=tok("score=");sr=tok("ratio=");sv=tok("violation=");next}
      /^t[[:space:]]+[0-9]+: train_loss =/{round=$2;sub(/:$/, "",round);loss=acc="";for(i=1;i<=NF;i++){if($i=="train_loss"){loss=$(i+2);sub(/,$/,"",loss)}if($i=="test_acc"){acc=$(i+2);sub(/,$/,"",acc)}};print csv(attack),csv(defense),csv(seed),csv(round),csv(loss),csv(acc),csv(af),csv(ai),csv(a),csv(r),csv(cv),csv(align),csv(limit),csv(left),csv(right),csv(sm),csv(sf),csv(rs),csv(rr),csv(rv),csv(ss),csv(sr),csv(sv);reset()}
    ' "${log}" > "${output}"
}

write_summary() {
    awk -F, -v target="$3" '
      BEGIN{OFS=",";print "attack,defense,seed,target_rounds,observed_rounds,final_acc,last10_mean_acc,mean_acc,min_acc,mean_A,last10_mean_A,mean_R,mean_CV,mean_alpha_feasible"}
      NR==1{next}{attack=$1;defense=$2;seed=$3;observed++;if($6!=""){n++;acc[n]=$6+0;sum+=$6;last=$6;if(!hm||$6+0<min){min=$6+0;hm=1}}if($9!=""){an++;av[an]=$9+0;asum+=$9}if($10!=""){rn++;rsum+=$10}if($11!=""){cn++;csum+=$11}if($7!=""){fn++;fsum+=$7}}
      function mean(s,n){return n?sprintf("%.6f",s/n):""}
      END{x10=xn=a10=aan=0;st=n-9;if(st<1)st=1;for(i=st;i<=n;i++){x10+=acc[i];xn++};st=an-9;if(st<1)st=1;for(i=st;i<=an;i++){a10+=av[i];aan++};print attack,defense,seed,target,observed,(n?last:""),mean(x10,xn),mean(sum,n),(hm?sprintf("%.6f",min):""),mean(asum,an),mean(a10,aan),mean(rsum,rn),mean(csum,cn),mean(fsum,fn)}
    ' "$1" > "$2"
}

error_hint_from_log() {
    local code="$1" log="$2" hint="" matched
    case "${code}" in 137)hint="exit 137: possibly SIGKILL or system OOM";;143)hint="exit 143: possibly SIGTERM";;129)hint="exit 129: possibly SIGHUP";;0);;*)hint="Python exited with code ${code}";;esac
    matched="$(awk 'tolower($0)~/traceback|runtimeerror|cuda out of memory|killed|non-finite|rollback=true|segmentation fault/{hit=$0}/first_bad_stage=/&&$0!~/first_bad_stage=none/{hit=$0}END{if(hit!="")print hit}' "${log}" 2>/dev/null || true)"
    [[ -z "${matched}" ]] || hint="${hint:+${hint}; }log: ${matched}"; printf '%s' "${hint}"
}

write_failure_diagnostics() {
    local output="$1" code="$2" log="$3" temp="${1}.dmesg.tmp"
    {
      printf 'timestamp=%s\npython_exit_code=%s\n\n===== last 100 lines of train.log =====\n' "$(timestamp)" "${code}";tail -n 100 "${log}" 2>&1||true
      printf '\n===== nvidia-smi =====\n';nvidia-smi 2>&1||printf 'nvidia-smi unavailable\n'
      printf '\n===== free -h =====\n';free -h 2>&1||printf 'free unavailable\n'
      printf '\n===== df -h =====\n';df -h 2>&1||printf 'df unavailable\n'
      printf '\n===== recent kernel OOM messages =====\n';if dmesg -T > "${temp}" 2>/dev/null;then grep -Ei 'Out of memory|Killed process' "${temp}"|tail -n 50||printf 'no recent OOM messages found\n';else printf 'dmesg unavailable\n';fi;rm -f "${temp}"
    } > "${output}" 2>&1 || true
}

sync_cell_artifacts() {
    local item; for item in train.log command.txt environment.txt status.txt metrics.csv summary.csv heartbeat.log failure_diagnostics.txt; do
      if [[ -f "$1/${item}" ]];then cp "$1/${item}" "$2/${item}";else rm -f "$2/${item}";fi
    done
}

write_matrix() {
    local input="$1" output="$2" kind="$3" attacks defenses
    attacks="$(IFS='|';printf '%s' "${RUN_ATTACKS[*]}")"; defenses="$(IFS='|';printf '%s' "${RUN_DEFENSES[*]}")"
    awk -F, -v attacks="${attacks}" -v defenses="${defenses}" -v kind="${kind}" '
      BEGIN{OFS=",";na=split(attacks,A,"\\|");nd=split(defenses,D,"\\|")}
      NR>1&&$6==1{k=$1 SUBSEP $2;n[k]++;sum[k]+=$9;sq[k]+=$9*$9;final[k]+=$8}
      END{printf "attack";for(j=1;j<=nd;j++)printf ",%s",D[j];print "";for(i=1;i<=na;i++){printf "%s",A[i];for(j=1;j<=nd;j++){k=A[i] SUBSEP D[j];v="";if(n[k]){m=sum[k]/n[k];if(kind=="mean")v=sprintf("%.6f",m);else if(kind=="final")v=sprintf("%.6f",final[k]/n[k]);else v=sprintf("%.6f",sqrt((sq[k]/n[k])-m*m))}printf ",%s",v}print ""}}
    ' "${input}" > "${output}"
}

write_drop_matrix() {
    local input="$1" output="$2" kind="$3" attacks defenses
    attacks="$(IFS='|';printf '%s' "${RUN_ATTACKS[*]}")"; defenses="$(IFS='|';printf '%s' "${RUN_DEFENSES[*]}")"
    awk -F, -v attacks="${attacks}" -v defenses="${defenses}" -v kind="${kind}" '
      BEGIN{OFS=",";na=split(attacks,A,"\\|");nd=split(defenses,D,"\\|")}
      NR>1&&$6==1{val[$1,$2,$3]=$9+0;has[$1,$2,$3]=1;seeds[$3]=1}
      END{printf "attack";for(j=1;j<=nd;j++)printf ",%s",D[j];print "";for(i=1;i<=na;i++){printf "%s",A[i];for(j=1;j<=nd;j++){n=sum=sq=0;for(s in seeds)if(has["non_attack",D[j],s]&&has[A[i],D[j],s]){d=val["non_attack",D[j],s]-val[A[i],D[j],s];n++;sum+=d;sq+=d*d}v="";if(n){m=sum/n;if(kind=="mean")v=sprintf("%.6f",m);else v=sprintf("%.6f",sqrt((sq/n)-m*m))}printf ",%s",v}print ""}}
    ' "${input}" > "${output}"
}

write_matrices() {
    write_matrix "$1" "${OUTPUT_ROOT}/matrix_last10_acc.csv" mean
    write_matrix "$1" "${OUTPUT_ROOT}/matrix_final_acc.csv" final
    write_drop_matrix "$1" "${OUTPUT_ROOT}/matrix_acc_drop_vs_clean.csv" mean
    if (( ${#RUN_SEEDS[@]} > 1 )); then
      write_matrix "$1" "${OUTPUT_ROOT}/matrix_last10_mean.csv" mean
      write_matrix "$1" "${OUTPUT_ROOT}/matrix_last10_std.csv" std
      write_drop_matrix "$1" "${OUTPUT_ROOT}/matrix_drop_mean.csv" mean
      write_drop_matrix "$1" "${OUTPUT_ROOT}/matrix_drop_std.csv" std
      paste -d, "${OUTPUT_ROOT}/matrix_last10_mean.csv" "${OUTPUT_ROOT}/matrix_last10_std.csv" |
        awk -F, -v n="${#RUN_DEFENSES[@]}" 'BEGIN{OFS=","}NR==1{printf "attack";for(i=2;i<=n+1;i++)printf ",%s",$i;print "";next}{printf "%s",$1;for(i=2;i<=n+1;i++)printf ",%s +/- %s",$i,$(i+n+1);print ""}' > "${OUTPUT_ROOT}/matrix_paper_preview.csv"
    fi
}

rebuild_results() {
    local results="${OUTPUT_ROOT}/results_long.csv" cell status summary attack defense seed
    local -a recorded_cells=()
    printf 'attack,defense,seed,target_rounds,observed_rounds,completed,exit_code,final_acc,last10_mean_acc,mean_acc,min_acc,runtime_seconds,error_hint,mean_A,last10_mean_A,mean_R,mean_CV,mean_alpha_feasible,attempt\n' > "${results}"
    if [[ "${DATASET}" == fmnist ]]; then
        # Preserve already completed cells when resuming only one matrix entry.
        mapfile -t recorded_cells < <(find "${OUTPUT_ROOT}" -mindepth 3 -maxdepth 3 -type d -name 'seed_*' | sort)
    else
        for defense in "${RUN_DEFENSES[@]}";do for attack in "${RUN_ATTACKS[@]}";do for seed in "${RUN_SEEDS[@]}";do
            recorded_cells+=("${OUTPUT_ROOT}/${attack}/${defense}/seed_${seed}")
        done;done;done
    fi
    for cell in "${recorded_cells[@]}";do
      status="${cell}/status.txt";summary="${cell}/summary.csv"
      [[ -f "${status}" && -f "${summary}" ]]||continue
      awk -F, -v sf="${status}" '
        BEGIN{while((getline line<sf)>0){p=index(line,"=");if(p){k=substr(line,1,p-1);s[k]=substr(line,p+1)}}close(sf)}
        NR==2{printf "%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,\"%s\",%s,%s,%s,%s,%s,%s\n",$1,$2,$3,$4,$5,(s["status"]=="COMPLETED"),s["exit_code"],$6,$7,$8,$9,s["elapsed_seconds"],q(s["error_hint"]),$10,$11,$12,$13,$14,s["attempt"]}function q(v){gsub(/"/,"\"\"",v);return v}
      ' "${summary}" >> "${results}"
    done
    write_matrices "${results}"
    if [[ "${DATASET}" == fmnist ]]; then
        # Keep historical metric column positions intact for shared matrix code.
        awk -F, -v d="${DATASET}" -v p="${PROTOCOL}" -v sf="${POISONEDFL_SCALE_FACTOR}" -v e="${POISONEDFL_FEEDBACK_INTERVAL}" 'NR==1{print $0 ",dataset,protocol,model,num_users,num_selected_users,malicious_ratio,iid,defense_budget,poisonedfl_scale_factor,poisonedfl_feedback_interval";next}{budget=($2=="signguard"?"not_applicable":"legacy_default_10"); print $0 "," d "," p ",CNNFmnist,100,25,0.20,1," budget "," ($1=="poisonedfl_attack"?sf:"") "," ($1=="poisonedfl_attack"?e:"")}' "${results}" > "${OUTPUT_ROOT}/summary.csv"
    fi
}

stop_heartbeat() {
    if [[ -n "${CURRENT_HEARTBEAT_PID}" ]]&&kill -0 "${CURRENT_HEARTBEAT_PID}" 2>/dev/null;then kill "${CURRENT_HEARTBEAT_PID}" 2>/dev/null||true;wait "${CURRENT_HEARTBEAT_PID}" 2>/dev/null||true;fi
    CURRENT_HEARTBEAT_PID=""
}

finalize_current() {
    local code="${1:-1}" end_epoch end_time elapsed last_round observed status hint attempt
    (( CURRENT_FINALIZED==0 ))||return 0;CURRENT_FINALIZED=1;stop_heartbeat
    if [[ -n "${CURRENT_TEE_PID}" ]];then wait "${CURRENT_TEE_PID}" 2>/dev/null||true;fi;CURRENT_TEE_PID=""
    end_epoch="$(date +%s)";end_time="$(timestamp)";elapsed=$((end_epoch-CURRENT_START_EPOCH));last_round="$(last_round_from_log "${CURRENT_ATTEMPT_DIR}/train.log")"
    write_metrics "${CURRENT_ATTACK}" "${CURRENT_DEFENSE}" "${CURRENT_SEED}" "${CURRENT_ATTEMPT_DIR}/train.log" "${CURRENT_ATTEMPT_DIR}/metrics.csv"
    observed="$(awk 'END{if(NR>0)print NR-1;else print 0}' "${CURRENT_ATTEMPT_DIR}/metrics.csv")";status=FAILED
    if ((code==0&&observed>=ROUNDS));then status=COMPLETED;fi
    hint="$(error_hint_from_log "${code}" "${CURRENT_ATTEMPT_DIR}/train.log")";if ((code==0&&observed<ROUNDS));then hint="process exited 0 but observed ${observed}/${ROUNDS} target rounds";fi
    [[ -z "${LAST_SHELL_ERROR}" ]] || hint="${hint:+${hint}; }shell: ${LAST_SHELL_ERROR}"
    attempt="$(basename "${CURRENT_ATTEMPT_DIR}")";attempt="${attempt#attempt_}"
    write_summary "${CURRENT_ATTEMPT_DIR}/metrics.csv" "${CURRENT_ATTEMPT_DIR}/summary.csv" "${ROUNDS}"
    if [[ "${DATASET}" == fmnist ]]; then
        awk -v d="${DATASET}" -v p="${PROTOCOL}" 'NR==1{print $0 ",dataset,protocol";next}{print $0 "," d "," p}' "${CURRENT_ATTEMPT_DIR}/summary.csv" > "${CURRENT_ATTEMPT_DIR}/summary.tmp"
        mv "${CURRENT_ATTEMPT_DIR}/summary.tmp" "${CURRENT_ATTEMPT_DIR}/summary.csv"
    fi
    printf 'status=%s\nattack=%s\ndefense=%s\nseed=%s\ntarget_rounds=%s\nobserved_rounds=%s\nstart_time=%s\nend_time=%s\nelapsed_seconds=%s\nshell_pid=%s\npython_pid=%s\nhostname=%s\nGPU=%s\nexit_code=%s\nsignal=%s\nlast_round=%s\nerror_hint=%s\nshell_error=%s\nattempt=%s\ngit_commit=%s\n' "${status}" "${CURRENT_ATTACK}" "${CURRENT_DEFENSE}" "${CURRENT_SEED}" "${ROUNDS}" "${observed}" "${CURRENT_START_TIME}" "${end_time}" "${elapsed}" "$$" "${CURRENT_CHILD_PID}" "${HOST_NAME}" "${GPU}" "${code}" "${CURRENT_SIGNAL}" "${last_round}" "${hint}" "${LAST_SHELL_ERROR}" "${attempt}" "${GIT_COMMIT}" > "${CURRENT_ATTEMPT_DIR}/status.txt"
    [[ "${status}" == COMPLETED ]]||write_failure_diagnostics "${CURRENT_ATTEMPT_DIR}/failure_diagnostics.txt" "${code}" "${CURRENT_ATTEMPT_DIR}/train.log"
    sync_cell_artifacts "${CURRENT_ATTEMPT_DIR}" "${CURRENT_CELL_DIR}";rebuild_results;CURRENT_CHILD_PID=""
}

on_err(){ local code=$?;LAST_SHELL_ERROR="line=${BASH_LINENO[0]:-unknown} command=${BASH_COMMAND} exit=${code}"; }
on_signal(){ local signal="$1" code="$2";CURRENT_SIGNAL="${signal}";[[ -z "${CURRENT_CHILD_PID}" ]]||kill -"${signal}" "${CURRENT_CHILD_PID}" 2>/dev/null||true;exit "${code}"; }
on_exit(){ local code=$?;trap - EXIT;stop_heartbeat;if ((CURRENT_FINALIZED==0));then local final_code="${code}";((final_code!=0))||final_code=1;finalize_current "${final_code}";fi;if ((code!=0));then printf 'Launcher exiting with code %s%s\n' "${code}" "${LAST_SHELL_ERROR:+ (${LAST_SHELL_ERROR})}" >&2;fi;exit "${code}"; }
trap on_err ERR
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

next_attempt_dir(){ local n=1;while [[ -e "$1/attempt_${n}" ]];do ((n+=1));done;printf '%s/attempt_%s' "$1" "${n}"; }

completed_config_matches() {
    local cell="$1" attack="$2" command="${1}/command.txt" environment="${1}/environment.txt"
    if [[ "${attack}" == poisonedfl_attack ]]; then
        [[ -f "${environment}" ]] \
          && grep -Fqx "POISONEDFL_SCALE_FACTOR=${POISONEDFL_SCALE_FACTOR}" "${environment}" \
          && grep -Fqx "POISONEDFL_FEEDBACK_INTERVAL=${POISONEDFL_FEEDBACK_INTERVAL}" "${environment}"
        return $?
    fi
    [[ "${attack}" == mos_attack && -n "${MOS_RUN_VARIANT}" ]] || return 0
    [[ -f "${command}" && -f "${environment}" ]] \
      && grep -Fqx "MOS_RUN_VARIANT=${MOS_RUN_VARIANT}" "${environment}" \
      && grep -Eq -- '--mos_objective_mode[[:space:]]+'"${MOS_OBJECTIVE_MODE}"'([[:space:]]|$)' "${command}" \
      && grep -Eq -- '--mos_boundary_only[[:space:]]+'"${MOS_BOUNDARY_ONLY}"'([[:space:]]|$)' "${command}" \
      && grep -Eq -- '--mos_adaptive_guided_init[[:space:]]+'"${MOS_ADAPTIVE_GUIDED_INIT}"'([[:space:]]|$)' "${command}" \
      && grep -Eq -- '--mos_use_radial_constraint[[:space:]]+'"${MOS_USE_RADIAL_CONSTRAINT}"'([[:space:]]|$)' "${command}"
}

run_one() {
    local attack="$1" defense="$2" seed="$3" cell="${OUTPUT_ROOT}/$1/$2/seed_$3" attempt config_file fifo train_status=0 final_status
    local defense_budget=not_applicable
    [[ "${defense}" != multi_krum && "${defense}" != tr_mean ]] || defense_budget=legacy_default_10
    RUN_ONE_FAILED=0
    if [[ -f "${cell}/status.txt" ]] \
       && grep -qx 'status=COMPLETED' "${cell}/status.txt" \
       && grep -qx "target_rounds=${ROUNDS}" "${cell}/status.txt" \
       && awk -F= -v rounds="${ROUNDS}" '$1=="observed_rounds" && $2+0>=rounds{ok=1}END{exit !ok}' "${cell}/status.txt" \
       && completed_config_matches "${cell}" "${attack}"; then
        printf 'Skipping completed cell: attack=%s defense=%s seed=%s\n' "${attack}" "${defense}" "${seed}";return 0
    fi
    mkdir -p "${cell}";attempt="$(next_attempt_dir "${cell}")";mkdir -p "${attempt}/config";cp -R "${ROOT_DIR}/config/." "${attempt}/config/"
    config_file="${attempt}/config/attack/${DATASET}/basee.yaml";sed -i -E "s/^round:[[:space:]]*[0-9]+.*/round: ${ROUNDS} # rounds of training/" "${config_file}";ln -s "${ROOT_DIR}/data" "${attempt}/data"
    local -a command=("${PYTHON_BIN}" -u "${ROOT_DIR}/main.py" --dataset "${DATASET}" --num_attackers 20 --attack "${attack}" --defend1 "${defense}" --seed "${seed}" --gpu "${GPU}" --repeat 1)
    if [[ "${DATASET}" == fmnist ]]; then command+=(--freeze_datasplit 1); fi
    if [[ "${attack}" == poisonedfl_attack ]]; then command+=(--poisonedfl_scale_factor "${POISONEDFL_SCALE_FACTOR}" --poisonedfl_feedback_interval "${POISONEDFL_FEEDBACK_INTERVAL}"); fi
    if [[ "${attack}" == mos_attack ]];then command+=(--mos_adaptive_guided_init "${MOS_ADAPTIVE_GUIDED_INIT}" --mos_constraint_mode strict --mos_objective_mode "${MOS_OBJECTIVE_MODE}" --mos_boundary_only "${MOS_BOUNDARY_ONLY}" --mos_use_radial_constraint "${MOS_USE_RADIAL_CONSTRAINT}");fi
    { printf 'cd %q\n' "${attempt}";printf 'PYTHONPATH=%q ' "${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}";printf '%q ' "${command[@]}";printf '\n';} > "${attempt}/command.txt"
    {
        printf 'ATTACK=%s\nDEFENSE=%s\nSEED=%s\nGPU=%s\nROUNDS=%s\nDATASET=%s\nNUM_ATTACKERS=20\nREPEAT=1\nHOSTNAME=%s\nGIT_COMMIT=%s\n' "${attack}" "${defense}" "${seed}" "${GPU}" "${ROUNDS}" "${DATASET}" "${HOST_NAME}" "${GIT_COMMIT}"
        printf 'PROTOCOL=%s\nDEFENSE_BUDGET=%s\nPOISONEDFL_SCALE_FACTOR=%s\nPOISONEDFL_FEEDBACK_INTERVAL=%s\nPOISONEDFL_SCALE_DECAY=0.7\nPOISONEDFL_MIN_SCALE=0.5\nPOISONEDFL_K99=normal_approximation\nPOISONEDFL_WARMUP=zero_trainable_first_round\nPOISONEDFL_SOURCE_COMMIT=266488e2cbe5953aab61712f315518546f457e55\nPYTHON_BIN=%s\n' "${PROTOCOL}" "${defense_budget}" "${POISONEDFL_SCALE_FACTOR}" "${POISONEDFL_FEEDBACK_INTERVAL}" "${PYTHON_BIN}"
        if [[ "${DATASET}" == fmnist ]]; then cat "${OUTPUT_ROOT}/protocol.txt"; fi
        printf 'MOS_RUN_VARIANT=%s\nMOS_ADAPTIVE_GUIDED_INIT=%s\nMOS_CONSTRAINT_MODE=%s\nMOS_OBJECTIVE_MODE=%s\nMOS_BOUNDARY_ONLY=%s\nMOS_USE_RADIAL_CONSTRAINT=%s\n' "${MOS_RUN_VARIANT}" "$([[ "${attack}" == mos_attack ]]&&printf "${MOS_ADAPTIVE_GUIDED_INIT}"||printf '')" "$([[ "${attack}" == mos_attack ]]&&printf strict||printf '')" "$([[ "${attack}" == mos_attack ]]&&printf "${MOS_OBJECTIVE_MODE}"||printf '')" "$([[ "${attack}" == mos_attack ]]&&printf "${MOS_BOUNDARY_ONLY}"||printf '')" "$([[ "${attack}" == mos_attack ]]&&printf "${MOS_USE_RADIAL_CONSTRAINT}"||printf '')"
        printf 'GIT_STATUS_BEGIN\n';git -C "${ROOT_DIR}" status --short 2>/dev/null||true;printf 'GIT_STATUS_END\n'
    } > "${attempt}/environment.txt"
    CURRENT_ATTACK="${attack}";CURRENT_DEFENSE="${defense}";CURRENT_SEED="${seed}";CURRENT_CELL_DIR="${cell}";CURRENT_ATTEMPT_DIR="${attempt}";CURRENT_START_EPOCH="$(date +%s)";CURRENT_START_TIME="$(timestamp)";CURRENT_SIGNAL="";CURRENT_FINALIZED=0;LAST_SHELL_ERROR=""
    printf 'status=RUNNING\nattack=%s\ndefense=%s\nseed=%s\nstart_time=%s\nshell_pid=%s\npython_pid=\nhostname=%s\nGPU=%s\n' "${attack}" "${defense}" "${seed}" "${CURRENT_START_TIME}" "$$" "${HOST_NAME}" "${GPU}" > "${attempt}/status.txt"
    : > "${attempt}/train.log";: > "${attempt}/heartbeat.log";sync_cell_artifacts "${attempt}" "${cell}"
    fifo="${attempt}/.train.pipe";mkfifo "${fifo}";tee -a "${attempt}/train.log" < "${fifo}" & CURRENT_TEE_PID=$!
    (cd "${attempt}";exec env PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${command[@]}") > "${fifo}" 2>&1 & CURRENT_CHILD_PID=$!
    printf 'python_pid=%s\n' "${CURRENT_CHILD_PID}" >> "${attempt}/status.txt"
    (while kill -0 "${CURRENT_CHILD_PID}" 2>/dev/null;do printf 'timestamp=%s attack=%s defense=%s seed=%s shell_pid=%s python_pid=%s last_round=%s\n' "$(timestamp)" "${attack}" "${defense}" "${seed}" "$$" "${CURRENT_CHILD_PID}" "$(last_round_from_log "${attempt}/train.log")" >> "${attempt}/heartbeat.log";sleep "${HEARTBEAT_SECONDS}";done)&CURRENT_HEARTBEAT_PID=$!
    wait "${CURRENT_CHILD_PID}"||train_status=$?;wait "${CURRENT_TEE_PID}" 2>/dev/null||true;CURRENT_TEE_PID="";rm -f "${fifo}";finalize_current "${train_status}"
    final_status="$(sed -n 's/^status=//p' "${cell}/status.txt"|head -n 1)"
    if [[ "${final_status}" != COMPLETED ]];then printf 'Cell failed/incomplete: attack=%s defense=%s seed=%s (exit=%s)\n' "${attack}" "${defense}" "${seed}" "${train_status}" >&2;RUN_ONE_FAILED=1;return 0;fi
    printf 'Completed cell: attack=%s defense=%s seed=%s\n' "${attack}" "${defense}" "${seed}"
}

printf 'Results directory: %s\nAttacks: %s\nDefenses: %s\nSeeds: %s\n' "${OUTPUT_ROOT}" "${RUN_ATTACKS[*]}" "${RUN_DEFENSES[*]}" "${RUN_SEEDS[*]}"
rebuild_results
overall_status=0
for defense in "${RUN_DEFENSES[@]}";do for attack in "${RUN_ATTACKS[@]}";do for seed in "${RUN_SEEDS[@]}";do
    run_one "${attack}" "${defense}" "${seed}"
    if ((RUN_ONE_FAILED));then overall_status=1;if ((STOP_ON_ERROR));then break 3;fi;fi
done;done;done
rebuild_results
if ((overall_status));then printf 'Finished with failed/incomplete cells. Results: %s\n' "${OUTPUT_ROOT}" >&2;exit 1;fi
printf 'Completed matrix. Results: %s\n' "${OUTPUT_ROOT}"
