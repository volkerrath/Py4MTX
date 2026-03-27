#!/usr/bin/env bash
set -euo pipefail

# --- function: extract job id from oarsub output -----------------------------
get_job_id() {
    # works for multiple OAR output formats
    grep -Eo 'OAR_JOB_ID=[0-9]+' | cut -d= -f2 \
    || grep -Eo 'IdJob *= *[0-9]+' | grep -Eo '[0-9]+' \
    || grep -Eo '^[0-9]+$'
}

# --- function: submit one job ------------------------------------------------
submit_job() {
    local script="$1"
    local name="$2"
    local walltime="$3"
    local resources="$4"
    local after="${5:-}"

    cmd=(oarsub)

    [[ -n "$name" ]] && cmd+=(-n "$name")

    # resources + walltime
    res=""
    [[ -n "$resources" ]] && res="$resources"
    [[ -n "$walltime" ]] && res="${res:+$res/}walltime=$walltime"
    [[ -n "$res" ]] && cmd+=(-l "$res")

    [[ -n "$after" ]] && cmd+=(--after "$after")

    cmd+=("$script")

    echo "[SUBMIT] ${cmd[*]}"
    out="$("${cmd[@]}" 2>&1)"

    job_id="$(echo "$out" | get_job_id)"

    if [[ -z "$job_id" ]]; then
        echo "ERROR: could not parse job id"
        echo "$out"
        exit 1
    fi

    echo "[OK] $name → $job_id"
    echo "$job_id"
}

# --- define job list ---------------------------------------------------------
# format: "script|name|walltime|resources"

jobs=(
  "preprocess.sh|pre|00:30:00|"
  "forward.sh|fwd|02:00:00|nodes=1/core=8"
  "invert.sh|inv|04:00:00|"
  "plot.sh|plot|00:20:00|"
)

# --- submit chain ------------------------------------------------------------
prev=""

ids=()

for job in "${jobs[@]}"; do
    IFS="|" read -r script name walltime resources <<< "$job"

    jid=$(submit_job "$script" "$name" "$walltime" "$resources" "$prev")

    ids+=("$jid")
    prev="$jid"
done

echo "Chain submitted: ${ids[*]}"
