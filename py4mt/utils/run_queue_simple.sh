#!/bin/bash
# run_all.sh
# General launcher to run multiple scripts sequentially with timestamped logging

# List of scripts to run — literal paths and/or glob patterns, e.g.:
#   "./scriptA.sh"          literal path
#   "./stage2_*.sh"         glob: all matching files, sorted
#   "/opt/jobs/step?.sh"    glob with single-char wildcard
SCRIPTS=(
    "./scriptA.sh"
    "./scriptB.sh"
    "./scriptC.sh"
)

# Mode: "strict" = stop if any script fails
#       "lenient" = continue regardless of exit status
MODE="strict"

# Log file (timestamped)
LOGFILE="run_all_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

log "Launcher started. Mode=$MODE"

# Expand SCRIPTS entries: globs are sorted, literals kept as-is.
# Entries that match nothing are skipped with a warning.
RESOLVED=()
for entry in "${SCRIPTS[@]}"; do
    # Check whether the entry contains a glob character
    if [[ "$entry" == *[\*\?\[]* ]]; then
        # Use nullglob so unmatched globs produce nothing
        shopt -s nullglob
        matches=( $entry )
        shopt -u nullglob
        if [ ${#matches[@]} -eq 0 ]; then
            log "[WARN] glob matched nothing: $entry"
        else
            # Sort matches and append
            IFS=$'\n' sorted=($(sort <<<"${matches[*]}")); unset IFS
            RESOLVED+=("${sorted[@]}")
        fi
    else
        RESOLVED+=("$entry")
    fi
done

if [ ${#RESOLVED[@]} -eq 0 ]; then
    log "No scripts to run. Exiting."
    exit 0
fi

log "Scripts to run (${#RESOLVED[@]}):"
for s in "${RESOLVED[@]}"; do log "  $s"; done

for script in "${RESOLVED[@]}"; do
    if [ ! -f "$script" ]; then
        log "[WARN] script not found, skipping: $script"
        continue
    fi
    log ">>> Starting $script"
    bash "$script"
    status=$?

    if [ "$status" -eq 0 ]; then
        log ">>> Finished $script successfully."
    else
        log "!!! $script exited with status $status"
        if [ "$MODE" = "strict" ]; then
            log "Stopping execution due to error."
            exit $status
        fi
    fi
done

log "All scripts finished."
