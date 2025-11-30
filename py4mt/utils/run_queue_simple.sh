# #!/bin/bash
# # run_all.sh
# # General launcher to run multiple scripts sequentially
#
# # List of scripts to run (absolute or relative paths)
# SCRIPTS=(
#     "./scriptA.sh"
#     "./scriptB.sh"
#     "./scriptC.sh"
# )
#
# # Mode: "strict" = stop if any script fails
# #       "lenient" = continue regardless of exit status
# MODE="strict"
#
# for script in "${SCRIPTS[@]}"; do
#     echo ">>> Running $script"
#     bash "$script"
#     status=$?
#
#     if [ "$status" -ne 0 ]; then
#         echo "!!! $script exited with status $status"
#         if [ "$MODE" = "strict" ]; then
#             echo "Stopping execution due to error."
#             exit $status
#         fi
#     fi
# done
#
# echo ">>> All scripts finished."

#!/bin/bash
# run_all.sh
# General launcher to run multiple scripts sequentially with timestamped logging

# List of scripts to run
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

for script in "${SCRIPTS[@]}"; do
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
