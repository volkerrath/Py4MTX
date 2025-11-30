#!/bin/bash
# run_all.sh
# General launcher to run multiple scripts sequentially

# List of scripts to run (absolute or relative paths)
SCRIPTS=(
    "./scriptA.sh"
    "./scriptB.sh"
    "./scriptC.sh"
)

# Mode: "strict" = stop if any script fails
#       "lenient" = continue regardless of exit status
MODE="strict"

for script in "${SCRIPTS[@]}"; do
    echo ">>> Running $script"
    bash "$script"
    status=$?

    if [ "$status" -ne 0 ]; then
        echo "!!! $script exited with status $status"
        if [ "$MODE" = "strict" ]; then
            echo "Stopping execution due to error."
            exit $status
        fi
    fi
done

echo ">>> All scripts finished."
