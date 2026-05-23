#!/usr/bin/env bash
# run_jobs.sh — run a job in each subdirectory XXX_I for a list of integers I
# Usage: run_jobs.sh I1 [I2 I3 ...]
# VR 2026-05-17

set -euo pipefail

# ── defaults (edit here) ─────────────────────────────────────────────────────
DEFAULT_PREFIX="XXX"
DEFAULT_RUNS=(0 1 2 3)   # fallback list of integers when none are given on the CLI
# ────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $(basename "$0") [--prefix PREFIX] [I1 I2 ...]" >&2
    echo "  --prefix PREFIX   override directory prefix (default: ${DEFAULT_PREFIX})" >&2
    echo "  I1 I2 ...         integer run indices      (default: ${DEFAULT_RUNS[*]})" >&2
    exit 1
}

PREFIX="${DEFAULT_PREFIX}"
RUNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || usage
            PREFIX="$2"; shift 2 ;;
        --prefix=*)
            PREFIX="${1#--prefix=}"; shift ;;
        --help|-h) usage ;;
        --) shift; RUNS+=("$@"); break ;;
        -*)
            echo "Unknown option: $1" >&2; usage ;;
        *)
            RUNS+=("$1"); shift ;;
    esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
    RUNS=("${DEFAULT_RUNS[@]}")
    echo "No run indices given — using defaults: ${RUNS[*]}"
fi

TOPDIR="$(pwd)"

for I in "${RUNS[@]}"; do
    SUBDIR="${PREFIX}_${I}"

    if [[ ! -d "${SUBDIR}" ]]; then
        echo "WARNING: directory '${SUBDIR}' not found — skipping" >&2
        continue
    fi

    echo ">>> entering ${SUBDIR}"
    cd "${SUBDIR}"

    # ── job to be defined ────────────────────────────────────────────────────
    # Replace the line(s) below with the actual command(s) to run.
    ./run_femtic_dias.sh
    # ────────────────────────────────────────────────────────────────────────

    cd "${TOPDIR}"
    echo "<<< back in ${TOPDIR}"
done

echo "Done."
