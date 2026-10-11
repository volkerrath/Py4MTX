#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  histfind.sh [options] <term1> [term2 ...]
Search shell history for lines containing one or more terms.

Options:
  -i        Case-insensitive search
  -a        AND mode: require ALL terms (default is OR/any)
  -r        Treat terms as regex patterns (default: fixed strings)
  -n        Keep history line numbers (default: prints full "history" lines anyway)
  -h        Help

Examples:
  histfind.sh ssh github
  histfind.sh -i python venv
  histfind.sh -a -i conda activate
  histfind.sh -r 'git (commit|push)' 'ssh.*-i'
EOF
}

ignore_case=0
and_mode=0
regex_mode=0

while getopts ":iarnh" opt; do
  case "$opt" in
    i) ignore_case=1 ;;
    a) and_mode=1 ;;
    r) regex_mode=1 ;;
    n) : ;;  # history already includes numbers; kept for UX compatibility
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
  echo "Error: provide at least one search term." >&2
  usage
  exit 2
fi

# Build grep flags
grep_flags=()
(( ignore_case )) && grep_flags+=("-i")
if (( regex_mode )); then
  grep_flags+=("-E")
else
  grep_flags+=("-F")
fi

# Get history text. Note: non-interactive shells may have empty `history`.
hist_text="$(history 2>/dev/null || true)"

if [[ -z "${hist_text}" ]]; then
  # Fallback: try bash history file if available
  hist_file="${HISTFILE:-$HOME/.bash_history}"
  if [[ -r "$hist_file" ]]; then
    hist_text="$(cat -- "$hist_file")"
  else
    echo "No history available (history command empty and no readable HISTFILE)." >&2
    exit 1
  fi
fi

if (( and_mode )); then
  # Require ALL terms: pipe through grep for each term
  out="$hist_text"
  for term in "$@"; do
    out="$(printf '%s\n' "$out" | grep "${grep_flags[@]}" -- "$term" || true)"
  done
  printf '%s\n' "$out"
else
  # OR mode: match ANY term. Build a single pattern.
  if (( regex_mode )); then
    # Join regex terms with |
    pat=""
    for term in "$@"; do
      if [[ -z "$pat" ]]; then pat="$term"; else pat="$pat|$term"; fi
    done
    printf '%s\n' "$hist_text" | grep "${grep_flags[@]}" -- "$pat" || true
  else
    # Fixed strings: best is grep -F with multiple -e
    cmd=(grep "${grep_flags[@]}")
    for term in "$@"; do
      cmd+=("-e" "$term")
    done
    cmd+=("--")
    printf '%s\n' "$hist_text" | "${cmd[@]}" || true
  fi
fi
