#!/usr/bin/env bash
# =============================================================================
# copy_to_dirs.sh - copy (or symlink) a set of files to a set of directories
# =============================================================================
# Source files and destination directories are both given as glob patterns.
# With -l, relative symbolic links are created instead of copies.
# Run with -h for usage.
#
# Requires bash >= 4.4 (associative arrays, globstar).
#
# Provenance: written by Claude (Anthropic), 2026-10-20.
# AI-generated code - review before production use.
# =============================================================================

set -u

PROG=${0##*/}

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    echo "$PROG: bash >= 4.4 is required (found $BASH_VERSION)" >&2
    exit 2
fi

# nullglob : a pattern without matches expands to nothing
# globstar : ** matches recursively
shopt -s nullglob globstar

# cd must never print or search CDPATH (used when resolving physical paths)
unset CDPATH

# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage:
  $PROG [options] -s FILE_PATTERN... -d DIR_PATTERN...

Copy every file matching the -s pattern(s) into every directory matching the
-d pattern(s). With -l, create relative symbolic links to the files instead.

Selection:
  -s, --src PATTERN...    source file pattern(s); may be repeated
  -d, --dest PATTERN...   destination directory pattern(s); may be repeated

Options:
  -l, --link         create relative symbolic links instead of copies
                     (also --symlink); -p has no effect in this mode
  -n, --dry-run      show what would be done, change nothing
  -v, --verbose      print every operation
  -r, --recursive    also include matching directories (cp -r, or link them
                     as a whole with -l)
  -p, --preserve     preserve mode, ownership and timestamps (cp -p)
  -m, --mkdir        create destination directories that are given WITHOUT
                     wildcards and do not exist yet (mkdir -p)
  -c, --no-clobber   never overwrite an existing file in a destination
  -u, --update       overwrite only if the source is newer than the target
  -h, --help         show this help
  --                 treat all remaining arguments as patterns

Notes:
  - By default, existing files in the destinations are overwritten (with -l:
    replaced by the link).
  - Patterns support *, ?, [...] and ** (recursive). Quote them so that this
    script expands them (avoids "argument list too long"). Unquoted patterns
    are expanded by the calling shell, which works too.
  - After -s or -d, all following arguments up to the next option belong to
    that option, so -s a*.dat b*.dat -d run_* out_* is fine.
  - Single-letter options may be bundled, e.g. -nv.
  - A pattern that names an existing path literally also matches that path,
    so file names containing [ ] * ? survive unquoted shell expansion.
  - A file is never copied or linked onto itself.
  - With -l, the link text is relative to each destination directory and is
    computed from physical paths (symbolic links in directory names are
    resolved), so the links keep working if source and destination trees are
    moved together. A link that already has the right target is skipped, and
    a real directory is never replaced by a link.
  - Exit status: 0 = ok, 1 = nothing to do or some operations failed,
    2 = usage error.

Examples:
  $PROG -s '*.dat' '*.cfg' -d 'run_*' 'out/case_??'
  $PROG -nv -s model.dat -d 'ens_*'                 # dry run, verbose
  $PROG -s inputs/*.txt -d 'sim/**/'                # sim/ and all subdirs
  $PROG -m -s run.sh -d results/a results/b         # create dirs if missing
  $PROG -rc -s 'templates/*' -d 'proj_*'            # dirs too, keep existing
  $PROG -l -s 'common/*.cfg' -d 'run_*'             # run_01/x.cfg -> ../common/x.cfg
EOF
}

die()  { echo "$PROG: error: $*" >&2; exit 2; }
warn() { echo "$PROG: warning: $*" >&2; }

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
src_patterns=()
dst_patterns=()
dry_run=0
verbose=0
recursive=0
preserve=0
mkdirs=0
link=0
mode=overwrite          # overwrite | noclobber | update

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
cur=""                  # which list the following words belong to: src | dst
past_dashdash=0

while (( $# )); do
    arg=$1
    shift

    if (( ! past_dashdash )); then
        case $arg in
            -s|--src|--source)  cur=src; continue ;;
            -d|--dest|--dir)    cur=dst; continue ;;
            -l|--link|--symlink) link=1; continue ;;
            -n|--dry-run)       dry_run=1; continue ;;
            -v|--verbose)       verbose=1; continue ;;
            -r|--recursive)     recursive=1; continue ;;
            -p|--preserve)      preserve=1; continue ;;
            -m|--mkdir)         mkdirs=1; continue ;;
            -c|--no-clobber)    mode=noclobber; continue ;;
            -u|--update)        mode=update; continue ;;
            -h|--help)          usage; exit 0 ;;
            --)                 past_dashdash=1; continue ;;
            -[!-]?*)
                # bundled short options, e.g. -nv -> -n -v
                rest=${arg#-}
                split=()
                for (( i = 0; i < ${#rest}; i++ )); do
                    split+=("-${rest:i:1}")
                done
                set -- "${split[@]}" "$@"
                continue
                ;;
            -*)                 die "unknown option: $arg (use -- before patterns that start with '-')" ;;
        esac
    fi

    case $cur in
        src) src_patterns+=("$arg") ;;
        dst) dst_patterns+=("$arg") ;;
        *)   die "'$arg' is not preceded by -s or -d" ;;
    esac
done

(( ${#src_patterns[@]} )) || die "no source patterns given (-s)"
(( ${#dst_patterns[@]} )) || die "no destination patterns given (-d)"

# -c and -u both given: the last one wins silently (mode is a single variable)

(( link && preserve )) && warn "-p has no effect together with -l"

cp_opts=()
(( recursive )) && cp_opts+=(-r)
(( preserve ))  && cp_opts+=(-p)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
GLOB_HITS=()

# glob_expand PATTERN -> fills GLOB_HITS. Expands ~ and glob characters but
# does no word splitting, so names with spaces are safe. If the pattern also
# names an existing path literally (e.g. a file called 'br[ack].dat' that the
# calling shell passed through unquoted), that path is accepted as well.
glob_expand() {
    local pat=$1 oldifs=$IFS h found=0
    pat=${pat/#\~/$HOME}
    IFS=
    # unquoted on purpose: pathname expansion, no splitting (IFS is empty)
    GLOB_HITS=( $pat )
    IFS=$oldifs
    if [[ -e $pat || -L $pat ]]; then
        for h in "${GLOB_HITS[@]}"; do
            [[ $h == "$pat" ]] && { found=1; break; }
        done
        (( found )) || GLOB_HITS+=("$pat")
    fi
}

# relpath TARGET_ABS FROM_DIR_ABS -> sets REL to the path of TARGET as seen
# from FROM_DIR. Both arguments must be absolute and free of . and .. parts.
REL=""
relpath() {
    local target=$1 from=$2 IFS=/
    local -a t=() fr=() out=()
    local i=0 j
    read -ra t <<< "${target#/}"
    read -ra fr <<< "${from#/}"
    while (( i < ${#t[@]} && i < ${#fr[@]} )) && [[ ${t[i]} == "${fr[i]}" ]]; do
        i=$(( i + 1 ))
    done
    for (( j = i; j < ${#fr[@]}; j++ )); do out+=(..); done
    for (( j = i; j < ${#t[@]}; j++ )); do out+=("${t[j]}"); done
    REL="${out[*]:-.}"          # IFS is "/" here, so this joins with "/"
}

declare -a files=() dirs=()
declare -A seen_src=() seen_dst=()
declare -A src_abs=() dst_abs=()

add_dir() {
    local d=$1
    [[ -n ${seen_dst["$d"]:-} ]] && return 0
    seen_dst["$d"]=1
    dirs+=("$d")
    return 0
}

# -----------------------------------------------------------------------------
# Collect source files
# -----------------------------------------------------------------------------
for pat in "${src_patterns[@]}"; do
    glob_expand "$pat"
    n=0
    for f in "${GLOB_HITS[@]}"; do
        f=${f%/}
        [[ -n $f ]] || continue
        if [[ -d $f ]]; then
            if (( ! recursive )); then
                (( verbose )) && echo "skipping directory (use -r): $f"
                continue
            fi
        elif [[ ! -f $f ]]; then
            continue
        fi
        [[ -n ${seen_src["$f"]:-} ]] && continue
        seen_src["$f"]=1
        files+=("$f")
        n=$(( n + 1 ))
    done
    (( n )) || warn "no files match: $pat"
done

# -----------------------------------------------------------------------------
# Collect destination directories
# -----------------------------------------------------------------------------
for pat in "${dst_patterns[@]}"; do
    glob_expand "$pat"
    n=0
    for d in "${GLOB_HITS[@]}"; do
        d=${d%/}
        [[ -n $d ]] || d=/
        [[ -d $d ]] || continue
        add_dir "$d"
        n=$(( n + 1 ))
    done

    if (( n == 0 )); then
        p=${pat/#\~/$HOME}
        case $p in
            *[\*\?\[]*)
                warn "no directories match: $pat"
                ;;
            *)
                if (( mkdirs )); then
                    p=${p%/}
                    if (( dry_run )); then
                        echo "[dry-run] mkdir -p $p"
                        add_dir "$p"
                    elif mkdir -p -- "$p"; then
                        (( verbose )) && echo "created: $p"
                        add_dir "$p"
                    else
                        warn "cannot create directory: $p"
                    fi
                else
                    warn "directory does not exist (use -m to create it): $pat"
                fi
                ;;
        esac
    fi
done

if (( ${#files[@]} == 0 || ${#dirs[@]} == 0 )); then
    echo "$PROG: nothing to do (${#files[@]} file(s), ${#dirs[@]} dir(s) matched)" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Link mode: resolve physical absolute paths once, up front
#   src_abs[f] : physical directory of f + "/" + its own name (a source that
#                is itself a symlink is linked as such, not resolved)
#   dst_abs[d] : physical directory d (unset if it does not exist yet, which
#                can only happen in a dry run together with -m)
# -----------------------------------------------------------------------------
if (( link )); then
    for f in "${files[@]}"; do
        if [[ $f == */* ]]; then fdir=${f%/*}; else fdir=.; fi
        [[ -n $fdir ]] || fdir=/
        if fabs=$(cd -P -- "$fdir" 2>/dev/null && pwd -P); then
            [[ $fabs == / ]] && fabs=""
            src_abs["$f"]=$fabs/${f##*/}
        fi
    done
    for d in "${dirs[@]}"; do
        if dabs=$(cd -P -- "$d" 2>/dev/null && pwd -P); then
            dst_abs["$d"]=$dabs
        fi
    done
fi

# -----------------------------------------------------------------------------
# Copy / link
# -----------------------------------------------------------------------------
n_done=0
n_skipped=0
n_failed=0

for d in "${dirs[@]}"; do
    for f in "${files[@]}"; do
        target=$d/${f##*/}
        REL=""

        if (( link )); then
            sabs=${src_abs["$f"]:-}
            dabs=${dst_abs["$d"]:-}
            if [[ -z $sabs || -z $dabs ]]; then
                if (( dry_run )); then
                    # destination does not exist yet (dry run with -m)
                    echo "[dry-run] ln -s <path to $f relative to $d/> $target"
                    n_done=$(( n_done + 1 ))
                else
                    warn "cannot resolve paths for: $f -> $d/"
                    n_failed=$(( n_failed + 1 ))
                fi
                continue
            fi
            relpath "$sabs" "$dabs"
        fi

        # same file (e.g. destination is the source's own directory); in link
        # mode an existing symlink is judged below by its link text instead
        if [[ $f -ef $target ]] && { (( ! link )) || [[ ! -L $target ]]; }; then
            (( verbose )) && echo "skip (same file): $f -> $d/"
            n_skipped=$(( n_skipped + 1 ))
            continue
        fi

        if [[ -e $target || -L $target ]]; then
            if (( link )) && [[ -L $target && $(readlink -- "$target") == "$REL" ]]; then
                (( verbose )) && echo "skip (already linked): $target -> $REL"
                n_skipped=$(( n_skipped + 1 ))
                continue
            fi
            case $mode in
                noclobber)
                    (( verbose )) && echo "skip (exists): $target"
                    n_skipped=$(( n_skipped + 1 ))
                    continue
                    ;;
                update)
                    if [[ ! $f -nt $target ]]; then
                        (( verbose )) && echo "skip (not newer): $target"
                        n_skipped=$(( n_skipped + 1 ))
                        continue
                    fi
                    ;;
            esac
        fi

        if (( link )); then
            if [[ -d $target && ! -L $target ]]; then
                warn "cannot replace a directory with a link: $target"
                n_failed=$(( n_failed + 1 ))
            elif (( dry_run )); then
                echo "[dry-run] ln -s $REL $target"
                n_done=$(( n_done + 1 ))
            elif ln -sfn -- "$REL" "$target"; then
                (( verbose )) && echo "link: $target -> $REL"
                n_done=$(( n_done + 1 ))
            else
                warn "link failed: $target"
                n_failed=$(( n_failed + 1 ))
            fi
            continue
        fi

        if (( dry_run )); then
            echo "[dry-run] $f -> $d/"
            n_done=$(( n_done + 1 ))
            continue
        fi

        if cp "${cp_opts[@]}" -- "$f" "$d/"; then
            (( verbose )) && echo "copy: $f -> $d/"
            n_done=$(( n_done + 1 ))
        else
            warn "copy failed: $f -> $d/"
            n_failed=$(( n_failed + 1 ))
        fi
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
if (( link )); then
    if (( dry_run )); then verb="would link"; else verb=linked; fi
else
    if (( dry_run )); then verb="would copy"; else verb=copied; fi
fi
printf '%s: %d %s, %d skipped, %d failed  (%d source(s) x %d dir(s))\n' \
    "$PROG" "$n_done" "$verb" "$n_skipped" "$n_failed" \
    "${#files[@]}" "${#dirs[@]}"

(( n_failed )) && exit 1
exit 0
