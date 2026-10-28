# femtic_clean.py

Delete all files in a list of directories except a list of protected files.

For every directory in `DIRS` (names or glob patterns) all files are removed
except those whose name matches one of the patterns in `FILES`. **Nothing is
deleted unless `--delete` is given**: the default is a dry run that reports,
per directory, how many files would be kept and deleted and how much space
would be freed.

Typical use: strip the output of many FEMTIC runs (`fem_001/`, `fem_002/`,
...) down to the files that are worth keeping.

---

## Usage

Edit the `USER SECTION` at the top of the script, or pass everything on the
command line. What is given on the command line replaces the corresponding
`USER SECTION` entry.

```bash
# dry run (default): only report
python femtic_clean.py --dirs './fem*' --files a.dat b.h5 'x*iter*.dat'

# really delete
python femtic_clean.py --dirs './fem*' --files a.dat b.h5 'x*iter*.dat' --delete

# key=value form (same thing); brackets, quotes and commas are optional
python femtic_clean.py dirs='./fem*' files=['a.dat', 'b.h5', 'x*iter*.dat']
python femtic_clean.py dirs=./fem* files=a.dat,b.h5,'x*iter*.dat' --delete

# several directories, also look into their subdirectories, list every file
python femtic_clean.py --dirs run_a run_b --files 'control.dat' 'results_iter*.h5' --recursive -v
```

Quote glob patterns (single quotes) so that the shell does not expand them
before the script sees them; the script expands `DIRS` itself. In the
`key=value` form items are separated by blanks and/or commas; with `--dirs` /
`--files` only by blanks.

### Command-line options

| Option | Meaning |
|---|---|
| `--dirs DIR [DIR ...]` or `dirs=...` | Directories or glob patterns to clean. Replaces `DIRS`. |
| `--files PATTERN [...]` or `files=...` | Protected file patterns. Replaces `FILES`. |
| `--work-dir DIR` | Base for relative `--dirs` patterns and location of the summary. Replaces `WORK_DIR`. |
| `--recursive` | Also clean files in subdirectories. |
| `--delete` | Really delete. Without it: dry run. |
| `--verbose`, `-v` | List every file with its KEEP / DELETE decision. |
| `--force` | Allow an empty protect list and protected patterns that match nothing (see Safety). |

Exit code: `0` ok, `1` some files could not be deleted (or read), `2` the run
was refused (bad arguments or a safety check).

### User section

| Parameter | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `"./"` | Base for relative `DIRS` patterns; the summary is written here. |
| `DIRS` | `[]` | Directories / glob patterns to clean. |
| `FILES` | `[]` | Protected file patterns. Empty = refused unless `FORCE`. |
| `RECURSIVE` | `False` | Also clean subdirectories. |
| `DELETE` | `False` | `True` = really delete (`False` = dry run). |
| `VERBOSE` | `False` | List every file. |
| `FORCE` | `False` | Skip the empty-protect-list and unmatched-pattern checks. |

The defaults are empty on purpose: the script does nothing until it is told
which directories to clean and which files to keep.

---

## Matching rules

- **`DIRS`**: directory names or shell wildcards. A relative pattern is taken
  relative to `WORK_DIR` (default: the current directory, so `./fem*` means
  what it says). Matches that are not directories are ignored; a directory
  reached twice (also through a symlink) is cleaned once.
- **`FILES`**: shell wildcards (`*`, `?`, `[seq]`), **case-sensitive**.
  - A pattern without `/` is matched against the file **name** only
    (`x*iter*.dat`); with `--recursive` it therefore protects matching files
    at any depth.
  - A pattern with `/` is matched against the path **relative to the cleaned
    directory** (`plots/*`, `sub/b.h5`).
- **Files only.** Without `--recursive` only the files directly inside each
  directory are considered; subdirectories are left alone and just counted
  (column `subdirs`). With `--recursive` the files below them are cleaned
  too. Directories are never removed, also not when they end up empty.
- **Symbolic links** are never followed. A link to a file (also a broken one)
  counts as a file: the link is removed, its target is not. A link to a
  directory is treated like a directory: not entered, not removed.
- With `--recursive`, a listed directory that lies inside another listed
  directory is cleaned as part of the outer one and not scanned twice.
- **The script itself** (also through a symlink) is always protected, so it
  can be run from inside a directory it cleans.

---

## Safety

- Dry run unless `--delete` / `DELETE = True`.
- No protected pattern at all: refused, because that would delete everything.
- With `--delete`: refused if a protected pattern matches **no file in any**
  of the directories (usually a typo such as `x*iter*.dta`). In a dry run
  this is only a warning; the table of pattern hit counts shows it as
  `<-- matches nothing`. A pattern that matches in some directories but not
  in others (a failed run without `b.h5`, say) is fine.
- `/` and the home directory are never cleaned.
- `--force` switches off the two checks on the protect list.
- Standard library only, so it also runs on a login node without the Py4MTX
  environment. It reads no `PY4MTX_*` variables and imports no py4mt module.

Check a new pattern list with a dry run (`-v` lists every decision) before
adding `--delete`.

---

## Output

Printed table (dry run):

```
==============================================================================
 femtic_clean: DRY RUN (nothing is deleted; use --delete)
==============================================================================
 Protected patterns : a.dat, b.h5, x*iter*.dat
 Subdirectories     : not searched
------------------------------------------------------------------------------
 directory    files     keep   delete         size  subdirs
------------------------------------------------------------------------------
 fem_001         11        4        7      6.8 KiB        2
 fem_002         13        4        9      6.9 KiB        3
 fem_003         10        3        7      6.8 KiB        2
------------------------------------------------------------------------------
 total           34       11       23     20.5 KiB
------------------------------------------------------------------------------
 Files matched by each protected pattern (all directories):
   a.dat              3
   b.h5               2
   x*iter*.dat        6
------------------------------------------------------------------------------
 Would delete 23 of 34 file(s), freeing 20.5 KiB. Nothing was changed.
==============================================================================
```

Sizes are those of the files themselves (a symbolic link counts as the link,
not its target).

`<WORK_DIR>/femtic_clean_summary.md` is written at the end of every run
(dry run included): the effective parameters (command line already applied),
script path, run date/time, and a result table (mode, directories, files
seen / kept / deleted, bytes, errors). If `WORK_DIR` cannot be written, the
summary goes next to the script instead.

---

## Tested

On a mock tree (three run directories with iteration files, subdirectories,
symlinks to a file, to a directory and a broken one, plus a non-matching
sibling directory):

- dry run and `--delete` with the `dirs=... files=[...]` form typed in bash,
  the `--dirs` / `--files` form, and comma / bracket / quote variants;
- `--recursive` with name and path patterns, nested listed directories;
- refusals (no protect list, typo pattern with `--delete`, `/`, home, no
  matching directory) and `--force`;
- symlinks (file link removed and target kept, directory link kept, broken
  link removed), the script protected when it sits in the cleaned directory;
- summary written to `--work-dir` (not next to the script), fallback when
  `WORK_DIR` is not writable, unlink errors giving exit code 1, `USER
  SECTION` only (no command line).

Not tested: a real FEMTIC run directory, very large directories, or
non-POSIX file systems.

---

## Dependencies

Python 3, standard library only (`argparse`, `fnmatch`, `glob`, `os`,
`sys`, `datetime`, `typing`).

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | Created. Dry run by default; `--delete` to act; protected patterns by file name or relative path; optional recursion; `key=value` command-line form; stdlib-only summary writer that puts `femtic_clean_summary.md` into `WORK_DIR`. AI-generated; review before production use (the script deletes files). |

Author: Volker Rath (DIAS)
