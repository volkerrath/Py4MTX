# mt_archive_run

Clean and optionally archive (currently FEMTIC-style) iteration directories.

This utility keeps selected low- and high-iteration files, preserves protected files, deletes the rest (optionally), and creates a compressed archive (`.zip`, `.tgz`, `.tar.gz`) of the retained files.

---

## ✨ Features

- Select files by iteration pattern (default: `_iter(\d+)`)
- Keep:
  - lowest `keep_n_low` iterations
  - highest `keep_n_high` iterations
- Never delete protected files:
  - by substring tokens (e.g. `obs`, `mesh`)
  - by suffix (e.g. `.log`, `.cnv`)
- Case-insensitive matching
- Safe by default (`dry-run`)
- Optional deletion (`--delete`)
- Optional compression:
  - `.zip`
  - `.tgz` / `.tar.gz`
- Optional inclusion of leading directory in archive

---

## 📂 Typical Use Case

FEMTIC / MT inversion runs often produce many files like:

```
model_iter0.dat
model_iter1.dat
...
model_iter100.dat
log.txt
mesh.cnv
```

This tool allows you to:

- keep only selected iterations (e.g. first + last)
- retain important metadata/log files
- clean the directory
- archive the result

---

## 🚀 Usage

### Basic (dry-run)
```
python mt_archive_run.py .
```

### Keep lowest + highest iteration
```
python mt_archive_run.py . \
    --keep-n-low 1 \
    --keep-n-high 1
```

### Actually delete unwanted files
```
python mt_archive_run.py . --delete
```

### Recursive mode
```
python mt_archive_run.py results --recursive
```

---

## 📦 Compression

### Create archive (recommended)
```
python mt_archive_run.py run \
    --delete \
    --compress run_cleaned.tgz
```

### Supported formats
- `.zip`
- `.tgz`
- `.tar.gz`

---

## 📁 Archive Structure

### Default (includes leading directory)
```
python mt_archive_run.py run --compress out.tgz
```

Archive contents:
```
run/
  model_iter100.dat
  mesh.cnv
  log.txt
```

### Without leading directory
```
python mt_archive_run.py run --compress out.tgz --no-root
```

Archive contents:
```
model_iter100.dat
mesh.cnv
log.txt
```

---

## 🔒 Protected Files

### By default

#### Tokens (substring match, case-insensitive)
```
obs, ref, mesh, iter0, control
```

#### Suffixes (full filename match)
```
.log, .sh, .cnv
```

Examples:
```
mesh.cnv      → protected
run.log       → protected
observations.dat → protected (token match)
```

---

## ⚙️ Options

| Option | Description |
|------|-------------|
| `directory` | Directory to process (default: `.`) |
| `--keep-n-low` | Number of lowest iterations to keep |
| `--keep-n-high` | Number of highest iterations to keep |
| `--recursive` | Scan subdirectories |
| `--delete` | Actually delete files (otherwise dry-run) |
| `--compress` | Output archive path |
| `--no-root` | Do not include leading directory in archive |

---

## 🧠 Notes

- Matching is **case-insensitive**
- Only files matching the iteration pattern are filtered
- Protected files are **never deleted**
- Archive paths are **relative** (portable)
- Default behavior is **safe (no deletion)**

---

## 🔧 Example Workflow

```
# preview
python mt_archive_run.py run

# clean + archive
python mt_archive_run.py run \
    --keep-n-low 1 \
    --keep-n-high 2 \
    --delete \
    --compress run_final.tgz
```

---

## 📜 License / Provenance

Author: Volker Rath (DIAS)  
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-07
