#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_continue.py

Prepare a set of FEMTIC run directories for continuation from their last
completed iteration.

For every directory in DIR_LIST:
  1. Locate control.dat (normally a symlink into a shared template dir).
  2. Preserve that symlink unchanged as control.orig (only once, unless
     OVERWRITE_ORIG is True).
  3. Determine the best/last completed iteration number in the directory
     by scanning for iteration output files (ITER_FILE_GLOB / ITER_NUMBER_REGEX).
  4. Read the template content, replace the "ITERATION" block

         ITERATION
         0 30

     with

         ITERATION
         XXX YYY

     where XXX is the detected iteration and YYY is MAX_ITERATIONS.
  5. Write the result as a plain (non-symlink) control.dat in the
     directory, replacing the original link.

Nothing is written if DRY_RUN is True; a summary is printed instead.

Author: Volker Rath (DIAS) with Claude
Date:   2026-08-18
"""

import os
import re
import glob
import shutil
import sys

# ============================== USER SECTION ==============================

# Directories to process (each should contain a control.dat symlink and
# the iteration output files from a previous, interrupted or finished run).
DIR_LIST = [
    "/path/to/run01",
    "/path/to/run02",
]

# Name of the control file to adapt (relative to each directory).
CONTROL_FILENAME = "control.dat"

# Name used to preserve the original symlink/file before modification.
ORIG_FILENAME = "control.orig"

# Glob pattern (relative to each directory) used to find per-iteration
# output files, e.g. resistivity_iterXXX.dat. Adjust to match whatever
# FEMTIC variant/version wrote the outputs in these directories.
ITER_FILE_GLOB = "resistivity_iter*.dat"

# Regex applied to each matched filename to extract the iteration number.
# Must contain exactly one capturing group with the integer.
ITER_NUMBER_REGEX = r"iter0*(\d+)"

# Maximum number of iterations to run to, written as the second number in
# the ITERATION block (e.g. 30 in "0 30").
MAX_ITERATIONS = 30

# If True, re-create control.orig even if it already exists (overwriting it).
# If False (default), an existing control.orig is left untouched and taken
# to already hold the pristine template link.
OVERWRITE_ORIG = False

# If True, only print what would be done; do not touch any files.
DRY_RUN = False

# Print progress/details while running.
VERBOSE = True

# =========================== END USER SECTION ==============================


def log(*args):
    if VERBOSE:
        print(*args)


def find_last_iteration(directory):
    """Return the highest iteration number found via ITER_FILE_GLOB /
    ITER_NUMBER_REGEX in `directory`, or None if nothing matched."""
    pattern = os.path.join(directory, ITER_FILE_GLOB)
    matches = glob.glob(pattern)
    rex = re.compile(ITER_NUMBER_REGEX)
    best = None
    for path in matches:
        name = os.path.basename(path)
        m = rex.search(name)
        if m:
            try:
                n = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if best is None or n > best:
                best = n
    return best


def backup_control_link(control_path, orig_path):
    """Preserve control.dat (typically a symlink) as control.orig,
    without following/resolving it, unless it already exists and
    OVERWRITE_ORIG is False."""
    if os.path.lexists(orig_path):
        if not OVERWRITE_ORIG:
            log(f"  control.orig already exists, leaving as is: {orig_path}")
            return
        if not DRY_RUN:
            if os.path.islink(orig_path) or os.path.isfile(orig_path):
                os.remove(orig_path)
            else:
                shutil.rmtree(orig_path)

    if os.path.islink(control_path):
        target = os.readlink(control_path)
        log(f"  backing up symlink control.dat -> {target} as {orig_path}")
        if not DRY_RUN:
            os.symlink(target, orig_path)
    else:
        log(f"  control.dat is a regular file, copying to {orig_path}")
        if not DRY_RUN:
            shutil.copy2(control_path, orig_path)


def set_iteration(content, best_iter):
    """Replace the numbers on the line following the ITERATION keyword.
    Keeps original indentation/spacing and any trailing comment."""
    lines = content.split("\n")
    num_line_re = re.compile(r"^(\s*)(\d+)(\s+)(\d+)(.*)$")

    for i, line in enumerate(lines):
        if line.strip() == "ITERATION" and not line.strip().startswith("#"):
            j = i + 1
            while j < len(lines) and (
                lines[j].strip() == "" or lines[j].strip().startswith("#")
            ):
                j += 1
            if j >= len(lines):
                raise ValueError("ITERATION keyword found but no data line follows")

            m = num_line_re.match(lines[j])
            if not m:
                raise ValueError(
                    f"could not parse ITERATION data line: {lines[j]!r}"
                )
            indent, _old_start, sep, _old_max, tail = m.groups()
            lines[j] = f"{indent}{best_iter}{sep}{MAX_ITERATIONS}{tail}"
            return "\n".join(lines)

    raise ValueError("ITERATION keyword not found in control file")


def process_directory(directory):
    log(f"\n{directory}")

    control_path = os.path.join(directory, CONTROL_FILENAME)
    orig_path = os.path.join(directory, ORIG_FILENAME)

    if not os.path.exists(directory):
        log(f"  SKIP: directory does not exist")
        return
    if not os.path.lexists(control_path):
        log(f"  SKIP: {CONTROL_FILENAME} not found")
        return

    best_iter = find_last_iteration(directory)
    if best_iter is None:
        log(f"  SKIP: no iteration output files matched {ITER_FILE_GLOB!r}")
        return
    log(f"  last completed iteration detected: {best_iter}")

    # Resolve the template content BEFORE touching the link.
    try:
        with open(control_path, "r") as f:
            content = f.read()
    except OSError as e:
        log(f"  SKIP: could not read {control_path}: {e}")
        return

    try:
        new_content = set_iteration(content, best_iter)
    except ValueError as e:
        log(f"  SKIP: {e}")
        return

    backup_control_link(control_path, orig_path)

    log(f"  writing updated {CONTROL_FILENAME} (iteration -> {best_iter})")
    if not DRY_RUN:
        if os.path.islink(control_path) or os.path.exists(control_path):
            os.remove(control_path)
        with open(control_path, "w") as f:
            f.write(new_content)


def main():
    if DRY_RUN:
        log("DRY RUN -- no files will be modified\n")
    for directory in DIR_LIST:
        process_directory(directory)


if __name__ == "__main__":
    main()
