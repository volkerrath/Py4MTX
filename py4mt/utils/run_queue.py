#!/usr/bin/env python3
"""
Submit a chain of OAR jobs from a list specification.

Each job depends on the previous one (linear pipeline).



jobs = [
    {"script": "mt_preprocess.sh", "name": "pre"},
    {"script": "mt_forward.sh", "name": "fwd", "resources": "nodes=1/core=16"},
    {"script": "mt_invert.sh", "name": "inv"},
    {"script": "mt_plot.sh", "name": "plot"},
]

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-03-27
"""

from __future__ import annotations

import re
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Any


# --- job id parsing ----------------------------------------------------------

_PATTERNS = [
    re.compile(r"OAR_JOB_ID=(\d+)"),
    re.compile(r"IdJob\s*=\s*(\d+)"),
    re.compile(r"^\s*(\d+)\s*$", re.MULTILINE),
]


def extract_job_id(text: str) -> int:
    for p in _PATTERNS:
        m = p.search(text)
        if m:
            return int(m.group(1))
    raise RuntimeError(f"Could not parse job id from:\n{text}")


# --- command runner ----------------------------------------------------------

def run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(shlex.quote(c) for c in cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


# --- single job submission ---------------------------------------------------

def submit_job(job: Dict[str, Any], after: int | None = None) -> int:
    cmd = ["oarsub"]

    if job.get("name"):
        cmd += ["-n", job["name"]]

    if job.get("queue"):
        cmd += ["-q", job["queue"]]

    # resources / walltime
    res_parts = []
    if job.get("resources"):
        res_parts.append(job["resources"])
    if job.get("walltime"):
        res_parts.append(f"walltime={job['walltime']}")

    if res_parts:
        cmd += ["-l", "/".join(res_parts)]

    if after is not None:
        cmd += ["--after", str(after)]

    # extra args
    if job.get("extra"):
        cmd += job["extra"]

    script = str(Path(job["script"]).expanduser())
    cmd.append(script)

    out = run(cmd)
    job_id = extract_job_id(out)

    print(f"[OK] {job.get('name','job')} → {job_id}")
    return job_id


# --- chain submission --------------------------------------------------------

def submit_chain(jobs: List[Dict[str, Any]]) -> List[int]:
    """
    Submit a linear chain of jobs.

    Parameters
    ----------
    jobs : list of dict
        Job specifications.

    Returns
    -------
    list of int
        Job IDs in submission order.
    """
    ids: List[int] = []
    prev_id: int | None = None

    for job in jobs:
        jid = submit_job(job, after=prev_id)
        ids.append(jid)
        prev_id = jid

    return ids


# --- example usage -----------------------------------------------------------

if __name__ == "__main__":
    jobs = [
        {"script": "preprocess.sh", "name": "pre", "walltime": "00:30:00"},
        {"script": "forward.sh", "name": "fwd", "walltime": "02:00:00"},
        {"script": "invert.sh", "name": "inv", "walltime": "04:00:00"},
        {"script": "plot.sh", "name": "plot", "walltime": "00:20:00"},
    ]

    ids = submit_chain(jobs)
    print("Chain submitted:", ids)
