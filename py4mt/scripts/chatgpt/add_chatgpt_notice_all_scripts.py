#!/usr/bin/env python3
"""
Batch-insert a ChatGPT + Author + Date notice across multiple projects (default: .py only).

For Python files:
- Inserts INSIDE the top-level module docstring (or creates one if missing).
- Skips files that already contain a "Generated or modified by ChatGPT" line unless --force is given.

Usage examples:
  python add_notice_all_projects.py /path/to/projectA /path/to/projectB
  python add_notice_all_projects.py --dry-run /code

Options:
  --dry-run          Preview changes; do not write.
  --force            Overwrite even if notice exists.
  --date YYYY-MM-DD  Set a specific date (defaults to today's date: 2025-11-09).
  --author "Name"    Set a specific author line (defaults to "Volker Rath (DIAS)").
"""

import sys
import re
import argparse
from pathlib import Path

CHATGPT_TAG = "Generated or modified by ChatGPT"


def build_notice(date_str: str, author: str) -> str:
    return f"""
---
Generated or modified by ChatGPT (OpenAI GPT-5)
Author: {author}
Date: {date_str}
---
""".strip("\n").format(author=author, date_str=date_str)


def insert_into_python_docstring(text: str, notice: str, force: bool = False) -> str:
    pattern = r'^(?P<prefix>\ufeff?\s*)(?P<quote>["\']{3})(?P<body>.*?)(?P=quote)'
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        prefix, quote, body = m.group(
            "prefix"), m.group("quote"), m.group("body")
        if (CHATGPT_TAG in body) and not force:
            return text
        new_doc = f'{prefix}{quote}{body.rstrip()}\n{notice}{quote}'
        return new_doc + text[m.end():]
    else:
        if (CHATGPT_TAG in text) and not force:
            return text
        return f'"""\n{notice}\n"""\n' + text


def process_file(path: Path, args, notice: str) -> bool:
    try:
        original = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    new_text = insert_into_python_docstring(original, notice, force=args.force)
    if new_text != original:
        if not args.dry_run:
            path.write_text(new_text, encoding="utf-8")
        return True
    return False


def iter_files(root: Path):
    for p in root.rglob("*.py"):
        if p.is_file() and p.name != Path(__file__).name:
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+",
                        help="Files or directories to update")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes; do not write")
    parser.add_argument("--force", action="store_true",
                        help="Force write even if notice exists")
    parser.add_argument("--date", default="2025-11-09",
                        help="Date to write into notice (YYYY-MM-DD)")
    parser.add_argument(
        '--author', default="Volker Rath (DIAS)", help='Author line to use')
    args = parser.parse_args()

    notice = build_notice(args.date, args.author)

    changed = 0
    scanned = 0
    for target in args.paths:
        p = Path(target)
        if p.is_dir():
            for f in iter_files(p):
                scanned += 1
                if process_file(f, args, notice):
                    changed += 1
        elif p.is_file() and p.suffix == ".py":
            scanned += 1
            if process_file(p, args, notice):
                changed += 1
        else:
            print(f"Skip (not found or not a .py): {p}")

    mode = "dry-run" if args.dry_run else "updated"
    print(f"Scanned: {scanned} .py files; {mode}: {changed} files.")


if __name__ == "__main__":
    main()
