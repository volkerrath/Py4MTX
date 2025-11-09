#!/usr/bin/env python3
import sys, re
from pathlib import Path
from datetime import datetime

NOTICE = """
---
Generated or modified by ChatGPT (OpenAI GPT-5)
Author: Volker Rath (DIAS)
Date: {date}
---
""".format(date="2025-11-09")

def insert_notice(text: str, notice: str) -> str:
    pattern = r'^(?P<prefix>\ufeff?\s*)(?P<quote>["\']{3})(?P<body>.*?)(?P=quote)'
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        prefix, quote, body = m.group("prefix"), m.group("quote"), m.group("body")
        if "Generated or modified by ChatGPT" in body:
            return text
        new_doc = f'{prefix}{quote}{body.rstrip()}\n{notice}{quote}'
        return new_doc + text[m.end():]
    else:
        return f'"""{notice.strip()}\n"""\n' + text

def process_file(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    new_txt = insert_notice(txt, NOTICE)
    if new_txt != txt:
        path.write_text(new_txt, encoding="utf-8")
        print(f"Updated: {path}")
    else:
        print(f"Skipped (already has notice): {path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: add_notice_to_docstrings.py <file_or_directory> [more ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for py in p.rglob("*.py"):
                process_file(py)
        elif p.suffix == ".py" and p.exists():
            process_file(p)
        else:
            print(f"Not found or not a .py: {p}")

if __name__ == "__main__":
    main()
