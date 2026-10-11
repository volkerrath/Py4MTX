#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file> <directory_pattern>"
    exit 1
fi

file="$1"
pattern="$2"

for d in $pattern; do
    [ -d "$d" ] && cp -p "$file" "$d/"
done
