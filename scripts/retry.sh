#!/usr/bin/env bash
# Usage: retry.sh <log> <python code>. Re-runs the command until it exits 0 (max 50 tries).
log=$1; shift
for i in $(seq 1 50); do
  uv run --offline python -c "$1" >> "$log" 2>&1 && { echo "DONE ok (attempt $i)" >> "$log"; exit 0; }
  echo "ATTEMPT $i failed, retrying" >> "$log"; sleep 30
done
echo "GAVE UP" >> "$log"; exit 1
