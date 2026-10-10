#!/usr/bin/env bash
# Launch ONE resumable, self-retrying single-cell fetch per line (idempotent: finished
# shards are skipped). Figshare lines get 32 connections, NCBI lines 24.
cd "$(dirname "$0")/.."
for spec in k562:32 rpe1:32 hepg2:24 jurkat:24; do
  n=${spec%%:*}; t=${spec##*:}
  nohup scripts/retry.sh data/fetch_$n.log "from pathlib import Path; import json; from canpert.ingest import fetch_line; fetch_line('$n', Path('data/raw/perturbseq'), json.loads(Path('data/panel_all4.json').read_text()), threads=$t)" >/dev/null 2>&1 &
done
