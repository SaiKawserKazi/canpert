#!/usr/bin/env bash
# Full real-data pipeline. Two jobs at a time, 2 threads each. Finished stages are skipped.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R="uv run --no-sync canpert"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a data/pipeline.log; }
[ -f data/processed/audit.json ] || { log "build-data"; $R build-data >> data/pipeline.log 2>&1 || { log "FAILED build-data"; exit 1; }; }
[ -f artifacts/observed/observed_rpe1.npz ] || { log "observed"; $R observed >> data/pipeline.log 2>&1 || { log "FAILED observed"; exit 1; }; }
log "t1 + t2 (20 epochs)"
($R t1 --epochs 20 > data/t1.log 2>&1; echo "t1 exit $?" >> data/pipeline.log) &
($R t2 --epochs 20 > data/t2.log 2>&1; echo "t2 exit $?" >> data/pipeline.log) & wait
log "ablation + transfer (10 epochs)"
($R ablation --epochs 10 > data/ablation.log 2>&1; echo "ablation exit $?" >> data/pipeline.log) &
($R transfer --epochs 10 > data/transfer.log 2>&1; echo "transfer exit $?" >> data/pipeline.log) & wait
log "analyze"; $R analyze >> data/pipeline.log 2>&1 || { log "FAILED analyze"; exit 1; }
log "figures"; $R figures >> data/pipeline.log 2>&1 || { log "FAILED figures"; exit 1; }
log "PIPELINE DONE"
