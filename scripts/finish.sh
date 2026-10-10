#!/usr/bin/env bash
# Run the transfer matrix once T2 is done, then analysis + figures after all experiments.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R="uv run --no-sync canpert"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a data/pipeline.log; }
until [ -f artifacts/t2/metrics_rpe1_s0.parquet ]; do sleep 60; done
log "transfer (after t2)"; $R transfer --epochs 10 > data/transfer.log 2>&1; log "transfer exit $?"
until [ -f artifacts/ablation/metrics_S32_pertid.parquet ]; do sleep 60; done
log "analyze (final)"; $R analyze >> data/pipeline.log 2>&1 && log "analyze ok" || log "FAILED analyze"
log "figures (final)"; $R figures >> data/pipeline.log 2>&1 && log "figures ok" || log "FAILED figures"
log "FINISH DONE"
