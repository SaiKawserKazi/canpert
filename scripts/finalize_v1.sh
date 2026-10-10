#!/usr/bin/env bash
# Runs after the review compute lane: noise-free variants, calibration, analysis, figures, paper, response.
cd "$(dirname "$0")/.."
L=data/finalize_v1.log
log() { echo "[$(date +%H:%M:%S)] $*" >> $L; }
until grep -q "LANE DONE" data/rev_lane.log; do sleep 120; done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R="uv run --no-sync"
log "clean variants"
$R python -c "
from pathlib import Path
from canpert.calib import add_clean_variants
for t in ('t1', 't1_meanctrl', 't1_grouped', 't2', 'transfer'):
    add_clean_variants(Path('artifacts'), Path('data/processed'), t)
" >> $L 2>&1 || { log "FAILED clean"; exit 1; }
log "calibration (full rerun incl. noise-free variants)"
rm -f reports/calibration/calib_*.parquet reports/calibration/reliability_*.parquet
$R python -c "
from pathlib import Path
from canpert.calib import calibrate_task
for t in ('t1', 't1_meanctrl', 't1_grouped', 't2'):
    calibrate_task(Path('artifacts'), Path('data/processed'), t, Path('reports/calibration'))
" >> $L 2>&1 || { log "FAILED calibration"; exit 1; }
log "analyze"; $R canpert analyze >> $L 2>&1 || { log "FAILED analyze"; exit 1; }
log "figures"; $R canpert figures >> $L 2>&1 || { log "FAILED figures"; exit 1; }
log "paper"; rm -rf reports/_eq; $R python -c "
from pathlib import Path
from canpert.paper import build_paper
build_paper(Path('reports'), pdf=True)
from canpert.response_letter import build
build(Path('reports'))
" >> $L 2>&1 || { log "FAILED paper"; exit 1; }
log "FINALIZE DONE"
