#!/usr/bin/env bash
# Review-round compute, ONE job at a time (8 GB RAM: two jobs page to disk). Steps are idempotent.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CANPERT_SKIP_PRED_WILCOXON=1
R="uv run --no-sync canpert"
L=data/rev_lane.log
step() { echo "[$(date +%H:%M:%S)] START $*" >> $L; $R "$@" >> $L 2>&1; echo "[$(date +%H:%M:%S)] END rc=$? $*" >> $L; }
step t1 --seeds 1 --epochs 20
step t1 --seeds 2 --epochs 20
step t1 --seeds 0 --epochs 20 --variant meanctrl
step t1 --seeds 0 --epochs 20 --variant grouped
step ablation --epochs 20 --outdir ablation20 --only S1,S8,S32_noattn,S32_nofilm,S32_noprior,S32_pertid
step ablation --epochs 20 --outdir ablation20 --seed 1 --only S32_noattn,S32_nofilm,S32_noprior
echo "[$(date +%H:%M:%S)] LANE DONE" >> $L
