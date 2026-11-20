#!/bin/bash
#SBATCH --job-name=smooth
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Smoothness tests: the data, then every null, then the verdict. Argument: LENS file (empty = CLIP).
# A failed step is logged and the job goes on; the verdict marks rules with missing inputs as pending.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
OUT=$DUROOM_OUT/smooth
LD=$DUROOM_EMB
L="${1:-}"
step() { echo "== $*"; $E/bin/python $SLURM_SUBMIT_DIR/smooth.py $OUT "$@" || echo "FAILED: $*"; }
step t4 real "$L"
step calib real "$L"
if [ -z "$L" ]; then
  step t1,t2,t3 real
  step t2 real $LD/dinov3-l.npz
  step t2 real $LD/siglip-so400m.npz
else
  step t1,t3 real "$L"
fi
for s in 0 1; do step t1,t3,t4 gauss "$L" $s; done
step t1,t3,t4 mix:64 "$L"
step t1,t3,t4 mix:16 "$L"
for s in 0 1; do step t4 sheet:ctrl:0 "$L" $s; done
for g in 0.25 0.5 0.65; do step t1,t4 sheet:ctrl:$g "$L"; done
step t3 sheet:ctrl:0.5 "$L"
step t3 sheet:ctrl:eq "$L"
for f in 4 2 1; do step t1,t3 curved:ctrl:$f "$L"; done
step t4 lumpy:ctrl "$L"
step t4 union:ctrl "$L"
step verdict real "$L"
