#!/bin/bash
#SBATCH --job-name=smooth_t2v
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# T2 in three lenses, then the verdict.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
OUT=$DUROOM_OUT/smooth
LD=$DUROOM_EMB
step() { echo "== $*"; $E/bin/python $SLURM_SUBMIT_DIR/smooth.py $OUT "$@" || echo "FAILED: $*"; }
step t2 real
step t2 real $LD/dinov3-l.npz
step t2 real $LD/siglip-so400m.npz
step verdict real
