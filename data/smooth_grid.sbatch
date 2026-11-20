#!/bin/bash
#SBATCH --job-name=smooth_grid
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# T1's d x sigma grid of flat sheets with noise, then the verdict again (it merges). Submit once calib.json exists. Argument: LENS file (empty = CLIP).
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
OUT=$DUROOM_OUT/smooth
$E/bin/python $SLURM_SUBMIT_DIR/smooth.py $OUT grid real "${1:-}"
$E/bin/python $SLURM_SUBMIT_DIR/smooth.py $OUT verdict real "${1:-}"
