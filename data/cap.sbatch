#!/bin/bash
#SBATCH --job-name=cap
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One cap or not: every work's angle to the average direction, in all three lenses.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/cap.py $DUROOM_OUT/cap
