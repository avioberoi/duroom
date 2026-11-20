#!/bin/bash
#SBATCH --job-name=parts
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/parts.py $DUROOM_OUT/parts "$@"
