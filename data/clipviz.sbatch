#!/bin/bash
#SBATCH --job-name=clipviz
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/clipviz.py $SLURM_SUBMIT_DIR/great-wave.jpg $DUROOM_OUT/clipviz
