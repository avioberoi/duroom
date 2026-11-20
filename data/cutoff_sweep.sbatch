#!/bin/bash
#SBATCH --job-name=cutoff_sweep
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1    # common.py sets these too; workers are the parallelism
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/cutoff_sweep.py $DUROOM_OUT/cutoff_sweep
