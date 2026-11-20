#!/bin/bash
#SBATCH --job-name=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1    # one BLAS thread per worker process
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/common.py $DUROOM_OUT/common "$@"
