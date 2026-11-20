#!/bin/bash
#SBATCH --job-name=dist_types
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK    # one process, the 132k x 132k products use the cores
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/dist_types.py $DUROOM_OUT/dist_types "$@"
