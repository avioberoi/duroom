#!/bin/bash
#SBATCH --job-name=newaxis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/newaxis.py $DUROOM_OUT/newaxis "$@"
