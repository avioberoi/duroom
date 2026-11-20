#!/bin/bash
#SBATCH --job-name=agree
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
cd $SLURM_SUBMIT_DIR
$E/bin/python $SLURM_SUBMIT_DIR/agree.py $DUROOM_OUT/agree $DUROOM_OUT/leftover $DUROOM_OUT/distance "$@"
