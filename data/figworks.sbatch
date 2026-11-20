#!/bin/bash
#SBATCH --job-name=figworks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
cd $SLURM_SUBMIT_DIR
O=$DUROOM_OUT
$E/bin/python $SLURM_SUBMIT_DIR/figworks.py $O/newaxis_check/k32/loads_1900.npz $O/newaxis_check/k32/newaxis.json $O/figworks 3
