#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# onoff_ff_v1.py: nearest-work split on the far-and-faithful population. CPU only.
set -u
unset LD_LIBRARY_PATH PYTHONPATH
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); OO=$DUROOM_OUT/tests12/onoff
export LD_LIBRARY_PATH=$E/lib PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date; hostname
cd $DUROOM_REPO/search && $E/bin/python -B $DUROOM_REPO/measure/onoff_ff_v1.py $OO/out/onoff_ff_v1
echo "rc=$?"; date
