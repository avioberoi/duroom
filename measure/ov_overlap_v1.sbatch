#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ov_overlap_v1.py: later works reached from different starts, with the shuffle. CPU only.
set -u
unset LD_LIBRARY_PATH PYTHONPATH
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); A=$DUROOM_OUT/tests12/overlap
export LD_LIBRARY_PATH=$E/lib PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
date; hostname
cd $DUROOM_REPO/search && $E/bin/python -B $DUROOM_REPO/measure/ov_overlap_v1.py $A/out/ov_overlap_v1
echo "rc=$?"; date
