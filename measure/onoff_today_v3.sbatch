#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# onoff_today_v3.py: off-the-manifold shares against today's reference. Args: PREFIX (qgc_ or qgt_), then further script arguments (--measure-npz FILE). CPU only.
set -u
unset LD_LIBRARY_PATH PYTHONPATH
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); OO=$DUROOM_OUT/tests12/onoff
export LD_LIBRARY_PATH=$E/lib PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
date; hostname
cd $DUROOM_REPO/search && $E/bin/python -B $DUROOM_REPO/measure/onoff_today_v3.py $OO/out/onoff_today3_${1%_} --prefix "$@"
echo "rc=$?"; date
