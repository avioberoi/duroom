#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=lo_pre_v1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Leftover basis, null and scores (lo_precompute_v1.py). CPU only.
set -u
unset LD_LIBRARY_PATH PYTHONPATH
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); LO=$DUROOM_OUT/tests12/leftover
export LD_LIBRARY_PATH=$E/lib PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $DUROOM_REPO/measure
$E/bin/python -c "import numpy; print('numpy', numpy.__version__)"
$E/bin/python -B lo_precompute_v1.py $LO/out/pre_v1
echo "precompute rc=$?"
