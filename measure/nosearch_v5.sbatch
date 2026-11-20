#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=nosearch_v5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# No-search families: far score, rank, gap and geometry per picture. CPU only.
unset LD_LIBRARY_PATH
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); A=$DUROOM_OUT/tests12/nosearch
export LD_LIBRARY_PATH=$E/lib PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $DUROOM_REPO/measure && $E/bin/python -B $DUROOM_REPO/measure/nosearch_v5.py $A/nosearch_v5
