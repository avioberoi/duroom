#!/bin/bash
#SBATCH --job-name=smooth_ax
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# The artist 2x2: stage ax of smooth_ax.py on the works, the Gaussian and the 64 clumps, CLIP only. Needs the stored calibration and the works' R of smooth.sbatch; it works on copies in its own root, so nothing under smooth/ is touched.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
SRC=$DUROOM_OUT/smooth
OUT=$DUROOM_OUT/smooth_ax
LENSDIR=clip-vit-l-14    # the results sit under <root>/<lens file stem>
[ -f $SRC/$LENSDIR/calib.json ] && [ -f $SRC/$LENSDIR/real_s0/smooth.json ] || { echo "FAILED: stored calibration not found under $SRC/$LENSDIR"; exit 1; }
mkdir -p $OUT/$LENSDIR/real_s0
cp -n $SRC/$LENSDIR/calib.json $OUT/$LENSDIR/calib.json
cp -n $SRC/$LENSDIR/real_s0/smooth.json $OUT/$LENSDIR/real_s0/smooth.json
step() { echo "== $*"; $E/bin/python $SLURM_SUBMIT_DIR/smooth_ax.py $OUT "$@" || echo "FAILED: $*"; }
step ax real
step ax gauss
step ax mix:64
