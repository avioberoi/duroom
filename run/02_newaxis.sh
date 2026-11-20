#!/usr/bin/env bash
# New axis at cutoffs 1900, 1950, 2000 with 32 directions on the stored corpus embeddings (CPU only): writes newaxis/k32/newaxis.json.
# The evaluator and the readouts read this file (the new-axis shift direction at 1900), so it runs before them; needs 01_encode.sh only.
set -eo pipefail
source "$(dirname "$0")/env.sh"
export SLURM_SUBMIT_DIR=$DUROOM_REPO/data
cd $DUROOM_REPO/data
mkdir -p $DUROOM_OUT/logs

bash $DUROOM_REPO/data/newaxis.sbatch
