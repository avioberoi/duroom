#!/bin/bash
#SBATCH --job-name=moves
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# common.py on the time groups, half the artists, the clusters and four styles and genres, in all three lenses; output goes to common_moves.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1    # one BLAS thread per worker process
cd $SLURM_SUBMIT_DIR
G="artist,after_1900,before_1900,after_1900_no_shared_artists,cluster,style:Ukiyo-e,genre:religious painting,style:Rococo,genre:abstract"
for L in "" $DUROOM_EMB/dinov3-l.npz $DUROOM_EMB/siglip-so400m.npz; do
  $E/bin/python $SLURM_SUBMIT_DIR/common.py $DUROOM_OUT/common_moves 5 "$G" "$L" || echo "FAILED lens [$L]"
done
