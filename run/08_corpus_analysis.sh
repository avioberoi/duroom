#!/usr/bin/env bash
# Runs on the stored corpus embeddings (CPU only): counts, cone, held-out group tests, smoothness tests, new-axis variants, leftover, distance, corner, lumps, retrodiction, definitions of "unexplored".
# Needs 01_encode.sh and 02_newaxis.sh; the smoothness tests also need the search and readout results (inputs/smooth_ids.csv) and the unexplored-definitions run needs the noise readings of 06_measures_prep.sh. Outputs go to $DUROOM_OUT/<script name>.
set -eo pipefail
source "$(dirname "$0")/env.sh"
export SLURM_SUBMIT_DIR=$DUROOM_REPO/data
cd $DUROOM_REPO/data
mkdir -p $DUROOM_OUT/logs $DUROOM_OUT/counts $DUROOM_OUT/unexp $DUROOM_OUT/inputs

# Rows, kept, dropped, named dated works, artists, styles, genres, works per decade: counts.json.
$DUROOM_PY $DUROOM_REPO/data/counts.py $DUROOM_OUT/counts/counts.json

# Angle of every work to the average direction in the three encoders: cap/cap.json.
bash $DUROOM_REPO/data/cap.sbatch

# Held-out group test, CLIP, all groups, 5 seeds: common/clip-vit-l-14/heldout.json.
bash $DUROOM_REPO/data/common.sbatch
# Held-out group test, DINOv3 (arguments SEEDS GROUPS LENS_FILE): common/dinov3-l/heldout.json.
bash $DUROOM_REPO/data/common.sbatch 5 all $DUROOM_EMB/dinov3-l.npz
# Held-out group test, SigLIP: common/siglip-so400m/heldout.json.
bash $DUROOM_REPO/data/common.sbatch 5 all $DUROOM_EMB/siglip-so400m.npz
# Move or reordering: the group test on time groups, half the artists, clusters and four styles and genres, in the three encoders: common_moves/<encoder>/heldout.json.
bash $DUROOM_REPO/data/moves.sbatch

# Smoothness tests (shares held, scaled overlap, local dimension, calibration, controls, verdict), CLIP. Needs inputs/smooth_ids.csv: the 26 later works that passed, 2 unfaithful, 484 non-passers, 3 starts, 9 oracle pairs; made from the later-works readings and the oracle targets.
$DUROOM_PY $DUROOM_REPO/data/make_ids.py $DUROOM_OUT/inputs/smooth_ids.csv
bash $DUROOM_REPO/data/smooth.sbatch
# Grid of flat sheets with noise, then the verdict again.
bash $DUROOM_REPO/data/smooth_grid.sbatch
# Sparse-places test in three encoders, then the verdict.
bash $DUROOM_REPO/data/smooth_t2v.sbatch

# New axis: rerun of the default compared with newaxis/k32/newaxis.json, per-encoder variants, window and sample variants, path nulls in the three encoders.
bash $DUROOM_REPO/data/newaxis_more.sbatch
# New axis with 64 and 128 directions: newaxis/k64, newaxis/k128.
bash $DUROOM_REPO/data/newaxis.sbatch 1900,1950,2000 64
bash $DUROOM_REPO/data/newaxis.sbatch 1900,1950,2000 128

# Leftover share per work, whole-corpus basis, CLIP: leftover/clip-vit-l-14_k32_cap0.
bash $DUROOM_REPO/data/leftover.sbatch
# Leftover share with DINOv3, with SigLIP, and with at most one work per artist in CLIP: leftover/dinov3-l_k32_cap0, siglip-so400m_k32_cap0, clip-vit-l-14_k32_cap1.
bash $DUROOM_REPO/data/leftover.sbatch $DUROOM_EMB/dinov3-l.npz 32 0
bash $DUROOM_REPO/data/leftover.sbatch $DUROOM_EMB/siglip-so400m.npz 32 0
bash $DUROOM_REPO/data/leftover.sbatch "" 32 1

# Three far scores per work, CLIP: distance/clip-vit-l-14.
bash $DUROOM_REPO/data/distance.sbatch
# Three far scores per work, DINOv3 and SigLIP: distance/dinov3-l, distance/siglip-so400m.
bash $DUROOM_REPO/data/distance.sbatch $DUROOM_EMB/dinov3-l.npz
bash $DUROOM_REPO/data/distance.sbatch $DUROOM_EMB/siglip-so400m.npz

# Straight walk against the decade path from the pre-1900 centre to the 1960 to 1979 centre: corner/corner.json.
bash $DUROOM_REPO/data/corner.sbatch

# Lumps of directions from the average, CLIP: parts/clip-vit-l-14.
bash $DUROOM_REPO/data/parts.sbatch
# Lumps with seeds 7 and 8, and against a Gaussian null: parts/clip-vit-l-14_gauss.
bash $DUROOM_REPO/data/parts.sbatch "" 7,8
bash $DUROOM_REPO/data/parts.sbatch "" "" gauss

# Retrodiction of later works against the works before the cutoff, cutoffs 1800, 1850, 1900 and 1950: retro/clip-vit-l-14_cut<N>.
for cut in 1800 1850 1900 1950; do bash $DUROOM_REPO/data/retro.sbatch $cut; done

# Definitions of "unexplored" on stored vectors, with the cap sweep (mode full; prints the numbers): unexp/unexp_full.out. Reads the noise readings of 06_measures_prep.sh.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $DUROOM_PY -B $DUROOM_REPO/data/unexp.py full > $DUROOM_OUT/unexp/unexp_full.out
