#!/usr/bin/env bash
# More corpus analyses: the cutoff sweep and turn test, raw-length check, data-figure numbers, the artist 2x2, distance types, encoder agreement, geometry of the corpus. Outputs go to $DUROOM_OUT/<script name>.
# Needs 01_encode.sh and 08_corpus_analysis.sh; clip_raw.sbatch needs one GPU, the rest is CPU only.
set -eo pipefail
source "$(dirname "$0")/env.sh"
export SLURM_SUBMIT_DIR=$DUROOM_REPO/data
cd $DUROOM_REPO/data
mkdir -p $DUROOM_OUT/logs

# Time gap at every cutoff 1800 to 2000 in equal 40-year windows against a no-time placebo, both directions: cutoff_sweep/cutoff_sweep.json.
bash $DUROOM_REPO/data/cutoff_sweep.sbatch
# Direction the record moves in at every cutoff, against a no-time placebo: turn/turn.json.
bash $DUROOM_REPO/data/turn.sbatch

# Raw (unscaled) CLIP vector of every work, on a GPU: reembed/wikiart_catfm.npz (keys ids, clip_raw; read by norm_effect.py and search/walk.py).
bash $DUROOM_REPO/data/clip_raw.sbatch
# Does scaling CLIP to length 1 change anything (length against year, neighbours raw vs unit, the time and artist gaps): norm_effect/norm_effect.json.
bash $DUROOM_REPO/data/norm_effect.sbatch

# Per-work numbers behind the data figures (sphere sample, time and East Asian groups, kept and missed, leftover histogram): figdata/figdata.json.
bash $DUROOM_REPO/data/figdata.sbatch
# CLIP on one public-domain print (data/great-wave.jpg): patch norms, end vector, angle to the average direction: clipviz/clipviz.json.
bash $DUROOM_REPO/data/clipviz.sbatch
# Typical works along the decade path, from newaxis_check of 08: figworks/figworks.json.
bash $DUROOM_REPO/data/figworks.sbatch

# Geometry and same-artist, style, genre and century shares of the 15 nearest works, in the three encoders: wikiart_geom/wikiart_geom.json.
bash $DUROOM_REPO/data/wikiart_geom.sbatch

# Artist 2x2 of the smoothness tests, on the works, a Gaussian and 64 clumps; needs smooth.sbatch of 08: smooth_ax/clip-vit-l-14/<cloud>_s0/ax.json.
bash $DUROOM_REPO/data/smooth_ax.sbatch

# Does the distance (chord, two Mahalanobis, two diffusion maps) change who a work's neighbours are: dist_types/dist_types.json.
bash $DUROOM_REPO/data/dist_types.sbatch
# Which works the three encoders agree are far, from the leftover and distance results of 08: agree/agree.json.
bash $DUROOM_REPO/data/agree.sbatch

# Read-only checks of the held-out and new-axis outputs of 08; prints to logs/check_bases_v2.out.
bash $DUROOM_REPO/data/check_bases_v2.sh > $DUROOM_OUT/logs/check_bases_v2.out
