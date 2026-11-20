#!/usr/bin/env bash
# Corpus embeddings in the three encoders: writes clip-vit-l-14.npz, dinov3-l.npz and siglip-so400m.npz (ids, embeddings) into $DUROOM_EMB.
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p $DUROOM_OUT/logs

for enc in clip-vit-l-14 dinov3-l siglip-so400m; do ENCODER=$enc bash $DUROOM_REPO/encode/run_encode.sbatch; done
