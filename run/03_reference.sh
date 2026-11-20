#!/usr/bin/env bash
# Pools, the repainted reference P, the held-out set P2, the later works pos and today's reference T, each read in the four encoders with artist and request vectors attached; outputs in $DUROOM_OUT/sphere.
# Needs 01_encode.sh. neg_emb.py needs network access once (SigLIP tokenizer). Put any photo at $DUROOM_OUT/inputs/content_photo.png first (a logged similarity column only).
set -eo pipefail
source "$(dirname "$0")/env.sh"
B=$DUROOM_OUT; C=$DUROOM_REPO/search; O=$B/sphere
: ${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
mkdir -p $O/eval/today $B/logs $B/inputs
[ -f $B/inputs/content_photo.png ] || echo "WARNING: put an image at $B/inputs/content_photo.png (read by the content-similarity column)"

# Negative-prompt embedding of the SigLIP coherence gate: $B/inputs/neg_emb_siglip.npy, read by every search, render and embed step below.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash $C/sphere.sbatch py $C/neg_emb.py

# Pools of pre-1900 works (cap 3 per artist) and the start-work table: pre1900_cap3.npz, starts_pre1900.npz.
bash $C/sphere.sbatch pools
# P: 2,048 pre-1900 works repainted in 4 shards, then read in the four encoders plus the gate and content readers, with artist, request vector and own fidelity: push_emb.npz.
for i in 0 1 2 3; do SPHERE_ENV=$DUROOM_SEARCH_ENV bash $C/sphere.sbatch push --n 2048 --shard $i --nshards 4; done
# P2: 1,000 held-out pre-1900 works, disjoint from P: push2_emb.npz.
SPHERE_ENV=$DUROOM_SEARCH_ENV bash $C/sphere.sbatch push2 --n 1000
bash $C/sphere.sbatch embed_push push
bash $C/sphere.sbatch embed_push push2

# pos: 512 works from 1900 to 1969 (pool cap 3 per artist): pools/pos_1900_1969_cap3.npz, pos_emb.npz.
bash $C/sphere.sbatch windowpool $O/pools/pos_1900_1969_cap3.npz 1900 1970 3
SPHERE_ENV=$DUROOM_SEARCH_ENV bash $C/sphere.sbatch pushpool --pool $O/pools/pos_1900_1969_cap3.npz --out $O/pos --n 512 --rng 21
bash $C/sphere.sbatch embed_push pos

# Pool of works of every date, cap 3 per artist, and its start-work table: pools/pre2027_cap3.npz, pools/starts_pre2027.npz.
bash $C/sphere.sbatch pools 2027 3
# Today's pool without the pos, P2 and start works: pools/today_cap3.npz.
$DUROOM_PY $C/make_today_pool.py $O/pools $O
# T: 2,048 works of every date repainted in 3 shards, then read in the four encoders: push_today_emb.npz.
for i in 0 1 2; do SPHERE_ENV=$DUROOM_SEARCH_ENV bash $C/sphere.sbatch pushpool --pool $O/pools/today_cap3.npz --out $O/push_today --n 2048 --rng 11 --shard $i --nshards 3; done
bash $C/sphere.sbatch embed_push push_today

# Request-to-reading chord of every P picture on the loop's own CLIP reader: push_fidelity_loop_qg.npz (its 99th percentile, 0.7788, is the --fidelity-max of the search runs).
cd $C && SPHERE_ENV=$DUROOM_SEARCH_ENV SPHERE_CLIP=quickgelu bash $C/sphere.sbatch py $C/p_fidelity_loop.py $O/push_emb.npz $O/push_fidelity_loop_qg.npz
