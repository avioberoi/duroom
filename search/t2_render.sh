#!/bin/bash
# Paints requests with the search's decoder, embeds the pictures and checks them. Usage: bash t2_render.sh NAME MODE   (prior256_v1 prior: 256 random requests; noise160_v1 noise: the 160 noise images' stored CLIP readings)
# Needs DUROOM_OUT, DUROOM_REPO, DUROOM_SEARCH_ENV and the variables of sphere.sbatch (python for prep/post: DUROOM_PY or python); writes to a new $DUROOM_OUT/tests12/t2/renders_v1/NAME, and renders 8 known-answer P works first.
set -u
unset LD_LIBRARY_PATH SPHERE_ENV PYTHONPATH PYTHONHOME
[ $# = 2 ] || { echo "usage: bash t2_render.sh NAME MODE"; exit 2; }
NAME=$1; MODE=$2
B=${DUROOM_OUT:?set DUROOM_OUT}; R=${DUROOM_REPO:?set DUROOM_REPO}; C=$R/search
NS=${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
SG=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
OUT=$B/tests12/t2/renders_v1/$NAME
say() { echo "$(date '+%F %T') $*"; }
mkdir -p $(dirname $OUT)
mkdir $OUT || { say "STOP: cannot make a new folder $OUT (it exists?)"; exit 3; }
mkdir $OUT/code $OUT/inputs $OUT/logs || exit 3
set -o noclobber
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
say "task $NAME start: host $(hostname), gpu '$GPU', out $OUT"
cp $C/t2_prep.py $OUT/code/prep.py
cp $C/t2_post.py $OUT/code/post.py
say "prep"
LD_LIBRARY_PATH=$SG/lib PYTHONDONTWRITEBYTECODE=1 $SG/bin/python -B $OUT/code/prep.py $OUT $MODE  > $OUT/logs/prep.out 2> $OUT/logs/prep.err \
  || { say "STOP: prep failed"; tail -20 $OUT/logs/prep.err; exit 4; }
cat $OUT/logs/prep.out
say "known-answer renders (8)"
SPHERE_ENV=$NS bash $C/sphere.sbatch pushpool --pool $OUT/inputs/kat_pool.npz --out $OUT/kat --n 8 --rng 11 > $OUT/logs/kat_render.out 2> $OUT/logs/kat_render.err \
  || { say "STOP: known-answer render failed"; tail -20 $OUT/logs/kat_render.err; exit 5; }
say "main renders"
if [ "$MODE" = prior ]; then
  SPHERE_ENV=$NS bash $C/sphere.sbatch py $C/baselines_render.py prior $OUT/prior > $OUT/logs/render.out 2> $OUT/logs/render.err \
    || { say "STOP: render failed"; tail -20 $OUT/logs/render.err; exit 5; }
else
  SPHERE_ENV=$NS bash $C/sphere.sbatch py $C/baselines_render.py vectors $OUT/noise --vectors $OUT/inputs/noise_req.npz > $OUT/logs/render.out 2> $OUT/logs/render.err \
    || { say "STOP: render failed"; tail -20 $OUT/logs/render.err; exit 5; }
fi
cat $OUT/inputs/expected_kat.txt $OUT/inputs/expected_main.txt > $OUT/inputs/embed_list.txt
miss=0; while read -r p; do [ -e "$p" ] || miss=$((miss+1)); done < $OUT/inputs/embed_list.txt
say "renders listed $(wc -l < $OUT/inputs/embed_list.txt), missing $miss"
[ $miss = 0 ] || { say "STOP: renders missing"; exit 6; }
say "embed (CLIP, DINOv3, SigLIP, gate)"
bash $C/sphere.sbatch embed --list $OUT/inputs/embed_list.txt --out $OUT/emb.npz --encoders clip-vit-l-14 dinov3-l siglip-so400m coherence > $OUT/logs/embed.out 2> $OUT/logs/embed.err \
  || { say "STOP: embed failed"; tail -20 $OUT/logs/embed.err; exit 7; }
say "checks"
LD_LIBRARY_PATH=$SG/lib PYTHONDONTWRITEBYTECODE=1 $SG/bin/python -B $OUT/code/post.py $OUT $MODE > $OUT/check.txt 2> $OUT/logs/post.err; rc=$?
cat $OUT/check.txt; [ -s $OUT/logs/post.err ] && tail -5 $OUT/logs/post.err
say "task $NAME end: PNGs $(find $OUT -name '*.png' | wc -l), check rc $rc"
exit $rc
