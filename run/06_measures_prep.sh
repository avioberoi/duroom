#!/usr/bin/env bash
# Inputs of the readouts: far kinds on stored vectors, baseline renders (random requests, round trips, noise pictures, region and gate frames), the renders of the random and noise requests, the oracle lines, the leftover basis, and the scales and guard cases of the place measure.
# Needs 03_reference.sh, 04_search.sh and 05_readings.sh; run it before 07_readouts.sh. Renders run in DUROOM_SEARCH_ENV, readings and scores in the environment of DUROOM_PY.
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
: ${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
B=$DUROOM_OUT; S=$DUROOM_REPO/search; M=$DUROOM_REPO/measure; O=$B/sphere; T=$B/tests12/t3; LO=$B/tests12/leftover
ENC4="clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge"
mkdir -p $T/out $LO/out $B/tests12/t1 $B/inputs $B/logs $O/baselines $O/eval/bound/noise_check $O/eval/probe $O/eval/regions

# Far kinds on stored vectors, DINOv3 and SigLIP side by side: tests12/t1/run_<id>_headline_dinov3 and _siglip (the folder name carries the Slurm job id or a timestamp).
bash $DUROOM_REPO/data/t1_head_v2.sbatch
for d in dinov3 siglip; do mv $B/tests12/t1/run_*_headline_$d $B/tests12/t1/run_headline_$d; done

# ---- baseline renders and readings (random requests, round trips)
# 256 random requests (192 in the pre-1900 cone, 64 on the sphere), painted: baselines/prior with manifest.npz and list.txt.
SPHERE_ENV=$DUROOM_SEARCH_ENV bash $S/sphere.sbatch py $S/baselines_render.py prior $O/baselines/prior
bash $S/sphere.sbatch embed --list $O/baselines/prior/list.txt --out $O/baselines/prior/emb.npz --encoders $ENC4
# Round trips (48 works x 7 steps, each step painted from the loop reader's reading of the previous picture) on the CLIP reader that matches the stored readings: baselines/chains_h200_qg.
SPHERE_CLIP=quickgelu SPHERE_ENV=$DUROOM_SEARCH_ENV bash $S/sphere.sbatch py $S/baselines_render.py chains $O/baselines/chains_h200_qg
bash $S/sphere.sbatch embed --list $O/baselines/chains_h200_qg/list.txt --out $O/baselines/chains_h200_qg/emb.npz --encoders $ENC4
# Scores of the random requests against P, with sensitivity columns: eval/sens_all.npz.
$DUROOM_PY -B $S/baselines_score.py $O/eval/sens_all.npz --baselines $O/baselines/prior

# ---- noise pictures
# 160 noise pictures (5 kinds x 32): eval/bound/noise_check/img and list.txt; their readings in CLIP, DINOv3 and SigLIP: noise_emb.npz; far scores against P: noise_check.out.
$DUROOM_PY -B $S/noise_make.py $O/eval/bound/noise_check/img $O/eval/bound/noise_check/list.txt
bash $S/sphere.sbatch embed --list $O/eval/bound/noise_check/list.txt --out $O/eval/bound/noise_check/noise_emb.npz --encoders clip-vit-l-14 dinov3-l siglip-so400m
$DUROOM_PY -B $S/noise_check.py > $O/eval/bound/noise_check/noise_check.out

# Gate on non-images: noise, flat and text pictures through the coherence gate and both observers: gate_negative/gate_negative.json.
bash $S/sphere.sbatch py $S/gate_negative.py $O/gate_negative

# ---- renders of the random and noise requests, painted again with their known-answer pictures and read in CLIP, DINOv3, SigLIP and the gate (tests12/t2/renders_v1/prior256_v1 and noise160_v1: emb.npz, manifest, expected_kat.txt).
bash $S/t2_render.sh prior256_v1 prior
bash $S/t2_render.sh noise160_v1 noise
# AIMv2 (with DINOv3 and SigLIP) readings of the random-request, noise-request and round-trip pictures: tests12/afterrun/acc/<set>_v2/emb3_v100_v2.npz.
bash $M/aft_acc_set.sh prior256_v1 $B/tests12/t2/renders_v1/prior256_v1/emb.npz
bash $M/aft_acc_set.sh noise160_v1 $B/tests12/t2/renders_v1/noise160_v1/emb.npz
bash $M/aft_acc_set.sh chains_h200_qg $O/baselines/chains_h200_qg/emb.npz

# ---- region and gate frames of the no-search families
# 868 region frames (midpoints, lines between region centres, 35 degree steps past a work): requests, then 4 render shards, their readings and scores (eval/regions/stage1/shard<i>, stage1.npz).
$DUROOM_PY -B $S/regions_frames.py $O/push_emb.npz $O/push2_emb.npz $O/pools/starts_pre1900.npz $O/eval/regions/regions_requests.npz
for i in 0 1 2 3; do
  SPHERE_ENV=$DUROOM_SEARCH_ENV bash $S/sphere.sbatch py $S/baselines_render.py vectors $O/eval/regions/stage1/shard$i --vectors $O/eval/regions/regions_requests.npz --shard $i --nshards 4
  bash $S/sphere.sbatch embed --list $O/eval/regions/stage1/shard$i/list.txt --out $O/eval/regions/stage1/shard$i/emb.npz --encoders $ENC4
done
$DUROOM_PY -B $S/baselines_score.py $O/eval/regions/stage1.npz --baselines $O/eval/regions/stage1/shard0 $O/eval/regions/stage1/shard1 $O/eval/regions/stage1/shard2 $O/eval/regions/stage1/shard3
# 1,056 gate frames (lines and record steps from the starts): requests, 8 render shards, readings and scores (eval/probe/stage1/shard<i>, stage1.npz).
$DUROOM_PY -B $S/probe_frames.py $O/eval/probe/probe_requests.npz
for i in 0 1 2 3 4 5 6 7; do
  SPHERE_ENV=$DUROOM_SEARCH_ENV bash $S/sphere.sbatch py $S/baselines_render.py vectors $O/eval/probe/stage1/shard$i --vectors $O/eval/probe/probe_requests.npz --shard $i --nshards 8
done
$DUROOM_PY -B $S/embed_dirs.py $O/eval/probe/stage1/shard0 $O/eval/probe/stage1/shard1 $O/eval/probe/stage1/shard2 $O/eval/probe/stage1/shard3 $O/eval/probe/stage1/shard4 $O/eval/probe/stage1/shard5 $O/eval/probe/stage1/shard6 $O/eval/probe/stage1/shard7
$DUROOM_PY -B $S/baselines_score.py $O/eval/probe/stage1.npz --baselines $O/eval/probe/stage1/shard0 $O/eval/probe/stage1/shard1 $O/eval/probe/stage1/shard2 $O/eval/probe/stage1/shard3 $O/eval/probe/stage1/shard4 $O/eval/probe/stage1/shard5 $O/eval/probe/stage1/shard6 $O/eval/probe/stage1/shard7

# ---- oracle lines
# Start vectors and the nearest far-and-faithful later works: inputs/oracle_targets.json and inputs/oracle_inputs.npz.
$DUROOM_PY -B $S/make_oracle_inputs.py
# Walk from each start toward its three targets at 10 to 70 degrees, painted and scored: sphere/oracle/oracle.json.
SPHERE_ENV=$DUROOM_SEARCH_ENV bash $S/sphere.sbatch py $S/oracle.py $B/inputs/oracle_inputs.npz $O/push_emb.npz $O/oracle

# Leftover basis (32 directions of P's readings) and null per encoder, plus leftover shares of the 15 main runs and of the later repaints: pre_v1_basis.npz, pre_v1_scores.npz, pre_v1.txt.
cd $DUROOM_REPO/measure && $DUROOM_PY -B lo_precompute_v1.py $LO/out/pre_v1

# Scales of the place measure (bars, radii, T's eligible rates): scales_v5.npz.
cd $T/out && $DUROOM_PY -B $DUROOM_REPO/measure/t3_measure_v5.py scales --out $T/out/scales_v5
# Known-answer and guard cases of the place measure: guards_v3.npz.
cd $T/out && $DUROOM_PY -B $DUROOM_REPO/measure/t3_measure_v5.py guards --out $T/out/guards_v3 --scales $T/out/scales_v5.npz --acc-aim-prior $B/tests12/afterrun/acc/prior256_v1_v2/emb3_v100_v2.npz --acc-aim-noise $B/tests12/afterrun/acc/noise160_v1_v2/emb3_v100_v2.npz
# Marker of the scales and guard files (md5 of each), checked by the measure: tests12/t3/FREEZE_v5.txt.
printf 'scales_md5=%s\nguards_md5=%s\n' "$(md5sum < $T/out/scales_v5.npz | cut -d' ' -f1)" "$(md5sum < $T/out/guards_v3.npz | cut -d' ' -f1)" > $T/FREEZE_v5.txt
