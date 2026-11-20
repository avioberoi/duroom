#!/bin/bash
# Safety flags (evaluator's safety(), CPU, search environment) and a DINOv3 / SigLIP / AIMv2 backup reading, never linked as the exact file, for the pictures of one search run.
# Usage: bash $DUROOM_REPO/measure/aft_safety_cell.sh TAG   (such as qgt_ra_s11; env as in run/env.sh.example plus DUROOM_SEARCH_ENV, a GPU for the backup reading, and the run's all_candidates.npz under $DUROOM_OUT/sphere/runs)
set -u
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONTENT_PHOTO SPHERE_KAT SPHERE_CLIP VARIATIONS_NORM VARIATIONS_GUIDANCE SPHERE_CODE SPHERE_ENV
TAG=${1:?usage: bash aft_safety_cell.sh TAG (such as qgt_ra_s11)}
B=$DUROOM_OUT; C=$DUROOM_REPO/search; M=$DUROOM_REPO/measure; O=$B/sphere; A=$B/tests12/afterrun
LD=$A/logs/${TAG}_v100
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); E2=${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
T3=$M/aft_tools_v4.py; ENC4="clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge"; ENC3="dinov3-l siglip-so400m aimv2-huge"
say() { echo "$(date '+%F %T') $*" | tee -a $LD/task.log; }
py() { LD_LIBRARY_PATH=$E/lib $E/bin/python -B "$@"; }
ROWS=1921
RUNF=$O/runs/${TAG}_committee_euclidean_min_seedrow/all_candidates.npz; CD=$A/cells/${TAG}_v100; FS=$O/eval/bound/ev_safety_qgt_$TAG.json
mkdir -p $A/cells $A/logs $O/eval/bound
mkdir $LD || { echo "STOP: $LD exists"; exit 3; }
set -o noclobber
say "task start: $TAG safety, host $(hostname), gpu '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)', task file $0"
for f in $FS $CD; do [ ! -e $f ] || { say "STOP: $f exists; nothing written"; exit 3; }; done
mkdir $CD || { say "STOP: cannot make $CD"; exit 3; }
py $T3 list $RUNF $CD/$TAG.txt $ROWS 2>&1 | tee -a $LD/task.log
[ "$(wc -l < $CD/$TAG.txt 2>/dev/null)" = $ROWS ] || { say "STOP: run file does not hold $ROWS rows (or cannot be read). Nothing linked for $TAG."; exit 4; }
say "safety start ($ROWS pictures, CPU)"
( cd $M && LD_LIBRARY_PATH=$E2/lib PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $E2/bin/python -B $T3 safety $CD/$TAG.txt $CD/${TAG}_safety.json $ROWS 2>&1 > $LD/safety.out | grep -v 'Potential NSFW content was detected' > $LD/safety.err; exit ${PIPESTATUS[0]} )
rc=$?   # the checker's per-batch "Potential NSFW content" warnings are dropped: their number would be a count from the run
cat $LD/safety.out >> $LD/task.log
say "safety end rc=$rc"
[ $rc = 0 ] && grep -q '^safety pass done' $LD/safety.out || { say "STOP: safety pass failed. Not linked."; exit 4; }
ln $CD/${TAG}_safety.json $FS || { say "STOP: link failed (name exists?)."; exit 5; }
say "safety complete: $FS"
say "backup embed start (DINOv3, SigLIP, AIMv2; not the exact file)"
bash $C/sphere.sbatch embed --list $CD/$TAG.txt --out $CD/${TAG}_emb3_v100.npz --encoders $ENC3 > $LD/embed3.out 2> $LD/embed3.err
say "backup embed rc=$?; card line '$(sed -n 2p $LD/embed3.out)'; 'done $ROWS images' lines $(grep -c " done $ROWS images" $LD/embed3.out)"
py $T3 valid $CD/${TAG}_emb3_v100.npz $RUNF dinov3-l,siglip-so400m,aimv2-huge 2>&1 | tee -a $LD/task.log
say "task end"
exit 0
