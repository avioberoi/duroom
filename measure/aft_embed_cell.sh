#!/bin/bash
# Exact four-encoder readings (CLIP, DINOv3, SigLIP, AIMv2) of the pictures of one search run, checked against the run file and P's 8 known-answer pictures before anything is linked.
# Usage: bash $DUROOM_REPO/measure/aft_embed_cell.sh TAG   (such as qgt_ra_s11; env as in run/env.sh.example, a GPU, and the run's all_candidates.npz under $DUROOM_OUT/sphere/runs)
set -u
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONTENT_PHOTO SPHERE_KAT SPHERE_CLIP VARIATIONS_NORM VARIATIONS_GUIDANCE SPHERE_CODE SPHERE_ENV
TAG=${1:?usage: bash aft_embed_cell.sh TAG (such as qgt_ra_s11)}
B=$DUROOM_OUT; C=$DUROOM_REPO/search; M=$DUROOM_REPO/measure; O=$B/sphere; A=$B/tests12/afterrun
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")"); T3=$M/aft_tools_v4.py; ENC4="clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge"
ROWS=1921
RUNF=$O/runs/${TAG}_committee_euclidean_min_seedrow/all_candidates.npz; CD=$A/cells/${TAG}_h200; LD=$A/logs/${TAG}_h200
FX=$O/eval/${TAG}_emb_h200_v1.npz; FL=$O/lists/${TAG}_h200_v1.txt
mkdir -p $A/cells $A/logs $O/lists $O/eval
mkdir $LD || { echo "STOP: $LD exists"; exit 3; }
set -o noclobber
say() { echo "$(date '+%F %T') $*" | tee -a $LD/task.log; }
py() { LD_LIBRARY_PATH=$E/lib $E/bin/python -B "$@"; }
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
say "task start: $TAG exact file, host $(hostname), gpu '$GPU', task file $0"
for f in $FX $FL $CD; do [ ! -e $f ] || { say "STOP: $f exists; nothing written"; exit 3; }; done
mkdir $CD || { say "STOP: cannot make $CD"; exit 3; }
py $T3 list $RUNF $CD/$TAG.txt $ROWS 2>&1 | tee -a $LD/task.log
[ "$(wc -l < $CD/$TAG.txt 2>/dev/null)" = $ROWS ] || { say "STOP: run file does not hold $ROWS rows (or cannot be read). Nothing linked for $TAG."; exit 4; }
say "embed start (4 encoders, $ROWS pictures)"
bash $C/sphere.sbatch embed --list $CD/$TAG.txt --out $CD/${TAG}_emb_h200_v1.npz --encoders $ENC4 > $LD/embed.out 2> $LD/embed.err
rc=$?; nC=$(grep -c '^CLIP loaded' $LD/embed.out); nD=$(grep -c " done $ROWS images" $LD/embed.out)
say "embed end rc=$rc; card line '$(sed -n 2p $LD/embed.out)'; 'CLIP loaded' lines $nC; 'done $ROWS images' lines $nD"
[ $rc = 0 ] && [ $nC = 1 ] && [ $nD = 4 ] || { say "STOP: embedding step not as expected (rc, CLIP loader or encoder count). Not linked."; exit 4; }
py $T3 valid $CD/${TAG}_emb_h200_v1.npz $RUNF 2>&1 | tee -a $LD/task.log
grep -q '^exact file valid' $LD/task.log || { say "STOP: exact file not valid. Not linked."; exit 4; }
py $T3 katlist $CD/kat_list.txt >> $LD/task.log 2>&1
bash $C/sphere.sbatch embed --list $CD/kat_list.txt --out $CD/kat_emb.npz --encoders $ENC4 > $LD/kat.out 2> $LD/kat.err
py $T3 katcmp $CD/kat_emb.npz 2>&1 | tee -a $LD/task.log
grep -q '^known-answer PASS' $LD/task.log || { say "STOP: known-answer against P failed on this card. Not linked."; exit 4; }
ln $CD/$TAG.txt $FL && ln $CD/${TAG}_emb_h200_v1.npz $FX || { say "STOP: a link failed (name exists?)."; exit 5; }
md5sum $FX $FL $CD/kat_emb.npz > $CD/COMPLETE
say "complete: $TAG exact file $FX, list $FL"
exit 0
