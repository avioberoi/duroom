#!/bin/bash
# Embeds a stored small picture set with DINOv3, SigLIP and AIMv2 (no CLIP) and compares row by row with the stored file (max chord, tolerance 2.5e-4).
# Usage: bash $DUROOM_REPO/measure/aft_acc_set.sh SET STORED_EMB_NPZ   (SET: prior256_v1, noise160_v1 or chains_h200_qg; env as in run/env.sh.example)
set -u
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONTENT_PHOTO SPHERE_KAT SPHERE_CLIP VARIATIONS_NORM VARIATIONS_GUIDANCE SPHERE_CODE SPHERE_ENV
SET=${1:?usage: bash aft_acc_set.sh SET STORED_EMB_NPZ (SET: prior256_v1, noise160_v1 or chains_h200_qg)}
ST=${2:?usage: bash aft_acc_set.sh SET STORED_EMB_NPZ}
case $SET in
  prior256_v1|noise160_v1|chains_h200_qg) ;;
  *) echo "unknown SET $SET"; exit 2 ;;
esac
B=$DUROOM_OUT; C=$DUROOM_REPO/search; M=$DUROOM_REPO/measure; O=$B/sphere; A=$B/tests12/afterrun
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
D=$A/acc/${SET}_v2; LD=$A/logs/acc_${SET}_v2; T2=$M/aft_tools_v4.py
mkdir -p $A/acc $A/logs; mkdir $LD && mkdir $D || { echo "STOP: $LD or $D exists"; exit 3; }
set -o noclobber
say() { echo "$(date '+%F %T') $*" | tee -a $LD/task.log; }
py() { LD_LIBRARY_PATH=$E/lib $E/bin/python -B "$@"; }
say "task start: $SET, host $(hostname), gpu '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)', task file $0"
py $T2 plist $ST $D/list.txt 2>&1 | tee -a $LD/task.log
N=$(wc -l < $D/list.txt)
bash $C/sphere.sbatch embed --list $D/list.txt --out $D/emb3_v100_v2.npz --encoders dinov3-l siglip-so400m aimv2-huge > $LD/embed.out 2> $LD/embed.err
rc=$?
say "embed end rc=$rc; card line '$(sed -n 2p $LD/embed.out)'; 'done $N images' lines $(grep -c " done $N images" $LD/embed.out)"
py $T2 cmpc $D/emb3_v100_v2.npz $ST 2>&1 | tee -a $LD/task.log
say "task end rc=$rc"
exit $rc
