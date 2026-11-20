#!/bin/bash
#SBATCH --job-name=t1_head
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# DINOv3 and SigLIP in parallel, 24 workers each
unset LD_LIBRARY_PATH; E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
T=$DUROOM_OUT/tests12/t1
hostname; date
SLURM_CPUS_PER_TASK=24 LD_LIBRARY_PATH=$E/lib $E/bin/python $DUROOM_REPO/data/t1_v2.py --out $T --tag headline_dinov3 --observers dinov3-l > $T/t1_head_v2_${SLURM_JOB_ID}_dinov3.log 2>&1 &
SLURM_CPUS_PER_TASK=24 LD_LIBRARY_PATH=$E/lib $E/bin/python $DUROOM_REPO/data/t1_v2.py --out $T --tag headline_siglip --observers siglip-so400m > $T/t1_head_v2_${SLURM_JOB_ID}_siglip.log 2>&1 &
wait
date; echo done
