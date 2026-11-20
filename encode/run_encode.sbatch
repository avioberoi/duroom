#!/bin/bash
#SBATCH --job-name=encode_corpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --export=ENCODER=clip-vit-l-14 encode/run_encode.sbatch   (ENCODER: clip-vit-l-14, dinov3-l or siglip-so400m;
# set partition/account on the command line; python is DUROOM_PY)

module load python/miniforge-25.3.0 cuda/12.2 cudnn/9.4.0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $DUROOM_REPO/encode

ENCODER=${ENCODER:-clip-vit-l-14}
echo "=== $(date) === Encoding corpus with $ENCODER ==="
${DUROOM_PY:-python} encode_corpus.py --encoder $ENCODER
