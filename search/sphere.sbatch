#!/bin/bash
# Sphere search tasks. Partition, account, time and gres come from
# sbatch flags. Tasks run in the environment SPHERE_ENV (default: the environment of DUROOM_PY).
#SBATCH --job-name=sphere
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
B=$DUROOM_OUT
C=$DUROOM_REPO/search
O=$B/sphere
EDA=$DUROOM_REPO/data           # data.py and basis.py
TASK=$1; shift
ENV=${SPHERE_ENV:-$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")}
module load cuda/12.2 cudnn/9.4.0 2>/dev/null || true
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_HOME=$B/cache/torch MPLCONFIGDIR=$B/cache/mpl HOME=$B/cache/home
export SPHERE_CODE=$C
PY=$ENV/bin/python
mkdir -p $O $O/lists $B/cache/home $B/cache/torch $B/cache/mpl
cd $B
echo "=== $(date) task=$TASK args=$* host=$(hostname) env=$ENV code=$C"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
case $TASK in
  envcheck)   $PY -c "import importlib; [print(m, bool(importlib.util.find_spec(m))) for m in ('diffusers','timm','open_clip','faiss','clip','cma','pandas','yaml')]" ;;
  pools)      cd $EDA && PYTHONPATH=$EDA $PY $C/make_pools.py $O/pools "$@" ;;
  windowpool) cd $EDA && PYTHONPATH=$EDA $PY $C/make_window_pool.py "$@" ;;
  push)       $PY $C/pushforward_sphere.py --pool $O/pools/pre1900_cap3.npz --out $O/push "$@" ;;
  pushpool)   $PY $C/pushforward_sphere.py "$@" ;;
  push2)      $PY $C/pushforward_sphere.py --pool $O/pools/pre1900_cap3.npz --out $O/push2 --exclude $O/push --rng 12 "$@" ;;
  embed_push) D=${1:-push}; $PY -c "import glob, numpy as np; open('$O/lists/$D.txt','w').write('\\n'.join(p for f in sorted(glob.glob('$O/$D/shard[0-9].npz')) for p in np.load(f, allow_pickle=True)['paths'])+'\\n')" && $PY $DUROOM_REPO/encode/embed_images.py --list $O/lists/$D.txt --out $O/${D}_emb.npz --encoders clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge coherence content && $PY $C/attach_artist.py $O/${D}_emb.npz $O/$D ;;
  search)     cd $C && $PY search_v2.py --config $C/config.yaml --out-root $O/runs "$@" ;;
  embed)      $PY $DUROOM_REPO/encode/embed_images.py "$@" ;;
  py)         $PY "$@" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== $(date) done task=$TASK"
