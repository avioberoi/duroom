#!/bin/bash
#SBATCH --job-name=newaxis_more
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Checks of newaxis.py: default rerun and comparison, other lenses, equal windows after each cut, sample size, equal pasts; then the path nulls.
# Each step runs on its own, so one failure does not stop the rest.
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib
cd $SLURM_SUBMIT_DIR
O=$DUROOM_OUT
P=$E/bin/python
# Rerun the default at 1900 into newaxis_check and compare it with newaxis/
$P $SLURM_SUBMIT_DIR/newaxis.py $O/newaxis_check 1900 32 5 || echo FAILED check
$P - <<'PY' || echo FAILED check_compare
import json, os
import numpy as np
a = json.load(open(os.environ['DUROOM_OUT'] + '/newaxis_check/k32/newaxis.json'))['cutoffs']['1900']
b = json.load(open(os.environ['DUROOM_OUT'] + '/newaxis/k32/newaxis.json'))['cutoffs']['1900']
def nums(x):
    return [v for v in (nums(y) for y in (x.values() if isinstance(x, dict) else x)) for v in v] if isinstance(x, (dict, list, tuple)) else [x] if isinstance(x, (int, float)) else []
for k in ('summary', 'stability', 'null_stability', 'past_rank', 'by_decade_plane'):
    u, v = np.array(nums(a[k]), float), np.array(nums(b[k]), float)
    print('REGRESSION', k, 'same length' if len(u) == len(v) else 'LENGTH DIFFERS', 'max abs diff %.3g' % (np.abs(u - v).max() if len(u) == len(v) else np.nan), flush=True)
PY
for opts in "1900,1950,2000 32 5 lens=dinov3-l" "1900,1950,2000 32 5 lens=siglip-so400m" "1900,1950,2000 32 5 later_window=27" \
            "1900,1950,2000 32 5 later_window=27 n_later=6002" "1900,1950 32 5 n_later=6002" \
            "1900,1950,2000 32 5 past_window=100 later_window=27" "1900,1950,2000 32 5 past_window=100 later_window=27 n_later=6002"; do
  echo "== newaxis $opts"
  $P $SLURM_SUBMIT_DIR/newaxis.py $O/newaxis $opts || echo "FAILED newaxis $opts"
done
for L in clip-vit-l-14 dinov3-l siglip-so400m; do $P $SLURM_SUBMIT_DIR/path_null.py $O/path_null $L || echo "FAILED path_null $L"; done
echo done
