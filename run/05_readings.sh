#!/usr/bin/env bash
# Exact re-readings of every run's pictures: the saved PNGs are read again in CLIP, DINOv3, SigLIP and AIMv2.
# Needs 04_search.sh. Writes $O/lists/<tag>.txt (picture paths) and the exact-reading file $O/eval/<tag>_emb.npz (the main runs) or <tag>_emb_h200_v1.npz (against today's reference, and the walk).
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
B=$DUROOM_OUT; C=$DUROOM_REPO/search; O=$B/sphere
mkdir -p $O/lists $O/eval

# The main runs and the CLIP-scored search with its random-score loop (starts 6 to 10): picture list and four-encoder readings.
for tag in qgc_ra_s{6..10} qgc_pin_s{6..10} qgc_c2_s{6..10} qgcclip_ra_s{6..10} qgcclip_pin_s{6..10}; do
  $DUROOM_PY -c "import numpy as np; z=np.load('$O/runs/${tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True); open('$O/lists/${tag}.txt','w').write('\n'.join(map(str, z['paths']))+'\n')"
  bash $C/sphere.sbatch embed --list $O/lists/${tag}.txt --out $O/eval/${tag}_emb.npz --encoders clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge
done
# Runs against today's reference (starts 11 to 15): list, readings, check of the 8 known-answer pictures of P and completion marker (afterrun/cells/<tag>_h200/COMPLETE).
for tag in qgt_ra_s{11..15} qgt_pin_s{11..15} qgt_c2_s{11..15}; do bash $DUROOM_REPO/measure/aft_embed_cell.sh $tag; done
# Walk runs: list and readings.
for tag in qgw_walk_s{6..10}; do
  $DUROOM_PY -c "import numpy as np; z=np.load('$O/runs/${tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True); open('$O/lists/${tag}.txt','w').write('\n'.join(map(str, z['paths']))+'\n')"
  bash $C/sphere.sbatch embed --list $O/lists/${tag}.txt --out $O/eval/${tag}_emb_h200_v1.npz --encoders clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge
done

# Weights of the NSFW checker the evaluator and the safety cells load from $DUROOM_OUT/cache/safety_checker (Hugging Face id CompVis/stable-diffusion-safety-checker).
$DUROOM_PY -c "from huggingface_hub import snapshot_download; snapshot_download('CompVis/stable-diffusion-safety-checker', local_dir='$DUROOM_OUT/cache/safety_checker')"
# Safety flags of every picture of the runs against today's reference (ev_safety_qgt_<tag>.json), with a DINOv3/SigLIP/AIMv2 backup reading.
for tag in qgt_ra_s{11..15} qgt_pin_s{11..15} qgt_c2_s{11..15}; do bash $DUROOM_REPO/measure/aft_safety_cell.sh $tag; done
