#!/usr/bin/env bash
# The region map and the readouts that follow it: far in CLIP and in AIMv2, the gap between request and reading, lean baselines and where the lean grows, the free search's lean, the regions the loops climb through (also for the search scored in CLIP), two worked pictures, the manifold line set on other works.
# Needs 07_readouts.sh and 09_search_extras.sh (and the steps before them). Outputs go to $DUROOM_OUT/tests12/more.
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
B=$DUROOM_OUT; C=$DUROOM_REPO/measure; M=$B/tests12/more
mkdir -p $M

# Region map of the three spaces from the Test 1 runs of 06 and 09: tests12/geom/geom_<space>_v1.json (the CLIP one is read by r5_v4.py and lk_yard_v1.py).
$DUROOM_PY -B $DUROOM_REPO/data/geom_v1.py
# Far in CLIP on the 15 main runs: r3_clip_far_v1.txt, .npz (reads the exact CLIP files; the observers' far flags come from the run files).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r3_clip_far_v1.py $M/r3_clip_far_v1
# Far in AIMv2 on the 15 main runs: aimv2_far_v1.txt, .npz (reads the exact AIMv2 files; the observers' far flags come from the run files).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/aimv2_far_v1.py $M/aimv2_far_v1
# Gap between request and reading, and where the lean grows each round: r1_gap_v5.txt, .npz (centres and gaps from the runs' own CLIP readings, the lean of the pictures from the exact files).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r1_gap_v5.py $M/r1_gap_v5
# Lean baselines of the reference sets, real works, noise, no-search pictures and the loops, with the loops' request lean: r2_lean_v2.txt, .npz (exact readings).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r2_lean_v2.py $M/r2_lean_v2
# One round on the shared part (all 32 against the picked 4), far-and-faithful share by round, Spearman of lean with score: r1b_v1.txt, .npz (split from the runs' own readings, the rest from the exact readings).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r1b_v1.py $M/r1b_v1
# Lean of the free search, the free search told random scores and random directions at its angles: lk_lean_v1.txt (exact readings).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/lk_lean_v1.py $M/lk_lean_v1
# Counts behind the region yardstick: lk_yard_v1.txt (reference repaints read in CLIP).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/lk_yard_v1.py $M/lk_yard_v1
# Regions the loops' centres climb through, far-and-faithful share by region class, score by round, off-manifold pictures, free search and walk: r5_v4.txt, .npz (pictures from the exact CLIP readings, centres from the runs' own readings).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r5_v4.py $M/r5_v4
# Far scores and faithful chord of two search pictures (rounds 15 and 7 of qgc_ra_s10 and qgc_ra_s6), printed (observers' readings from the run files, faithful on the exact CLIP reading).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/two_pictures_v1.py > $M/two_pictures_v1.txt
# Path through the 32 regions, far-and-faithful share by region class and score by round of the CLIP-scored search and its random-score loop: r5b_v1.txt (needs the qgcclip_* runs of 04 and 05).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r5b_v1.py $M/r5b_v1
# The manifold line set on other works (all originals, the 1,512 repaints not used, halves and date parts of today's reference, other cuts) and the on/off and dated labels under each line: r6_v1.txt, .npz (reads onoff_v3_C, onoff_ff_v1 and onoff_dated_v2 of 07).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/r6_v1.py $M/r6_v1
