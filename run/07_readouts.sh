#!/usr/bin/env bash
# The evaluator and the readouts, in dependency order: evaluator table of the main runs, on/off tables, far-and-faithful table, dated split, overlap between starts, the three loops against today's reference, the no-search families.
# Needs 02_newaxis.sh, 03_reference.sh, 04_search.sh, 05_readings.sh and 06_measures_prep.sh. Outputs go to $DUROOM_OUT/tests12 and $DUROOM_OUT/sphere/eval/bound.
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
: ${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
B=$DUROOM_OUT; C=$DUROOM_REPO/measure; O=$B/sphere; R=$O/runs; L=$B/logs; E=$O/eval; BD=$E/bound
T=$B/tests12/t3; OO=$B/tests12/onoff; Q=$B/tests12/qgtread; AR=$B/tests12/afterrun
SEARCH_PY="env LD_LIBRARY_PATH=$DUROOM_SEARCH_ENV/lib $DUROOM_SEARCH_ENV/bin/python"
mkdir -p $BD $OO/out $B/tests12/overlap/out $Q $B/tests12/tworulers/out $B/tests12/nosearch

# ---- evaluator on the 15 main runs
cell() { echo "$1=$R/$1_committee_euclidean_min_seedrow,$E/$1_emb.npz,$L/$1.out"; }
RA="$(cell qgc_ra_s6) $(cell qgc_ra_s7) $(cell qgc_ra_s8) $(cell qgc_ra_s9) $(cell qgc_ra_s10)"
PIN="$(cell qgc_pin_s6) $(cell qgc_pin_s7) $(cell qgc_pin_s8) $(cell qgc_pin_s9) $(cell qgc_pin_s10)"
C2="$(cell qgc_c2_s6) $(cell qgc_c2_s7) $(cell qgc_c2_s8) $(cell qgc_c2_s9) $(cell qgc_c2_s10)"
ALL="$RA $PIN $C2"
CACHE=$BD/ev_safety_conf.json
VERD1=C1=qgc_ra_s6:qgc_pin_s6,qgc_ra_s7:qgc_pin_s7,qgc_ra_s8:qgc_pin_s8,qgc_ra_s9:qgc_pin_s9,qgc_ra_s10:qgc_pin_s10
VERD2=C2=qgc_ra_s6:qgc_c2_s6,qgc_ra_s7:qgc_c2_s7,qgc_ra_s8:qgc_c2_s8,qgc_ra_s9:qgc_c2_s9,qgc_ra_s10:qgc_c2_s10
PAIRS=""; for n in 6 7 8 9 10; do PAIRS="$PAIRS --pair qgc_ra_s$n:qgc_pin_s$n --pair qgc_ra_s$n:qgc_c2_s$n"; done
cd $C

# Safety-checker flags of every admitted picture of the 15 main runs: ev_safety_conf.json (the checker needs diffusers, so it runs in the search environment).
$SEARCH_PY -B $C/controls_evaluate.py safety $O --safety-cache $CACHE $ALL
# Replay of each run's admission at its own fidelity limit (stops on any mismatch).
$DUROOM_PY -B $C/controls_evaluate.py selftest $O $(for c in $ALL; do echo "${c%%,*}"; done | tr '\n' ' ')
# Evaluator table over the 15 main runs: far, faithful, covered, hits and the verdicts of the two controls (conf_table.out).
$DUROOM_PY -B $C/controls_evaluate.py table $O --safety-cache $CACHE --verdict $VERD1 --verdict $VERD2 $PAIRS $ALL > $BD/conf_table.out

# ---- grid (reads the evaluator table and safety cache written above)
mkdir -p $B/tests12/arms $B/tests12/grid
# The arms without bars (hits per arm): tests12/arms/arms_barfree_v1.npz, .txt, _rows.csv.
$DUROOM_PY -B $C/arms_barfree_v1.py $B/tests12/arms/arms_barfree_v1
# Far x faithful grid for every group of pictures: tests12/grid/grid_v1.npz, .txt, _rows.csv.
$DUROOM_PY -B $C/grid_v1.py $B/tests12/grid/grid_v1

# ---- on / off tables
# On/off table and nearest-work column, runs and known-answer sets: onoff_v3_AB.txt, onoff_v3_AB.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $DUROOM_REPO/measure/onoff_v3.py $OO/out/onoff_v3_AB AB
# Large-leftover line and its nearest-work column: onoff_v3_C.txt, onoff_v3_C.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $DUROOM_REPO/measure/onoff_v3.py $OO/out/onoff_v3_C C
# Far-and-faithful pictures split by the nearest work: onoff_ff_v1.txt, onoff_ff_v1.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $DUROOM_REPO/measure/onoff_ff_v1.py $OO/out/onoff_ff_v1
# Dates of the real works near each picture, within the manifold line: onoff_dated_v2.txt, onoff_dated_v2.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $DUROOM_REPO/measure/onoff_dated_v2.py $OO/out/onoff_dated_v2
# Later works reached from different starts, with the shuffle: ov_overlap_v1.txt, ov_overlap_v1.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $DUROOM_REPO/measure/ov_overlap_v1.py $B/tests12/overlap/out/ov_overlap_v1

# ---- the three loops against today's reference
# Needs the safety files (ev_safety_qgt_<tag>.json) and completion markers of the runs against today's reference, made in 05_readings.sh.
# Place measure on the 15 runs: eligible pictures, covered, places (measure_qgt.txt, measure_qgt.npz).
cd $T/out && $DUROOM_PY -B $C/t3_measure_v5.py measure --out $T/out/measure_qgt --scales $T/out/scales_v5.npz --guards $T/out/guards_v3.npz --freeze $T/FREEZE_v5.txt --cells "V:qgt_ra_s11,qgt_ra_s12,qgt_ra_s13,qgt_ra_s14,qgt_ra_s15;C1:qgt_pin_s11,qgt_pin_s12,qgt_pin_s13,qgt_pin_s14,qgt_pin_s15;C2:qgt_c2_s11,qgt_c2_s12,qgt_c2_s13,qgt_c2_s14,qgt_c2_s15" --run-tmpl "$O/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz" --exact-tmpl "$O/eval/{tag}_emb_h200_v1.npz" --safety-tmpl "$O/eval/bound/ev_safety_qgt_{tag}.json" --afterrun $AR
# Replay of the search loop's centre path on the 15 runs against today's reference (pycma of the search environment): abq_v3_qgt_push_today_emb_replay.npz.
cd $Q && $SEARCH_PY -B $C/arms_barfree_qgt_v3.py replay $Q/abq_v3_qgt_push_today_emb_replay.npz --prefix qgt_
# The three loops compared without a bar against today's reference: abq_v3_qgt_push_today_emb.txt, .npz, _rounds.csv.
cd $Q && $DUROOM_PY -B $C/arms_barfree_qgt_v3.py readout $Q/abq_v3_qgt_push_today_emb --prefix qgt_ --ref push_today_emb.npz --exact '{tag}_emb_h200_v1.npz' --replay $Q/abq_v3_qgt_push_today_emb_replay.npz
# Leftover basis and null of today's reference T: tr_basis_v1_T_basis.npz (also tr_basis_v1_P_check.npz).
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/tr_fitbasis_v1.py $B/tests12/tworulers/out/tr_basis_v1
# Leftover share of every picture against today's reference, per arm and start: third_readout_today_v2.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/tr_readout_v2.py $B/tests12/tworulers/out/third_readout_today_v2 --prefix qgt_ --ref push_today_emb.npz --basis $B/tests12/tworulers/out/tr_basis_v1_T_basis.npz --label 'leftover readout, runs against T'
# Off-the-manifold shares against today's reference, per arm and start, with their leftover and the places of the measure above: onoff_today3_qgt.txt, onoff_today3_qgt.npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_today_v3.py $OO/out/onoff_today3_qgt --prefix qgt_ --measure-npz $T/out/measure_qgt.npz

# ---- no-search families
# Far score, rank and geometry of every no-search request: nosearch_v5.txt and _rows.csv.
cd $DUROOM_REPO/measure && $DUROOM_PY -B $DUROOM_REPO/measure/nosearch_v5.py $B/tests12/nosearch/nosearch_v5
