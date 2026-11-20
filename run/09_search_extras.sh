#!/usr/bin/env bash
# The free search, the extra arms, the single walk's readouts, the loop facts, the overlap checks and the bound and decoder checks that the earlier steps do not cover, in dependency order.
# Needs 01 to 08 (07 first for the onoff, t3 and grid files); runs in DUROOM_SEARCH_ENV where a step says SEARCH_PY. Outputs go to $DUROOM_OUT/tests12 and $DUROOM_OUT/sphere/eval/bound.
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
: ${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
B=$DUROOM_OUT; S=$DUROOM_REPO/search; C=$DUROOM_REPO/measure; O=$B/sphere; L=$B/logs; E=$O/eval; BD=$E/bound; OO=$B/tests12/onoff/out; OV=$B/tests12/overlap/out
SEARCH_PY="env LD_LIBRARY_PATH=$DUROOM_SEARCH_ENV/lib $DUROOM_SEARCH_ENV/bin/python"
ENC4="clip-vit-l-14 dinov3-l siglip-so400m aimv2-huge"
export SPHERE_CLIP=quickgelu
SE=$DUROOM_SEARCH_ENV
mkdir -p $BD $OO $OV $B/tests12/extra $B/tests12/loopfacts $B/tests12/shift $B/tests12/meetreal/out $B/tests12/walk/out $B/tests12/agree $B/tests12/pf $B/tests12/nosearch $B/tests12/afterrun/oracle $L $O/lists
COMMON="--space sphere --start seedrow --start-pool $O/pools/starts_pre1900.npz --norm-from-push --aggregation min --admit-q99 --delta 0.575 --warmup 1 --gate-threshold 0 --no-early-stop --log-candidates --maxiter 60 --popsize 32 --sigma-cap 2 --pushforward $O/push_emb.npz --render-seeded --reanchor 4"

# ---- free search: seeds 1 to 3, no restart, no fidelity limit, config.yaml
COMMON_FREE="${COMMON% --reanchor 4}"
# Search (CMA-ES on the score): runs/main_s<N>.
for s in 1 2 3; do SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode cma $COMMON_FREE --seed $s --tag main_s${s} > $L/main_s${s}.out 2> $L/main_s${s}.err; done
# Random-score loop: runs/rf_s<N>.
for s in 1 2 3; do SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode cma --random-fitness $COMMON_FREE --seed $s --tag rf_s${s} > $L/rf_s${s}.out 2> $L/rf_s${s}.err; done
# Random directions at the angles of the search run of the same seed: runs/b0_s<N> (all_draws.npz).
for s in 1 2 3; do SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode random --angles-from $O/runs/main_s${s}_committee_euclidean_min_seedrow/all_candidates.npz $COMMON_FREE --seed $s --tag b0_s${s} > $L/b0_s${s}.out 2> $L/b0_s${s}.err; done
# Exact four-encoder readings of the free runs: eval/<tag>_emb.npz.
for tag in main_s{1..3} rf_s{1..3} b0_s{1..3}; do
  f=$O/runs/${tag}_committee_euclidean_min_seedrow/all_candidates.npz; [ -f $f ] || f=$O/runs/${tag}_committee_euclidean_min_seedrow/all_draws.npz
  $DUROOM_PY -c "import numpy as np; z=np.load('$f', allow_pickle=True); open('$O/lists/${tag}.txt','w').write('\n'.join(map(str, z['paths']))+'\n')"
  bash $S/sphere.sbatch embed --list $O/lists/${tag}.txt --out $E/${tag}_emb.npz --encoders $ENC4
done
# The grid of 07 again with the free-run groups present: tests12/grid/grid_v1_free.txt, .npz, _rows.csv.
$DUROOM_PY -B $C/grid_v1.py $B/tests12/grid/grid_v1_free
# Admission rate, share of nearest works dated 1900 or later and loadings of the free runs, years read in DINOv3: eval/score_g_free.json.
SCORE_JUDGE=dinov3-l $DUROOM_PY -B $C/score_g.py $E/score_g_free.json --pools $O/pools/starts_pre1900.npz --axes $B/newaxis/k32/newaxis.json --p-emb $O/push_emb.npz --runs main:1=$O/runs/main_s1_committee_euclidean_min_seedrow/all_candidates.npz main:2=$O/runs/main_s2_committee_euclidean_min_seedrow/all_candidates.npz main:3=$O/runs/main_s3_committee_euclidean_min_seedrow/all_candidates.npz rf:1=$O/runs/rf_s1_committee_euclidean_min_seedrow/all_candidates.npz rf:2=$O/runs/rf_s2_committee_euclidean_min_seedrow/all_candidates.npz rf:3=$O/runs/rf_s3_committee_euclidean_min_seedrow/all_candidates.npz b0:1=$O/runs/b0_s1_committee_euclidean_min_seedrow/all_draws.npz b0:2=$O/runs/b0_s2_committee_euclidean_min_seedrow/all_draws.npz b0:3=$O/runs/b0_s3_committee_euclidean_min_seedrow/all_draws.npz --emb main_s1=$E/main_s1_emb.npz main_s2=$E/main_s2_emb.npz main_s3=$E/main_s3_emb.npz rf_s1=$E/rf_s1_emb.npz rf_s2=$E/rf_s2_emb.npz rf_s3=$E/rf_s3_emb.npz b0_s1=$E/b0_s1_emb.npz b0_s2=$E/b0_s2_emb.npz b0_s3=$E/b0_s3_emb.npz

# ---- extra arms (each run is a search command of 04 with one change; qgx on one card type, qgh on another, to compare cards)
# Search with the restart taken from 1, 16 and 32 pictures instead of 4: runs/qgm_m<M>_s<N>.
for m in 1 16 32; do for s in 6 7 8 9 10; do
  SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode cma ${COMMON/--reanchor 4/--reanchor $m} --seed $s --tag qgm_m${m}_s${s} --config $S/config_sphere15.yaml --fidelity-max 0.7788 > $L/qgm_m${m}_s${s}.out 2> $L/qgm_m${m}_s${s}.err
done; done
# No-step command told the real score (no --random-fitness): runs/qgx_sc_s<N> (first card type) and runs/qgh_sc_s<N> (second).
for t in qgx qgh; do for s in 6 7 8 9 10; do
  SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode cma $COMMON --seed $s --tag ${t}_sc_s${s} --config $S/config_sphere15_c2.yaml --sigma-path $O/controls/c2_sigma_1e-5.npz --fidelity-max 0.7788 > $L/${t}_sc_s${s}.out 2> $L/${t}_sc_s${s}.err
done; done
# The no-step loop of 04 repeated: runs/qgx_c2_s<N>.
for s in 6 7 8 9 10; do
  SPHERE_ENV=$SE bash $S/sphere.sbatch search --mode cma --random-fitness $COMMON --seed $s --tag qgx_c2_s${s} --config $S/config_sphere15_c2.yaml --sigma-path $O/controls/c2_sigma_1e-5.npz --fidelity-max 0.7788 > $L/qgx_c2_s${s}.out 2> $L/qgx_c2_s${s}.err
done
# Exact readings of the restart-from-1 runs: eval/qgm_m1_s<N>_emb_h200_v1.npz.
for s in 6 7 8 9 10; do bash $C/aft_embed_cell.sh qgm_m1_s${s}; done
# Replay of the search loop's centre path on every cell of the extra arms (pycma of the search environment): tests12/extra/xa_replay_v2.npz.
$SEARCH_PY -B $C/xa_readout_v2.py replay $B/tests12/extra/xa_replay_v2.npz
# Far, spaced and kept counts, pair chances, gap and per-round moves of the extra arms beside the main runs: xa_readout_v2.txt, _rounds.csv, .npz.
$DUROOM_PY -B $C/xa_readout_v2.py readout $B/tests12/extra/xa_readout_v2 --replay $B/tests12/extra/xa_replay_v2.npz
# Restart-from-1 runs beside the search runs: far and faithful, hits, large leftover and dated classes: overlap/out/rc_m1_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/rc_m1_v1.py $OV/rc_m1_v1
# A far bar per region of P for the 15 main runs: overlap/out/rc_region_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/rc_region_v1.py $OV/rc_region_v1

# ---- single walk: readouts against the loops
# Counts of the walk, search and controls from run files: tests12/walk/out/walk_readout_v4.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/walk_readout_v4.py $B/tests12/walk/out/walk_readout_v4
# Exact readings of the walk runs under the loops' file name: eval/qgw_walk_s<N>_emb.npz.
for s in 6 7 8 9 10; do bash $S/sphere.sbatch embed --list $O/lists/qgw_walk_s${s}.txt --out $E/qgw_walk_s${s}_emb.npz --encoders $ENC4; done
# On/off chain on the 15 loop cells and the 5 walk cells, walk read from eval/<tag>_emb.npz: onoff_walk_v2_rtx.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_walk_v2.py $OO/onoff_walk_v2_rtx --walk-exact rtx
# The same with the walk read from eval/<tag>_emb_h200_v1.npz: onoff_walk_v2_h200.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_walk_v2.py $OO/onoff_walk_v2_h200 --walk-exact h200
# Pictures that stay apart at DINOv3 chord 0.575, lead row and off row, per cell and in reverse order: onoff/out/lead_spaced_v1.txt, .npz.
$DUROOM_PY -B $C/lead_spaced_v1.py $OO/lead_spaced_v1 --run

# ---- oracle lines, undated and artist-era splits, leftover directions
A=$B/tests12/afterrun/oracle
# Picture list of the 74 oracle frames (the 71 line pictures and the 3 start pictures) in request order: afterrun/oracle/oracle_list_v1.txt.
$DUROOM_PY -c "import os, numpy as np; f=[str(x) for x in np.load('$B/tests12/nosearch/nosearch_v5_oracle_requests.npz', allow_pickle=True)['frame_file']]; open('$A/oracle_list_v1.txt','w').write('\n'.join(x if x.startswith('/') else os.path.join('$O/oracle/frames', x) for x in f)+'\n')"
# Exact readings of the oracle pictures: afterrun/oracle/oracle_emb_h200_v1.npz.
bash $S/sphere.sbatch embed --list $A/oracle_list_v1.txt --out $A/oracle_emb_h200_v1.npz --encoders $ENC4
# Known-answer pictures of P read on the same card: afterrun/oracle/kat_emb.npz, compared with P's stored readings.
$DUROOM_PY -B $C/aft_tools_v4.py katlist $A/kat_list.txt
bash $S/sphere.sbatch embed --list $A/kat_list.txt --out $A/kat_emb.npz --encoders $ENC4
$DUROOM_PY -B $C/aft_tools_v4.py katcmp $A/kat_emb.npz
# Far score, rank, gap and geometry of the oracle-line pictures: nosearch/nosearch_oracle_v1.txt and _rows.csv.
cd $DUROOM_REPO/measure && $DUROOM_PY -B $C/nosearch_oracle_v1.py $B/tests12/nosearch/nosearch_oracle_v1
# On/off classes with the artist era of undated originals: onoff/out/onoff_artist_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_artist_v1.py $OO/onoff_artist_v1
# SINCE-CLEAN against SINCE with an undated original near, both-reader classes, no-search families and oracle lines: onoff/out/onoff_since_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_since_v1.py $OO/onoff_since_v1
# Leftover with 16 to 64 directions for the lead rows, Spearman against the far score: onoff/out/onoff_dirs_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/onoff_dirs_v1.py $OO/onoff_dirs_v1

# ---- loop facts of the 15 main runs
# Replay of pycma on the 15 runs (pycma of the search environment): loopfacts/lf_cma_v2.npz.
$SEARCH_PY -B $C/lf_cma_v2.py $B/tests12/loopfacts/lf_cma_v2.npz
# Loop facts (centre moves, drift, no-step runs, feasibility gate, coordinates 644 and 413): loopfacts/lf_main_v2.txt, .npz.
$DUROOM_PY -B $C/lf_main_v2.py $B/tests12/loopfacts/lf_main_v2 $B/tests12/loopfacts/lf_cma_v2.npz
# Shift score, coverage against today's reference and the per-round table: shift/req4_v3.txt and loopfacts/lf_rounds_v2.csv.
cd $DUROOM_REPO/measure && $DUROOM_PY -B $C/req4_v3.py $B/tests12/shift/req4_v3 $B/tests12/loopfacts/lf_rounds_v2.csv $B/tests12/loopfacts/lf_cma_v2.npz $B/tests12/loopfacts/lf_main_v2.npz

# ---- do the starts meet at the same later works
# Equal samples per start, all and thinned: overlap/out/meet_equal_v1.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/meet_equal_v1.py $OV/meet_equal_v1
# Thinned draws, paired, search against the random-score loop: overlap/out/meet_equal_v2.txt, .npz.
cd $DUROOM_REPO/search && $DUROOM_PY -B $C/meet_equal_v2.py $OV/meet_equal_v2
# Places among the far works of the originals dated 1970 on, 20 groupings, two radii: meetreal/out/meetreal_v1.txt, .npz.
$DUROOM_PY -B $C/meetreal_v1.py $B/tests12/meetreal/out/meetreal_v1
# Why no places: balls, same-artist share, a ladder of rules, positive control: meetreal/out/meetreal_diag_v1.txt.
$DUROOM_PY -B $C/meetreal_diag_v1.py $B/tests12/meetreal/out/meetreal_v1.npz $B/tests12/meetreal/out/meetreal_diag_v1

# ---- agreement of the encoders on which later works are far
# t1_v2.py with CLIP as the only observer: tests12/t1/run_rob_clip.
SLURM_CPUS_PER_TASK=12 $DUROOM_PY -B $DUROOM_REPO/data/t1_v2.py --out $B/tests12/t1 --tag rob_clip --observers clip-vit-l-14 > $B/tests12/t1/t1_rob_clip.log 2>&1
mv $B/tests12/t1/run_*_rob_clip $B/tests12/t1/run_rob_clip
# Per-work scores of the three encoders side by side, Spearman, far counts, regions under 1% pre-1900 share: agree/agree_p1_v1.npz.
$DUROOM_PY -B $C/agree_p1_v1.py
# Later works flagged at the line that flags k held-out earlier works, AUC, AND rule: agree/l1_v1.out.
$DUROOM_PY -B $C/l1_v1.py > $B/tests12/agree/l1_v1.out

# ---- bound, noise, decoder and baseline checks
# Reading against request inside a round, on the main runs: eval/bound/gain.out.
$DUROOM_PY -B $C/gain.py $O/runs $O/push_emb.npz qgc_ra_s6:N qgc_ra_s7:N qgc_ra_s8:N qgc_ra_s9:N qgc_ra_s10:N qgc_pin_s6:R qgc_pin_s7:R qgc_pin_s8:R qgc_pin_s9:R qgc_pin_s10:R > $BD/gain.out
# Bands in which later works' offsets lie: eval/bound/band.out.
$DUROOM_PY -B $C/band.py $DUROOM_REPO/data > $BD/band.out
# Optimizers on the novelty score in embedding space, no decoder: eval/bound/optim_walk.out.
$DUROOM_PY -B $C/optim_walk.py $DUROOM_REPO/data > $BD/optim_walk.out
# Noise images against today's reference: far, covered: eval/bound/noise_covered.out.
$DUROOM_PY -B $C/noise_covered.py > $BD/noise_covered.out
# One bar for P against a bar per region, P2 and POS: eval/bound/perregion_v2.out.
$DUROOM_PY -B $C/perregion_v2.py > $BD/perregion_v2.out
# Far shares per bar level and k for P2, POS and the runs scored in 06: eval/sb_grid.json and eval/bound/sb_grid.out.
$DUROOM_PY -B $C/sb_grid.py > $BD/sb_grid.out
# Prior and round-trip baselines scored against P: eval/sens_read.npz, then the readout eval/bound/baselines_read.out.
$DUROOM_PY -B $S/baselines_score.py $E/sens_read.npz --baselines $O/baselines/prior $O/baselines/chains_h200_qg
$DUROOM_PY -B $C/baselines_read.py $E/sens_read.npz > $BD/baselines_read.out
# Per-round far, H, DH and QH counts of the 15 main runs: eval/bound/confirm_rounds.json.
$DUROOM_PY -B $C/confirm_rounds.py | tail -n 1 > $BD/confirm_rounds.json
# Off-the-manifold pictures that are also faithful, runs against today's reference: onoff/out/p04_count_strict.out.
bash $C/p04_count_strict.sh > $OO/p04_count_strict.out

# ---- last: needs the picture lists inputs/pictures_v2.json, pictures_v3.json, and reembed/wikiart_catfm.npz from run/10 (clip_raw); run 10 first
# Facts per picture of the picture list inputs/pictures_v2.json (and pictures_v3.json): tests12/pf/pf_facts_v1.txt, .json and pf_facts_v2.
$DUROOM_PY -B $C/pf_facts_v1.py $B/tests12/pf/pf_facts_v1
$DUROOM_PY -B $C/pf_facts_v2.py $B/tests12/pf/pf_facts_v2
# Decoder reach: frames along directions from named starts (plan, paint, read): walk/manifest.json, scores.json, summary.json and contact-sheet images. Needs reembed/wikiart_catfm.npz from the data side (no run file here writes it).
mkdir -p $B/walk
$DUROOM_PY -B $S/walk.py $B/walk plan
SPHERE_ENV=$SE bash $S/sphere.sbatch py $S/walk.py $B/walk render
bash $S/sphere.sbatch py $S/walk.py $B/walk score
# Needs reembed/wikiart_catfm.npz (see above). Decoder at guidance 3 and 10 on 256 works, two seeds: decode/variations_n256/score.json and decode/compare_n256.json.
mkdir -p $B/decode
$DUROOM_PY -B $S/decode.py variations $B/decode plan n256
SPHERE_ENV=$SE bash $S/sphere.sbatch py $S/decode.py variations $B/decode render n256
bash $S/sphere.sbatch py $S/decode_score.py $B/decode/variations_n256 0
$DUROOM_PY -B $S/decode_compare.py $B/decode
