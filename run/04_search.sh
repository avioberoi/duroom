#!/usr/bin/env bash
# The main runs: search (qgc_ra), random-score loop (qgc_pin) and no-step loop (qgc_c2) at starts 6 to 10; the same three against today's reference T (qgt_*, starts 11 to 15); the single walk (qgw_walk).
# Needs 03_reference.sh; runs in DUROOM_SEARCH_ENV. Each run writes $O/runs/<tag>_committee_euclidean_min_seedrow/all_candidates.npz (1,921 rows); its stdout and stderr go to $DUROOM_OUT/logs/<tag>.out and .err.
set -eo pipefail
source "$(dirname "$0")/env.sh"
unset LD_LIBRARY_PATH
B=$DUROOM_OUT; C=$DUROOM_REPO/search; O=$B/sphere
export SPHERE_CLIP=quickgelu
: ${DUROOM_SEARCH_ENV:?set DUROOM_SEARCH_ENV to the prefix of the search environment (requirements-search.txt)}
export SPHERE_ENV=$DUROOM_SEARCH_ENV
COMMON="--space sphere --start seedrow --start-pool $O/pools/starts_pre1900.npz --norm-from-push --aggregation min --admit-q99 --delta 0.575 --warmup 1 --gate-threshold 0 --no-early-stop --log-candidates --maxiter 60 --popsize 32 --sigma-cap 2 --pushforward $O/push_emb.npz --render-seeded --reanchor 4"
mkdir -p $O/controls $B/logs

# Step-size file of the no-step loop: 60 values of 1e-5.
$DUROOM_PY -c "import numpy as np; np.savez('$O/controls/c2_sigma_1e-5.npz', log_sigma=np.full(60, 1.0e-5, dtype=np.float64))"

# Search: CMA-ES on the CLIP sphere, admission by the bar, fidelity limit 0.7788.
for s in 6 7 8 9 10; do
  bash $C/sphere.sbatch search --mode cma $COMMON --seed $s --tag qgc_ra_s${s} --config $C/config_sphere15.yaml --fidelity-max 0.7788 > $B/logs/qgc_ra_s${s}.out 2> $B/logs/qgc_ra_s${s}.err
done
# Random-score loop: random scores, taking the step sizes logged by the search run of the same start.
for s in 6 7 8 9 10; do
  bash $C/sphere.sbatch search --mode cma --random-fitness $COMMON --seed $s --tag qgc_pin_s${s} --config $C/config_sphere15.yaml --sigma-path $O/runs/qgc_ra_s${s}_committee_euclidean_min_seedrow/all_candidates.npz --fidelity-max 0.7788 > $B/logs/qgc_pin_s${s}.out 2> $B/logs/qgc_pin_s${s}.err
done
# No-step loop: random scores, step size 1e-5.
for s in 6 7 8 9 10; do
  bash $C/sphere.sbatch search --mode cma --random-fitness $COMMON --seed $s --tag qgc_c2_s${s} --config $C/config_sphere15_c2.yaml --sigma-path $O/controls/c2_sigma_1e-5.npz --fidelity-max 0.7788 > $B/logs/qgc_c2_s${s}.out 2> $B/logs/qgc_c2_s${s}.err
done

# The same three loops against today's reference T (start pool of every date).
COMMON_T="${COMMON/starts_pre1900.npz --norm/starts_pre2027.npz --norm}"; COMMON_T="${COMMON_T/push_emb.npz --render/push_today_emb.npz --render}"
# Search against T: qgt_ra_s11 to s15.
for s in 11 12 13 14 15; do
  bash $C/sphere.sbatch search --mode cma $COMMON_T --seed $s --tag qgt_ra_s${s} --config $C/config_sphere15.yaml --fidelity-max 0.7788 > $B/logs/qgt_ra_s${s}.out 2> $B/logs/qgt_ra_s${s}.err
done
# Random-score loop against T, taking the step sizes of the search run of the same start: qgt_pin_s11 to s15.
for s in 11 12 13 14 15; do
  bash $C/sphere.sbatch search --mode cma --random-fitness $COMMON_T --seed $s --tag qgt_pin_s${s} --config $C/config_sphere15.yaml --sigma-path $O/runs/qgt_ra_s${s}_committee_euclidean_min_seedrow/all_candidates.npz --fidelity-max 0.7788 > $B/logs/qgt_pin_s${s}.out 2> $B/logs/qgt_pin_s${s}.err
done
# No-step loop against T: qgt_c2_s11 to s15.
for s in 11 12 13 14 15; do
  bash $C/sphere.sbatch search --mode cma --random-fitness $COMMON_T --seed $s --tag qgt_c2_s${s} --config $C/config_sphere15_c2.yaml --sigma-path $O/controls/c2_sigma_1e-5.npz --fidelity-max 0.7788 > $B/logs/qgt_c2_s${s}.out 2> $B/logs/qgt_c2_s${s}.err
done

# Single walk: step angle per round and start from the five search runs (walk_theta_v1.npz), then one picture at a time, a step only on a higher score, same budget as the loops.
COMMON_W="${COMMON% --reanchor 4}"
mkdir -p $B/tests12/walk/theta
LD_LIBRARY_PATH=$DUROOM_SEARCH_ENV/lib $DUROOM_SEARCH_ENV/bin/python -B $C/walk_theta_v1.py $B/tests12/walk/theta/walk_theta_v1
for s in 6 7 8 9 10; do
  bash $C/sphere.sbatch py $C/search_walk_v1.py --out-root $O/runs --mode walk $COMMON_W --seed $s --tag qgw_walk_s${s} --config $C/config_sphere15.yaml --fidelity-max 0.7788 --walk-theta $B/tests12/walk/theta/walk_theta_v1.npz > $B/logs/qgw_walk_s${s}.out 2> $B/logs/qgw_walk_s${s}.err
done
