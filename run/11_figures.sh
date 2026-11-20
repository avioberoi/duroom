#!/usr/bin/env bash
# Figures of the main text: derive steps (data files in $DUROOM_OUT/figures/data), the figure scripts, then the print versions. Figures go to $DUROOM_OUT/figures.
# Needs 01 to 10. Needs matplotlib, Pillow, PyMuPDF, pikepdf (figures/requirements-figures.txt). The last block needs files you supply in $DUROOM_OUT/figures/marks (see the module docstrings).
set -eo pipefail
source "$(dirname "$0")/env.sh"
F=$DUROOM_REPO/figures; D=$F/derive; B=$DUROOM_OUT
mkdir -p $B/figures/data
cd $F

# ---- derive steps
# Dated works by decade: data/record_plot.json (counts of 08).
$DUROOM_PY -B $D/derive_record.py
# Group gaps and kept-against-k curves: data/q1.json (common of 08).
$DUROOM_PY -B $D/derive_q1.py
# Decade means in the first two new directions: data/q1b.json (newaxis_check of 08).
$DUROOM_PY -B $D/derive_q1b.py
# Sphere sample, angle histogram and percentiles: data/measured_sphere.json (figdata of 10).
$DUROOM_PY -B $D/derive_measured_visuals.py
# CLIP pass of the public-domain print: data/image_to_vector.json (clipviz of 10).
$DUROOM_PY -B $D/derive_image_to_vector.py
# Leftover fan and the two pasts, with thumbnails from $WIKIART_DIR/artwork_images: data/q3_fan.json, q3_two_pasts.json (leftover of 08).
$DUROOM_PY -B $D/derive_q3_fan.py
$DUROOM_PY -B $D/derive_q3_two_pasts.py
# Three far scores and the straight route: data/q5_far.json, q5_walk.json, then data/q5_three_far.json (distance and corner of 08).
$DUROOM_PY -B $D/derive_q4_q6.py
$DUROOM_PY -B $D/derive_q5_three_far.py
# Decoder-walk curves and retrodiction tables: data/search_walk.json, search_retro.json (walk of 09, retro of 08).
$DUROOM_PY -B $D/derive_search.py
# Points and cell counts of every kind of picture: data/four_cells.json (grid of 07).
$DUROOM_PY -B $D/derive_four_cells.py
# Far and faithful counts by way of asking and by dates of the nearest works: data/ways_of_asking.csv, dated_split.csv (07, 09).
$DUROOM_PY -B $D/derive_ways_dated.py
# Distinct hits of the search and its controls: data/q10_confirm.json (conf_table.out of 07).
$DUROOM_PY -B $D/derive_q10_confirm.py
# Fact record of each gallery picture: data/picture_facts_v1.json (pf_facts of 09).
$DUROOM_PY -B $D/derive_picture_facts.py
# ---- figures
# Drawn figures, no data.
$DUROOM_PY -B $F/manifold_definition_v3.py
$DUROOM_PY -B $F/shape_walk.py
$DUROOM_PY -B $F/shape_orient.py
# Data figures.
$DUROOM_PY -B $F/image_to_vector.py
$DUROOM_PY -B $F/record_decades.py
$DUROOM_PY -B $F/sphere.py
$DUROOM_PY -B $F/q1_gap_all.py
$DUROOM_PY -B $F/q1_kept_k.py
$DUROOM_PY -B $F/q1b_path.py
$DUROOM_PY -B $F/q3_fan.py
$DUROOM_PY -B $F/q3_two_pasts_v2.py
$DUROOM_PY -B $F/q5_three_far.py
$DUROOM_PY -B $F/q5_walk.py
$DUROOM_PY -B $F/search_figures.py
$DUROOM_PY -B $F/search_response_v2.py
$DUROOM_PY -B $F/four_cells_v4.py
$DUROOM_PY -B $F/dated_split_v4.py
$DUROOM_PY -B $F/q10_confirm_v4.py
# Far and faithful by way of asking: ways_of_asking_vert_v2c and ways_of_asking_ladder_v2.
$DUROOM_PY -B $F/ways_of_asking_titles_v1.py

# ---- print versions (white page)
# NAME-print.pdf: the paper colour becomes white.
$DUROOM_PY -B $F/print_variants_v1.py warm manifold_definition_v3 shape_walk_s four_cells_v4 dated_split_v4 q10_confirm_v4 q1_kept_k q1b_path q3_two_pasts_v2 record_decades ways_of_asking_ladder_v2 ways_of_asking_vert_v2c

# ---- figures that need supplied files in $DUROOM_OUT/figures/marks
# Downscaled copies and credit tables of the gallery pictures, from marks/search_finds_v1.csv and marks/search_finds_today_v1.csv.
$DUROOM_PY -B $D/derive_search_finds.py
# The pair of pictures of the introduction (search_find_pair_v2).
$DUROOM_PY -B $F/search_finds_v1.py
# The galleries of the search results (search_finds_main_v2 with the extra lines if marks/byeye_v1.json exists, search_finds_today_v4, search_finds_off1900_v5) and the genre chart.
$DUROOM_PY -B $F/picture_facts_figs_v2.py
$DUROOM_PY -B $F/search_finds_today_v4.py
$DUROOM_PY -B $F/search_finds_off1900_v5.py
$DUROOM_PY -B $F/search_genres_v3.py
# NAME-print4_p1.pdf, _p2.pdf: the tall galleries cut into page parts on the rules between rows.
$DUROOM_PY -B $F/print_split_v4.py search_finds_main_v2 search_finds_today_v4
