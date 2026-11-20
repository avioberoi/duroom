"""Fact blocks and column heads of the off-the-manifold gallery; a work whose corpus title is '-' is named without it.
Usage: imported by search_finds_off1900_v5.py"""
import picture_facts_figs_v1 as pf

HEADS = ('starting work, before 1900', 'painted by the search', 'nearest real work in SigLIP, any date', 'round, and the nearest real works')


def facts(r):
    b = pf.fact_blocks(r['picture_id'], large_leftover=r['large_leftover'] == '1')
    return [[(t.replace(', -, ', ', '), it) for t, it in blk] for blk in b]
