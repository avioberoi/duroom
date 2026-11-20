"""Downscaled copies of the pictures in two reader-supplied tables (figures/marks/search_finds_v1.csv, search_finds_today_v1.csv), and the credit tables figures/data/search_finds_v1.csv, search_finds_today_v1.csv.
Usage: python derive_search_finds.py"""
import csv
import os
from pathlib import Path
from PIL import Image

FIG = Path(os.environ['DUROOM_OUT']) / 'figures'
HERE = FIG / 'data'
HERE.mkdir(parents=True, exist_ok=True)
SRC = FIG / 'marks'
OUT = FIG / 'finds'
OUT.mkdir(exist_ok=True)
CAP = 720
# usual forms of four artist names
NAMES = {'Vincent Van Gogh': 'Vincent van Gogh', 'Kathe Kollwitz': 'Käthe Kollwitz',
         'Zdzislaw Beksinski': 'Zdzisław Beksiński', 'Nuri Iyem': 'Nuri İyem'}
LEAVE_OUT = 'p_s6_1438.png'


def copy(src, name):
    src = Path(src)
    assert src.is_file(), src
    dst = OUT / name
    im = Image.open(src).convert('RGB')
    size = im.size
    im.thumbnail((CAP, CAP), Image.LANCZOS)
    im.save(dst, 'JPEG', quality=88, optimize=True, progressive=True)
    return f'finds/{name}', size, im.size


def derive(table, third, out_name):
    if not (SRC / table).exists():
        raise SystemExit(f'missing {SRC / table}: supply the table')
    rows = list(csv.DictReader(open(SRC / table, encoding='utf-8')))
    keep = [r for r in rows if Path(r['found_path']).name != LEAVE_OUT]
    out = []
    for r in keep:
        found, fs, fs2 = copy(r['found_path'], f"found_{Path(r['found_path']).stem}.jpg")
        start, ss, ss2 = copy(r['start_path'], f"{Path(r['start_path']).stem}.jpg")
        other, os_, os2 = copy(r[f'{third}_path'], f"{third}_{Path(r[f'{third}_path']).stem}.jpg")
        assert r[f'{third}_show'] == 'yes'
        out.append({
            'found': found, 'round': int(r['round']), 'loop': r['loop'],
            'start': start, 'start_artist': NAMES.get(r['start_artist'], r['start_artist']), 'start_title': r['start_title'],
            'start_year': r['start_year'],
            third: other, f'{third}_artist': NAMES.get(r[f'{third}_artist'], r[f'{third}_artist']),
            f'{third}_title': r[f'{third}_title'], f'{third}_year': r[f'{third}_year'],
            'source': r[f'{third}_source'], 'fidelity': r['fidelity'], 'chord': r['chord'], 'bar': r['bar'],
            'sizes_px': f'start {ss[0]}x{ss[1]}, found {fs[0]}x{fs[1]}, {third} {os_[0]}x{os_[1]}'})
    with open(HERE / out_name, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(out_name, len(out), 'rows')
    return out


if __name__ == '__main__':
    derive('search_finds_v1.csv', 'later', 'search_finds_v1.csv')
    derive('search_finds_today_v1.csv', 'nearest', 'search_finds_today_v1.csv')
    print(sorted(p.name for p in OUT.iterdir()))
