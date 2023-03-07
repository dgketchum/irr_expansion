import json
import os
from collections import OrderedDict
from itertools import combinations
from pprint import pprint

import numpy as np
import pandas as pd

from gridded_data import BASIN_STATES
from utils.cdl import cdl_key


def get_openet_cdl(in_dir, join_csv, out_dir):
    with open('tiles.json', 'r') as f_obj:
        tiles_dct = json.load(f_obj)

    crop_counts = {}
    cdl = cdl_key()

    for state in BASIN_STATES:

        tiles = tiles_dct[state]
        l = [os.path.join(join_csv, 'openet_cdl_{}_{}_2021.csv'.format(state, tile)) for tile in tiles]
        first = True
        for f_ in l:
            c = pd.read_csv(f_, index_col='OPENET_ID')
            if first:
                adf = c.copy()
                first = False
            else:
                adf = pd.concat([adf, c])

        f = os.path.join(in_dir, '{}.csv'.format(state))
        df = pd.read_csv(f, index_col='OPENET_ID')
        match = [i for i in df.index if i in adf.index]
        df = df.loc[match]
        df['CROP_2021'] = [0 for _ in range(df.shape[0])]
        df.loc[match, 'CROP_2021'] = adf.loc[match, 'mode'].values.astype(int)
        outf = os.path.join(out_dir, '{}.csv'.format(state))
        counts = np.unique(df.values, return_counts=True)
        for code, ct in zip(counts[0], counts[1]):
            if code not in crop_counts.keys():
                crop_counts[code] = ct
            else:
                crop_counts[code] += ct
        df.to_csv(outf)

    crop_counts = {k: v for k, v in crop_counts.items() if k > 0}
    codes = list(crop_counts.keys())
    l = sorted([(c, (cdl[c][0], crop_counts[c])) for c in codes if len(cdl[c][0]) > 3],
               key=lambda x: x[1][1], reverse=True)
    dct = OrderedDict(l)
    pprint(dct)


def transition_probability(cdl_npy, prices, out_matrix, glob='2MAR2023'):
    counts, set_ = None, None
    cdl = cdl_key()

    time_series_length = 17
    df = None

    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    set_, classes = list(df.columns), [cdl[c][0] for c in df.columns]

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = [(int(a), int(b)) for a, b in keys]
    rec = np.moveaxis(rec, 0, 1)

    counts = {k: 0 for k in keys}
    from_crop = {k: [] for k in keys}
    to_crop = {k: [] for k in keys}
    spei = {k: [] for k in keys}

    df = df.to_dict()
    df = {k: {kk.strftime('%Y-%m-%d'): vv for kk, vv in v.items()} for k, v in df.items()}

    ct = 0
    for i, v in enumerate(rec):
        for e, c in enumerate(v.T):
            # skip last year of series, and first three (null) years
            if e == 16 or e < 2:
                continue

            fc, tc = int(c[3]), int(v.T[e + 1, 3])
            trans = (fc, tc)
            if not (trans[0] in set_ and trans[1] in set_):
                continue

            y = e + 2005
            counts[trans] += 1
            dtstr = '{}-01-01'.format(y)
            from_crop[trans].append(df[fc][dtstr])
            to_crop[trans].append(df[tc][dtstr])
            spei[trans].append(c[1])
            ct += 1
            if fc != tc:
                pass

        if i % 10000 == 0.:
            print(i)

    print('{} transtions'.format(ct))
    map = np.zeros((len(classes), len(classes)))
    for r, c in counts.keys():
        map[set_.index(r), set_.index(c)] = counts[r, c]

    prob = np.divide(map, np.sum(map, axis=0))

    tdf = pd.DataFrame(columns=classes, index=classes, data=prob)
    out_file = os.path.join(out_matrix, 'transitions_{}_crop.csv'.format(len(set_)))
    tdf.to_csv(out_file, float_format='%.3f')

    tdf = pd.DataFrame(columns=classes, index=classes, data=map)
    out_file = os.path.join(out_matrix, 'transitions_{}_crop_counts.csv'.format(len(set_)))
    tdf.to_csv(out_file, float_format='%.0f')

    outs = zip([counts, from_crop, to_crop, spei], ['counts', 'fc_ppi', 'tc_ppi', 'spei'])
    for d, js in outs:
        d = {'{}'.format(k): v for k, v in d.items()}
        out_js = os.path.join(out_matrix, '{}_{}.json'.format(js, glob))
        with open(out_js, 'w') as fp:
            json.dump(d, fp, indent=4)
    pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    met_cdl = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met4_ag3_fr8.npy'
    transistion = '/media/research/IrrigationGIS/expansion/analysis/transition'
    ppi_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/ppi_cdl_monthly.csv'
    # transition_probability(met_cdl, ppi_, transistion)

# ========================= EOF ====================================================================
