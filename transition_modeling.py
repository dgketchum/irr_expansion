import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

from utils.cdl import cdl_key


def transition_probability(cdl_npy, prices, out_matrix, files_js):

    with open(files_js, 'r') as fp:
        dct = json.load(fp)

    prices = {}
    for code, file_ in dct.items():
        df = pd.read_csv(file_, index_col=0)
        df = df.mean(axis=1)
        df = df.to_dict()
        prices[code] = df.copy()

    cdl = cdl_key()
    time_series_length = 17
    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    set_, classes = list(df.columns), [cdl[c][0] for c in df.columns]

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = [(int(a), int(b)) for a, b in keys]
    rec = np.moveaxis(rec, 0, 1)

    spei = {k: [] for k in keys}
    price = {k: [] for k in keys}

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

            spei[trans].append(c[1])

            y = e + 2005
            price_lag_str = '{}-07-01'.format(y - 1)
            price[trans].append(df)
            ct += 1

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

    outs = zip([price, spei], ['price', 'spei'])
    for d, files_js in outs:
        d = {'{}'.format(k): v for k, v in d.items()}
        out_js = os.path.join(out_matrix, '{}_{}.json'.format(files_js, glob))
        with open(out_js, 'w') as fp:
            json.dump(d, fp, indent=4)
    pass


if __name__ == '__main__':
    met_cdl = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met4_ag3_fr8.npy'
    transistion = '/media/research/IrrigationGIS/expansion/analysis/transition'
    ppi_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/ppi_cdl_monthly.csv'
    files_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/price_files.json'

    transition_probability(met_cdl, ppi_, transistion, files_)
# ========================= EOF ====================================================================
