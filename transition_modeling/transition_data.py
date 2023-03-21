import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta as reldt

# from utils.cdl import cdl_key

DEFAULTS = {'draws': 1000,
            'tune': 1000}

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]


def _open_js(file_):
    with open(file_, 'r') as fp:
        dct = json.load(fp)
    return dct


def data_to_json(climate, from_price, to_price, labels_dir, samples, glob):
    label_file = os.path.join(labels_dir, '{}.json'.format(glob))
    dct = {}
    for fc in KEYS:
        try:
            print('\nTransition from Crop: {}'.format(fc))
            c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                               from_crop=fc, n_samples=samples)
            s = pd.Series(labels)
            y = pd.get_dummies(s).values
            print(s.shape[0], 'observations of', fc)

            x = np.array([c_data, fp_data, tp_data]).T

            dct[fc] = {'keys': keys,
                       'x': x.tolist(),
                       'y': y.tolist(),
                       'counts': list([str(c) for c in y.sum(axis=0)]),
                       'labels': [t.split('_')[1] for t in keys]}
        except KeyError:
            print('KeyError, skipping', fc)

    with open(label_file, 'w') as fp:
        json.dump(dct, fp, indent=4)


def load_data(climate, to_price, from_price, from_crop, n_samples=None):
    climate = _open_js(climate)
    to_price = _open_js(to_price)
    from_price = _open_js(from_price)

    crop_codes = sorted(set([int(k.split('_')[0]) for k in climate.keys()]))
    target_keys = ['{}_{}'.format(from_crop, to_crop) for to_crop in crop_codes]

    c_data = [i for l in [climate[k] for k in target_keys] for i in l]
    fp_data = [i for l in [from_price[k] for k in target_keys] for i in l]
    tp_data = [i for l in [to_price[k] for k in target_keys] for i in l]

    label_ct = [len(to_price[k]) for k in target_keys]
    labels = [[code for _ in range(length)] for code, length in zip(crop_codes, label_ct)]
    labels = [i for l in labels for i in l]

    combo = list(zip(c_data, fp_data, tp_data, labels))
    len_ = len(combo)
    combo = [c for c in combo if np.all(np.isfinite(c))]
    print('dropped {} of {}'.format(len_ - len(combo), len_))
    random.shuffle(combo)
    if n_samples:
        c_data[:], fp_data[:], tp_data[:], labels[:] = zip(*combo[:n_samples])
    else:
        c_data[:], fp_data[:], tp_data[:], labels[:] = zip(*combo)

    return c_data, fp_data, tp_data, labels, target_keys


def crop_transitions(cdl_npy, price_files, response_timescale, out_matrix):
    with open(price_files, 'r') as fp:
        dct = json.load(fp)

    prices = {}
    for code, file_ in dct.items():
        df = pd.read_csv(file_, index_col=0)
        df = df.mean(axis=1)
        df = df.to_dict()
        prices[int(code)] = df.copy()

    with open(response_timescale, 'r') as fp:
        timescale = json.load(fp)

    cdl = cdl_key()
    time_series_length = 17
    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    set_, classes = [x for x in prices.keys()], [cdl[c][0] for c in prices.keys()]

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = ['{}_{}'.format(a, b) for a, b in keys]

    spei = {k: [] for k in keys}
    fprice = {k: [] for k in keys}
    tprice = {k: [] for k in keys}

    rec = np.moveaxis(rec, 0, 1)
    ct = 0

    for i, v in enumerate(rec):
        for e, c in enumerate(v.T):

            if e == 16 or e < 2:
                continue

            fc, tc = int(c[3]), int(v.T[e + 1, 3])

            if not (fc in set_ and tc in set_):
                continue

            trans = '{}_{}'.format(fc, tc)
            fl, tl = timescale[trans]['flag'], timescale[trans]['tlag']
            y = e + 2005
            ref_time = pd.to_datetime('{}-05-01'.format(y))
            fprice_lag_str = (ref_time - reldt(months=fl)).strftime('%Y-%m-%d')
            tprice_lag_str = (ref_time - reldt(months=tl)).strftime('%Y-%m-%d')
            fprice[trans].append(prices[fc][fprice_lag_str])
            tprice[trans].append(prices[tc][tprice_lag_str])

            spei[trans].append(c[1])

            ct += 1

        if i % 10000 == 0.:
            print(i)

    print('{} transtions'.format(ct))

    outs = zip([fprice, tprice, spei], ['fprice', 'tprice', 'spei'])
    for d, price_files in outs:
        d = {'{}'.format(k): v for k, v in d.items()}
        out_js = os.path.join(out_matrix, '{}.json'.format(price_files))
        with open(out_js, 'w') as fp:
            json.dump(d, fp, indent=4)
    pass


if __name__ == '__main__':
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')

    transitions_ = os.path.join(root, 'analysis/transition')
    model_dir_ = os.path.join(transitions_, 'models')

    from_price_ = os.path.join(transitions_, 'fprice.json')
    to_price_ = os.path.join(transitions_, 'tprice.json')
    climate_ = os.path.join(transitions_, 'spei.json')
    glob = 'model_sft'

    model_dir = os.path.join(transitions_, 'models')
    sample_data = os.path.join(transitions_, 'sample_data')

    samples_ = 1000
    data_to_json(climate_, from_price_, to_price_, sample_data, samples=samples_, glob='sample_{}'.format(samples_))

# ========================= EOF ====================================================================
