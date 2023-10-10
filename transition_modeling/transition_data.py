import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta as reldt
import matplotlib.pyplot as plt

from field_points.crop_codes import cdl_key

DEFAULTS = {'draws': 1000,
            'tune': 1000}

OLD_KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 49, 53]


def _open_js(file_):
    with open(file_, 'r') as fp:
        dct = json.load(fp)
    return dct


def data_to_json(climate, from_price, to_price, label_file, samples):
    dct = {}
    for fc in KEYS:

        print('\nTransition from Crop: {}'.format(fc))
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                           from_crop=fc, n_samples=samples)
        s = pd.Series(labels)
        labels, counts = np.unique(s, return_counts=True)
        label_map = {int(l): i for i, l in enumerate(labels)}
        counts = {int(l_): int(ct) for l_, ct in zip(labels, counts)}
        y = s.apply(lambda l: label_map[l])
        print(s.shape[0], 'observations of', fc)

        feature_order = ['climate', 'from_price', 'to_price']
        x = np.array([c_data, fp_data, tp_data]).T
        x = (x - x.mean(axis=0)) / x.std(axis=0)

        dct[fc] = {'x': x.tolist(),
                   'y': list(y),
                   'features': feature_order,
                   'labels': [label_map[l_] for l_ in labels],
                   'label_map': label_map,
                   'counts': [counts[l_] for l_ in labels],
                   'counts_map': counts}

    with open(label_file, 'w') as fp:
        json.dump(dct, fp, indent=4)
        print(label_file)


def load_data(climate, from_price, to_price, from_crop, n_samples=None):
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
    params = 5
    rec = np.fromfile(cdl_npy, dtype=float).reshape((params, -1, time_series_length))
    set_, classes = [x for x in KEYS], [cdl[c][0] for c in KEYS]
    prices = {k: v for k, v in prices.items() if k in KEYS}

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = ['{}_{}'.format(a, b) for a, b in keys]

    spei = {k: [] for k in keys}
    tprice = {k: [] for k in keys}
    fprice = {k: [] for k in keys}

    rec = np.moveaxis(rec, 0, 1)
    ct = 0

    for i, v in enumerate(rec):
        for e, c in enumerate(v.T):

            if e == 16 or e < 2:
                continue

            fc, tc = int(v.T[e - 1, 2]), int(c[2])

            if not (fc in set_ and tc in set_):
                continue

            trans = '{}_{}'.format(fc, tc)
            tl = timescale['{}_{}'.format(tc, tc)]['tlag']
            fl = timescale['{}_{}'.format(fc, fc)]['tlag']
            y = e + 2005
            ref_time = pd.to_datetime('{}-05-01'.format(y))
            tprice_lag_str = (ref_time - reldt(months=tl)).strftime('%Y-%m-%d')
            fprice_lag_str = (ref_time - reldt(months=fl)).strftime('%Y-%m-%d')

            try:
                tprice_ = prices[tc][tprice_lag_str]
                fprice_ = prices[fc][fprice_lag_str]
            except KeyError:
                continue

            tprice[trans].append(tprice_)
            fprice[trans].append(fprice_)
            spei[trans].append(c[0])

            ct += 1

        if i % 10000 == 0. and i > 1:
            print(i)

    print('{} transtions'.format(ct))

    outs = zip([fprice, tprice, spei], ['fprice', 'tprice', 'spei'])
    for d, price_files in outs:
        dc = {'{}'.format(k): v for k, v in d.items()}
        out_js = os.path.join(out_matrix, '{}.json'.format(price_files))
        with open(out_js, 'w') as fp:
            json.dump(dc, fp, indent=4)
    pass


def climate_price_correlation(sample_data):
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    cdl = cdl_key()

    for fc, d in data.items():
        crop = cdl[int(fc)][0]
        inv_dict = {v: k for k, v in d['label_map'].items()}
        x = np.array(d['x'])
        y = [1 if int(inv_dict[e]) == int(fc) else 0 for e in d['y']]
        mask = [i for i, e in enumerate(y) if e == 1]
        x = x[mask, :]
        climate, price = x[:, 0], x[:, 1]
        corr = np.corrcoef(climate, price)[0, 1]
        plt.scatter(climate, price)
        plt.show()
        print('Correlation for {}: {:.3f}'.format(crop, corr))


if __name__ == '__main__':
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')

    mode_ = 'uv'
    transitions_ = os.path.join(root, 'analysis/transition/uv_price')
    model_dir_ = os.path.join(transitions_, 'models')
    met_cdl = '/media/nvm/field_pts/fields_data/cdl_spei/cdl_spei.npy'
    files_ = os.path.join(root, 'tables/crop_value/price_files.json')
    response_timescale_ = os.path.join(root, 'analysis/transition/{}_price/time_scales.json'.format(mode_))
    # crop_transitions(met_cdl, files_, response_timescale_, transitions_)

    sample = 50000
    glb = 'sample_{}'.format(sample)
    transitions_ = os.path.join(root, 'analysis/transition')
    sample_data_ = os.path.join(transitions_, 'sample_data', '{}.json'.format(glb))
    climate_price_correlation(sample_data_)

# ========================= EOF ====================================================================
