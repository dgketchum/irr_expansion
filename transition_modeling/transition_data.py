import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta as reldt

from field_points.crop_codes import cdl_key

DEFAULTS = {'draws': 1000,
            'tune': 1000}

OLD_KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 49, 53]


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
            one_hot = pd.get_dummies(s).values
            y = [x.item() for x in np.argmax(one_hot, axis=1)]
            labels = [str(l) for l in list(set(s))]
            print(s.shape[0], 'observations of', fc)

            feature_order = ['climate', 'from_price', 'to_price']
            x = np.array([c_data, fp_data, tp_data]).T
            x = (x - x.mean(axis=0)) / x.std(axis=0)

            dct[fc] = {'x': x.tolist(),
                       'y': list(y),
                       'features': feature_order,
                       'labels': labels,
                       'counts': list([str(c) for c in one_hot.sum(axis=0)])}
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
    params = 5
    rec = np.fromfile(cdl_npy, dtype=float).reshape((params, -1, time_series_length))
    set_, classes = [x for x in KEYS], [cdl[c][0] for c in KEYS]
    prices = {k: v for k, v in prices.items() if k in KEYS}

    set_ = [s for s in set_ if s > 0]
    spei = {k: [] for k in set_}
    tprice = {k: [] for k in set_}

    rec = np.moveaxis(rec, 0, 1)
    ct = 0

    for i, v in enumerate(rec):
        for e, c in enumerate(v.T):

            if e < 2:
                continue

            tc = int(c[3])

            if tc not in set_:
                continue

            y = e + 2005
            tl = timescale['{}_{}'.format(tc, tc)]['tlag']
            ref_time = pd.to_datetime('{}-05-01'.format(y))
            tprice_lag_str = (ref_time - reldt(months=tl)).strftime('%Y-%m-%d')
            try:
                price = prices[tc][tprice_lag_str]
            except KeyError:
                continue
            tprice[tc].append(price)
            spei[tc].append(c[0])

            ct += 1

        if i % 10000 == 0.:
            print(i)

    print('{} transtions'.format(ct))

    outs = zip([tprice, spei], ['tprice', 'spei'])
    for d, price_files in outs:
        d = {'{}'.format(k): v for k, v in d.items()}
        out_js = os.path.join(out_matrix, '{}.json'.format(price_files))
        with open(out_js, 'w') as fp:
            json.dump(d, fp, indent=4)
    pass


if __name__ == '__main__':
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    samples_ = 1000
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')
        samples_ = 10000

    mode_ = 'uv'
    transitions_ = os.path.join(root, 'analysis/transition/uv_price')
    model_dir_ = os.path.join(transitions_, 'models')
    met_cdl = '/media/nvm/field_pts/fields_data/cdl_spei/cdl_spei.npy'
    files_ = os.path.join(root, 'tables/crop_value/price_files.json')
    response_timescale_ = os.path.join(root, 'analysis/transition/{}_price/time_scales.json'.format(mode_))
    crop_transitions(met_cdl, files_, response_timescale_, transitions_)

    to_price_ = os.path.join(transitions_, 'tprice.json')
    climate_ = os.path.join(transitions_, 'spei.json')

# ========================= EOF ====================================================================
