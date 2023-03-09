import json
import os
import pickle
import random
from itertools import combinations
from dateutil.relativedelta import relativedelta as reldt

import aesara
import aesara.tensor as tt
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc.sampling_jax
from sklearn.linear_model import LogisticRegression

from utils.cdl import cdl_key

DEFAULTS = {'draws': 1000,
            'tune': 1000}

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]

def _open_js(file_):
    with open(file_, 'r') as fp:
        dct = json.load(fp)
    return dct


def load_data(climate, to_price, from_price, from_crop, n_samples=100):
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


def logistic_regression(climate, from_price, to_price, from_crop=24, samples=10000):
    split = int(np.floor(0.7 * samples))
    c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                 from_crop=from_crop, n_samples=samples)

    x, y = np.array([c_data[:split], fp_data[:split], tp_data[:split]]).T, np.array(labels[:split])
    x_test, y_test = np.array([c_data[split:], fp_data[split:], tp_data[split:]]).T, np.array(labels[split:])

    clf = LogisticRegression().fit(x, y)
    pred = clf.predict(x_test)
    df = pd.DataFrame(columns=['pred', 'y'], data=np.array([pred, y_test]).T)
    df = df.loc[df['y'] != 24]
    df['correct'] = df[df['y'] == df['pred']]
    pass


def dirichlet_regression(climate, from_price, to_price, save_model=None):

    for fc in KEYS:
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price, from_crop=fc)
        s = pd.Series(labels)
        obs = pd.get_dummies(s).values
        # obs = obs.sum(axis=0)
        n, k = obs.shape

        climate = aesara.shared(np.array(c_data))
        fprice = aesara.shared(np.array(fp_data))
        tprice = aesara.shared(np.array(tp_data))

        with pm.Model() as dmr_model:
            a = pm.Normal('a', mu=0, sigma=1, shape=k)
            b = pm.Normal('b', mu=0, sigma=1, shape=k)
            b = pm.Normal('c', mu=0, sigma=1, shape=k)

            alpha = pm.Deterministic('alpha', pm.math.exp(a + tt.outer(tprice, b) + tt.outer(climate, c)))

            p = pm.Dirichlet('p', a=alpha, shape=(n, k))

            dm = pm.DirichletMultinomial('dm', 1, p, observed=obs)

            pm.model_to_graphviz(dmr_model).view()

            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model.format(fc), 'wb') as buff:
                    pickle.dump({'trace': trace}, buff)
                    print('saving', save_model)


def summarize_pymc_model(saved_model):
    with open(saved_model, 'rb') as buff:
        mdata = pickle.load(buff)
        trace = mdata['trace']
    summary = az.summary(trace, hdi_prob=0.95, var_names=['alpha'])
    pass


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

    met_cdl = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met4_ag3_fr8.npy'
    transitions_ = '/media/research/IrrigationGIS/expansion/analysis/transition'
    files_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/price_files.json'
    response_timescale_ = '/media/research/IrrigationGIS/expansion/analysis/transition/time_scales.json'

    crop_transitions(met_cdl, files_, response_timescale_, transitions_)

    fp = os.path.join(transitions_, 'fprice.json')
    tp = os.path.join(transitions_, 'tprice.json')
    c = os.path.join(transitions_, 'spei.json')
    model_dst = os.path.join(transitions_, 'model_{}.pkl')
    # dirichlet_regression(c, fp, tp, from_crop=target, save_model=model_dst)
    # logistic_regression(c, fp, tp, from_crop=target)

    # summarize_pymc_model(model_dst)
# ========================= EOF ====================================================================
