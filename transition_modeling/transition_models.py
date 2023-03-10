import os
import pickle

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.linear_model import LogisticRegression

from transition_data import load_data

DEFAULTS = {'draws': 1000,
            'tune': 1000,
            'chains': 4,
            'cores': 1}

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]


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


def dirichlet_regression(y, x, from_crop, save_model=None):
    with pm.Model() as dmr_model:
        beta_0 = pm.Normal('climate', mu=1, sigma=3)
        beta_1 = pm.Normal('from_crop', mu=1, sigma=3)
        beta_2 = pm.Normal('to_crop', mu=1, sigma=3)

        alpha = pm.Normal('alpha', mu=1, sigma=3)

        theta = alpha + beta_0 * x[:, 0] + beta_1 * x[:, 1] + beta_2 * x[:, 2]

        p = pm.Deterministic('p', pm.math.exp(theta))

        obs = pm.DirichletMultinomial('obs', n=y, a=p, shape=y.shape[0])

        trace = pm.sample(**DEFAULTS)
        # trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

        model_file = save_model.format(from_crop)

        az.to_netcdf(trace, model_file)

        print(model_file)


def run_dirichlet(climate, from_price, to_price, save_model=None):
    for fc in KEYS:
        print('\nTransition from Crop: {}\n\n'.format(fc))
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price, from_crop=fc, n_samples=100)
        s = pd.Series(labels)
        y = pd.get_dummies(s).values
        y = y.sum(axis=0)
        x = np.array([c_data, fp_data, tp_data]).T
        dirichlet_regression(y, x, 24, save_model)


def summarize_pymc_model(saved_model, crop=1):
    model_file = saved_model.format(crop)
    with open(model_file, 'rb') as buff:
        mdata = pickle.load(buff)
        trace = mdata['trace']
    summary = az.summary(trace, hdi_prob=0.95)
    pass


if __name__ == '__main__':
    transitions_ = '/media/research/IrrigationGIS/expansion/analysis/transition'

    fp = os.path.join(transitions_, 'fprice.json')
    tp = os.path.join(transitions_, 'tprice.json')
    c = os.path.join(transitions_, 'spei.json')
    # logistic_regression(c, fp, tp, from_crop=24)

    model_dst = os.path.join(transitions_, 'model_sft_{}.pkl')
    run_dirichlet(c, fp, tp, save_model=model_dst)

    # summarize_pymc_model(model_dst, crop=24)
# ========================= EOF ====================================================================
