import os
import pickle

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

from transition_data import load_data

DEFAULTS = {'draws': 1000,
            'tune': 1000,
            'chains': 4}

np.random.seed(1234)

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]


def dirichlet_regression(y, x, from_crop, save_model=None, cores=1):
    """

    :param y: one-hot encoded class for each observation, shape (n_observations, k classes)
    :param x: feature data for each observation, shape (n_observations, n_features)
    :param from_crop: crop for which model is fit
    :param save_model:
    :param cores:
    :return:
    """

    DEFAULTS.update({'cores': cores})

    # feature count
    n_feat = x.shape[1]

    # class count
    k = y.shape[1]

    # total counts in each replicate
    n = y.shape[0]

    # total counts of each class outcome
    total_counts = y.sum(axis=0)

    prior_shape = (k, n_feat)

    with pm.Model() as dmr_model:
        # order: [climate, from_crop_price, to_crop_price]
        coeff = pm.Normal('coeff', mu=0.1, sigma=3, shape=prior_shape)

        theta = pm.math.dot(coeff, x.T)

        alpha = pm.Deterministic('alpha', pm.math.exp(theta))

        counts = pm.DirichletMultinomial('counts', n=total_counts, a=alpha, shape=(k, n), observed=y.T)

        pm.model_to_graphviz(dmr_model).view()

        trace = pm.sample(**DEFAULTS)

        model_file = save_model.format(from_crop)

        az.to_netcdf(trace, model_file)

        print(model_file)


def run_dirichlet(climate, from_price, to_price, save_model=None, cores=4):
    for fc in KEYS:
        print('\nTransition from Crop: {}\n\n'.format(fc))
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                           from_crop=fc, n_samples=100)
        s = pd.Series(labels)
        y = pd.get_dummies(s).values
        x = np.array([c_data, fp_data, tp_data]).T
        dirichlet_regression(y, x, fc, save_model, cores=cores)


def summarize_pymc_model(saved_model, crop=1):
    model_file = saved_model.format(crop)
    trace = az.from_netcdf(model_file)
    summary = az.summary(trace, hdi_prob=0.95)
    pass


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    transitions_ = os.path.join(root, 'expansion/analysis/transition')

    fp = os.path.join(transitions_, 'fprice.json')
    tp = os.path.join(transitions_, 'tprice.json')
    c = os.path.join(transitions_, 'spei.json')
    # logistic_regression(c, fp, tp, from_crop=24)

    model_dst = os.path.join(transitions_, 'model_sft_{}.pkl')
    # run_dirichlet(c, fp, tp, save_model=model_dst)

    x_ = np.random.random((100, 3))
    y_ = np.eye(7)[np.random.choice(7, 100)].astype(np.uint8)
    # dirichlet_regression(y_, x_, 21, model_dst, cores=4)

    summarize_pymc_model(model_dst, crop=1)
# ========================= EOF ====================================================================
