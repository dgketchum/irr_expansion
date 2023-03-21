import os

import arviz as az
import numpy as np
import pymc as pm
import pandas as pd

DEFAULTS = {'draws': 1000,
            'tune': 2000,
            'chains': 4,
            'progressbar': True}


def dirichlet_regression(y, x, save_model=None, cores=4):
    if cores == 1:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

    x = np.array(x)
    y = np.array(y)

    # feature count
    n_feat = x.shape[1]
    n = x.shape[0]
    # y = pd.get_dummies(y).values

    # class count
    k = y.shape[1]

    coeff = pm.Normal.dist(mu=0, sigma=3, shape=(n_feat, k)).eval()
    theta = pm.math.dot(x, coeff).eval()
    alpha = pm.math.exp(theta).eval()
    counts = pm.DirichletMultinomial.dist(n=1, a=alpha, shape=(n, k)).eval()

    with pm.Model() as dmr_model:
        coeff = pm.Normal('coeff', mu=0, sigma=3, shape=(n_feat, k))

        theta = pm.math.dot(x, coeff)

        alpha = pm.math.exp(theta)

        obs = pm.DirichletMultinomial('obs', n=1, a=alpha, shape=(n, k), observed=y)

        DEFAULTS.update({'trace': [coeff]})
        DEFAULTS.update({'cores': cores})

        trace = pm.sample(**DEFAULTS)

        if save_model:
            az.to_netcdf(trace, save_model)
            print(save_model)


def softmax_regression(y, x, save_model=None, cores=4):
    """

    :param y: one-hot encoded class for each observation, shape (n_observations, k classes)
    :param x: feature data for each observation, shape (n_observations, n_features)
    :param save_model:
    :param cores:
    :return:
    """

    if cores == 1:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

    DEFAULTS.update({'cores': cores})

    x = np.array(x)
    y = np.array(y)

    # feature count
    n_feat = x.shape[1]

    # class count
    k = len(np.unique(y))

    with pm.Model() as sft_model:
        # order: [climate, from_crop_price, to_crop_price]
        a = pm.Normal('a', mu=0, sigma=0.5, shape=(k - 1,))
        a_f = pm.math.concatenate([[0], a])
        coeff = pm.Normal('coeff', mu=0, sigma=0.5, shape=(n_feat, k - 1))
        coeff_f = pm.math.concatenate([np.zeros((n_feat, 1)), coeff], axis=1)

        theta = a_f + pm.math.dot(x, coeff_f)

        z = pm.math.softmax(theta)

        obs = pm.Categorical('obs', p=z, observed=y)

        trace = pm.sample(**DEFAULTS)

        summary = az.summary(trace, hdi_prob=0.95)
        coeff_rows = [i for i in summary.index if 'coeff' in i]
        df = summary.loc[coeff_rows]

        if save_model:
            az.to_netcdf(trace, save_model)
            print(save_model)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
