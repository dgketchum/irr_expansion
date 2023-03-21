import json
import os
from multiprocessing import Pool

import arviz as az
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from transition_data import load_data
from transition_models import softmax_regression, dirichlet_regression

from utils.cdl import cdl_key


def multiproc_model(sample_data, multiproc=20, glob=None):
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    if multiproc:
        pool = Pool(multiproc)
        cores = 1
    else:
        cores = 1

    for fc, d in data.items():

        print('starting', fc)
        model_file = os.path.join(model_dir, '{}_{}.nc'.format(glob, fc))

        if not multiproc:
            softmax_regression(d['y'], d['x'], model_file, cores=cores)
        else:
            pool.apply_async(softmax_regression, args=(d['y'], d['x'], model_file, cores))

    if multiproc > 0:
        pool.close()
        pool.join()


def run_model(sample_data, model_dir=None, glob=None, cores=4):
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    for fc, d in data.items():

        if fc != '21':
            continue

        model_file = os.path.join(model_dir, '{}_{}.nc'.format(glob, fc))

        dirichlet_regression(d['y'], d['x'], model_file, cores=cores)


def save_coefficients(trace, ofile):
    summary = az.summary(trace, hdi_prob=0.95)
    coeff_rows = [i for i in summary.index if 'coeff' in i]
    df = summary.loc[coeff_rows]
    df.to_csv(ofile)


def summarize_pymc_model(saved_model, coeff_summary, mglob=None, cglob=None):
    cdl = cdl_key()
    l = [os.path.join(saved_model, x) for x in os.listdir(saved_model) if x.endswith('.nc')]
    k = [x.split('.')[0].split('_')[-1] for x in os.listdir(saved_model) if x.endswith('.nc')]
    for f, crop in zip(l, k):
        print(cdl[int(crop)][0])
        model_file = os.path.join(saved_model, mglob.format(crop))
        ofile = os.path.join(coeff_summary, cglob.format(crop))
        # if os.path.exists(ofile):
        #     print(ofile, 'exists, skipping')
        #     continue
        trace = az.from_netcdf(model_file)
        summary = az.summary(trace, hdi_prob=0.95)
        coeff_rows = [i for i in summary.index if 'coeff' in i]
        df = summary.loc[coeff_rows]
        df.to_csv(ofile)
        print(ofile)


if __name__ == '__main__':

    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')
    transitions_ = os.path.join(root, 'analysis/transition')
    glob_ = 'model_sft'
    model_dir = os.path.join(transitions_, 'models', 'debug')
    mglob_ = 'model_sft_{}.nc'
    cglob_ = 'model_sft_{}.csv'
    sample_data_ = os.path.join(transitions_, 'sample_data', 'sample_1000.json')
    # run_model(sample_data_, glob=glob_, model_dir=model_dir, cores=4)

    sample_data_ = os.path.join(transitions_, 'sample_data', 'sample_10000.json')
    # multiproc_model(sample_data_, multiproc=0, glob=glob_)

    old_coeffs = os.path.join(transitions_, 'coefficients')
    # summarize_pymc_model(model_dir, model_dir, mglob_, cglob_)

    from sklearn import datasets
    x, y = datasets.load_iris(return_X_y=True)
    # softmax_regression(y, x, cores=4)

    irr = '/media/research/IrrigationGIS/irrmapper/EE_extracts/concatenated/state/AZ_22NOV2021.csv'
    df = pd.read_csv(irr).sample(n=10000)
    y = df['POINT_TYPE'].values
    x = df[['nd_mean_gs', 'pet_tot_spr', 'B4_2']]
    x = np.subtract(x, x.mean(axis=0)).divide(x.std(axis=0))
    softmax_regression(y, x, cores=4)

# ========================= EOF ====================================================================
