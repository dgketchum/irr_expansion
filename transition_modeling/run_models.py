import json
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import arviz as az
from transition_models import softmax_regression
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

        softmax_regression(d['y'], d['x'], model_file, cores=cores)


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
    run_model(sample_data_, glob=glob_, model_dir=model_dir, cores=4)
    #
    # multiproc_model(sample_data_, multiproc=5, glob=glob_)

    old_coeffs = os.path.join(transitions_, 'coefficients')
    # summarize_pymc_model(model_dir, model_dir, mglob_, cglob_)

# ========================= EOF ====================================================================
