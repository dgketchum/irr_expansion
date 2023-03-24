import json
import os
from multiprocessing import Pool

import numpy as np
import arviz as az

from transition_models import softmax_regression
from field_points.crop_codes import cdl_key


def multiproc_model(sample_data, model_dst, multiproc=20, glob=None):
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    if multiproc:
        pool = Pool(multiproc)
        cores = 1
    else:
        cores = 1

    for fc, d in data.items():

        print('starting', fc)
        model_file = os.path.join(model_dst, '{}_{}.nc'.format(glob, fc))

        d['x'] = np.array(d['x'])
        d['y'] = np.array(d['y'])

        if not multiproc:
            softmax_regression(d, model_file, cores=cores)
        else:
            pool.apply_async(softmax_regression, args=(d, model_file, cores))

    if multiproc > 0:
        pool.close()
        pool.join()


def run_model(sample_data, model_dir=None, glob=None, cores=4):
    cdl = cdl_key()
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    for fc, d in data.items():
        print('\n\nmodeling {}: {}'.format(fc, cdl[int(fc)][0]))

        model_file = os.path.join(model_dir, '{}_{}.nc'.format(glob, fc))

        d['x'] = np.array(d['x'])
        d['y'] = np.array(d['y'])

        softmax_regression(d, model_file, cores=cores)


def summarize_pymc_model(saved_model, coeff_summary, input_data):
    cdl = cdl_key()

    l = [os.path.join(saved_model, x) for x in os.listdir(saved_model) if x.endswith('.nc')]
    k = [x.split('.')[0].split('_')[-1] for x in os.listdir(saved_model) if x.endswith('.nc')]

    with open(input_data, 'r') as f_obj:
        stations = json.load(f_obj)

    for f, crop in zip(l, k):
        d = stations[crop]
        print(cdl[int(crop)][0])
        model_file = os.path.join(saved_model, 'model_sft_{}.nc'.format(crop))
        ofile = os.path.join(coeff_summary, 'model_sft_{}.csv'.format(crop))

        trace = az.from_netcdf(model_file)
        summary = az.summary(trace, hdi_prob=0.95)

        df['count'] = np.array([int(x) for x in d['counts']]).repeat(3)
        df = df[['mean', 'sd', 'count', 'hdi_2.5%', 'hdi_97.5%', 'r_hat']]
        df.to_csv(ofile)
        print(ofile)


if __name__ == '__main__':

    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    sample = 1000
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')
        sample = 10000

    glob_ = 'model_sft'
    mglob_ = 'model_sft_{}.nc'
    cglob_ = 'model_sft_{}.csv'

    transitions_ = os.path.join(root, 'analysis/transition')
    sample_data_ = os.path.join(transitions_, 'sample_data', 'sample_{}.json'.format(sample))
    model_dir = os.path.join(transitions_, 'models')

    run_model(sample_data_, glob=glob_, model_dir=model_dir, cores=4)
    # multiproc_model(sample_data_, model_dst=model_dir, multiproc=30, glob=glob_)

    # summarize_pymc_model(model_dir, model_dir, sample_data_)

# ========================= EOF ====================================================================
