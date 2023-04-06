import json
import os
from multiprocessing import Pool

import numpy as np
import arviz as az
import pandas as pd
import pymc as pm
from scipy.special import softmax

from field_points.crop_codes import cdl_key
from transition_models import softmax_regression


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
        model_file = os.path.join(model_dir, '{}_{}.nc'.format(glob, fc))

        # if os.path.exists(model_file):
        #     continue

        d['x'] = np.array(d['x'])
        d['y'] = np.array(d['y'])

        print('\n\nmodeling {}: {}     {} samples'.format(fc, cdl[int(fc)][0], d['x'].shape[0]))
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
        df = az.summary(trace, hdi_prob=0.95)
        reind, counts, labels, coeff, crop_name = [], [], [], [], []
        for ct, label in zip(d['counts'], d['labels']):
            reind.append('a[{}]'.format(label))
            coeff.append('a')
            reind.append('b[climate, {}]'.format(label))
            coeff.append('climate')
            reind.append('b[from_price, {}]'.format(label))
            coeff.append('from_price')
            reind.append('b[to_price, {}]'.format(label))
            coeff.append('to_price')
            counts.append(int(ct))
            labels.append(label)
            crop_str = cdl[int(label)][0]
            crop_name.append(crop_str)

        counts = np.array(counts).repeat(4)
        crop_name = np.array(crop_name).repeat(4)
        labels = np.array(labels).repeat(4)
        df = df.reindex(reind)
        df['coeff'] = coeff
        df['counts'] = counts
        df['label'] = labels
        df['crop'] = crop_name
        df = df[['label', 'coeff', 'mean', 'sd', 'counts', 'hdi_2.5%', 'hdi_97.5%', 'crop']]
        df.to_csv(ofile)
        print(ofile)


def test_on_irrmapper():
    irr = '/media/research/IrrigationGIS/irrmapper/EE_extracts/concatenated/state/AZ_22NOV2021.csv'
    df = pd.read_csv(irr).sample(n=10000)
    y = df['POINT_TYPE'].values
    labels, counts = np.unique(y, return_counts=True)
    df['rnd'] = np.random.rand(df.shape[0]) + 10
    x = df[['nd_mean_gs', 'B10_gs', 'rnd']]
    x = np.subtract(x, x.mean(axis=0)).divide(x.std(axis=0))

    x_test, x, y_test, y = x[:3000].values, x[3000:].values, y[:3000], y[3000:]

    features = ['nd', 'lst', 'rnd']
    d = {'x': x, 'y': y, 'features': features, 'counts': counts, 'labels': labels}

    y_test_p = pd.get_dummies(y_test).values.T

    def dev(true, pred):
        loglik = np.sum(true * np.log(pred))
        rdev = -2 * loglik
        return rdev

    null = dev(y_test_p, np.mean(y_test_p, axis=0))

    model_file_ = 'irr_model.nc'
    if not os.path.exists(model_file_):
        softmax_regression(d, model_file_, cores=4)
    trace = az.from_netcdf(model_file_)

    summary = az.summary(trace)
    a = summary['mean'][:4].values.reshape(4, 1)
    b = summary['mean'][4:].values.reshape((len(counts), len(features)))

    # full
    z = a + np.dot(b, x_test.T)
    p = pm.math.softmax(z, axis=0).eval()
    full = dev(y_test_p, p)

    # climate-only
    z = a + np.dot(b[:, 0, np.newaxis], x_test[:, 0, np.newaxis].T)
    p = pm.math.softmax(z, axis=0).eval()
    climate = dev(y_test_p, p)

    z = a + np.dot(b[:, 1, np.newaxis], x_test[:, 1, np.newaxis].T)
    p = pm.math.softmax(z, axis=0).eval()
    fprice = dev(y_test_p, p)

    z = a + np.dot(b[:, 2, np.newaxis], x_test[:, 2, np.newaxis].T)
    p = pm.math.softmax(z, axis=0).eval()
    tprice = dev(y_test_p, p)


if __name__ == '__main__':

    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    sample = 10000
    if not os.path.exists(root):
        sample = None
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')

    transitions_ = os.path.join(root, 'analysis/transition')

    glob_ = 'model_sft'
    mglob_ = 'model_sft_{}.nc'
    cglob_ = 'model_sft_{}.csv'

    from_price_ = os.path.join(transitions_, 'fprice.json')
    to_price_ = os.path.join(transitions_, 'tprice.json')
    climate_ = os.path.join(transitions_, 'spei.json')
    glob = 'model_sft'

    model_dir = os.path.join(transitions_, 'models')
    sample_data = os.path.join(transitions_, 'sample_data')

    if sample:
        glb = 'sample_{}'.format(sample)
    else:
        glb = 'sample'.format(sample)

    model_dir = os.path.join(transitions_, 'models')
    summaries = os.path.join(transitions_, 'summaries')
    sample_data_ = os.path.join(transitions_, 'sample_data', '{}.json'.format(glb))

    # data_to_json(climate_, from_price_, to_price_, sample_data, samples=sample, glob=glb)
    # run_model(sample_data_, glob=glob_, model_dir=model_dir, cores=4)
    # multiproc_model(sample_data_, model_dst=model_dir, multiproc=30, glob=glob_)
    # summarize_pymc_model(model_dir, summaries, sample_data_)

    test_on_irrmapper()
# ========================= EOF ====================================================================
