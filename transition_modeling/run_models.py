import json
import os
from multiprocessing import Pool
from pprint import pprint

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import log_loss

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


def run_model(sample_data, model_dir=None, glob=None, cores=4, overwrite=False):
    cdl = cdl_key()
    with open(sample_data, 'r') as fp:
        data = json.load(fp)

    for fc, d in data.items():

        model_file = os.path.join(model_dir, '{}_{}.nc'.format(glob, fc))

        if os.path.exists(model_file) and not overwrite:
            print(model_file, 'exists')
            continue

        d['x'] = np.array(d['x'])
        d['y'] = np.array(d['y'])
        pprint(d['label_map'])
        pprint(d['counts_map'])

        print('\n\nmodeling {}: {}     {} samples'.format(fc, cdl[int(fc)][0], d['x'].shape[0]))
        softmax_regression(d, model_file, cores=cores)


def summarize_pymc_model(saved_model, coeff_summary, input_data, deviance_file):
    cdl = cdl_key()
    table_select = ['crop', 'label', 'crop_code', 'coeff', 'mean', 'sd', 'counts', 'dev']
    l = [os.path.join(saved_model, x) for x in os.listdir(saved_model) if x.endswith('.nc')]
    k = [x.split('.')[0].split('_')[-1] for x in os.listdir(saved_model) if x.endswith('.nc')]
    ct = 0
    mdf = pd.DataFrame()
    coeff_stack = []
    with open(input_data, 'r') as f_obj:
        stations = json.load(f_obj)

    devdf = pd.DataFrame(columns=['climate', 'from_price', 'to_price'], index=k)
    for f, crop in zip(l, k):
        d = stations[crop]
        print(cdl[int(crop)][0], 'Model')
        model_file = os.path.join(saved_model, 'model_sft_{}.nc'.format(crop))
        ofile = os.path.join(coeff_summary, 'model_sft_{}.csv'.format(crop))

        trace = az.from_netcdf(model_file)
        df = az.summary(trace, hdi_prob=0.95)

        if np.any(df['r_hat'] > 1.1):
            raise NotImplementedError

        def dev(true, pred):
            rdev = -2 * log_loss(true, pred)
            return rdev

        y_test_p = pd.get_dummies(d['y']).values.T
        x = np.array(d['x'])

        idx = len(d['counts'])
        a = df['mean'][:idx].values.reshape(idx, 1)
        b = df['mean'][idx:].values.reshape((idx, len(d['features'])))

        mean_ = y_test_p.mean(axis=1)
        mean_ = np.repeat(mean_[:, np.newaxis], y_test_p.shape[1], axis=1)
        null = dev(y_test_p, mean_)

        # full
        z = a + np.dot(b, x.T)
        p = pm.math.softmax(z, axis=0).eval()
        full = dev(y_test_p, p)

        deviances = {'a': None}
        coeffs = trace.posterior['b'].features.values
        for i, feat in enumerate(coeffs):
            z = a + np.dot(b[:, i, np.newaxis], x[:, i, np.newaxis].T)
            p = pm.math.softmax(z, axis=0).eval()
            d1 = dev(y_test_p, p)
            dev_ = (d1 - full) / null
            deviances[feat] = dev_

        print(['{} {:.3f}'.format(p, deviances[p]) for p in devdf.columns])

        devdf.loc[crop] = deviances
        reind, counts, labels, coeff, crop_name = [], [], [], [], []
        label_map = {v: k for k, v in d['label_map'].items()}
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
            crop_str = cdl[int(label_map[label])][0]
            crop_name.append(crop_str)

        counts = np.array(counts).repeat(4)
        crop_name = np.array(crop_name).repeat(4)
        labels = np.array(labels).repeat(4)
        df = df.reindex(reind)
        df['coeff'] = coeff
        df['counts'] = counts

        df['label'] = labels
        df['crop_code'] = [label_map[l_] for l_ in labels]
        df['crop'] = crop_name
        df['dev'] = df['coeff'].apply(lambda x: deviances[x])
        df = df[['label', 'crop_code', 'coeff', 'mean', 'sd', 'counts',
                 'hdi_2.5%', 'hdi_97.5%', 'crop', 'dev']]

        df.to_csv(ofile)
        for l in np.unique(labels):
            coeff_stack.append(df[df['label'] == l]['mean'].values)
        desc = df[table_select].T.index
        mdf_ = df[table_select].T
        mdf_cols = list(mdf_.columns)
        mdf_['desc'] = desc
        mdf_ = mdf_[['desc'] + mdf_cols]
        mdf = mdf.append(mdf_)
        mdf = mdf.append(pd.Series(), ignore_index=True)
        ct += 1
        print(ofile)

    cdf = pd.DataFrame(columns=['c', 'fp', 'tp'], data=np.array(coeff_stack)[:, 1:])
    mdf.to_csv(os.path.join(coeff_summary, 'coefficients.csv'))
    devdf.to_csv(deviance_file)


if __name__ == '__main__':

    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    sample = 50000
    if not os.path.exists(root):
        sample = None
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')

    transitions_ = os.path.join(root, 'analysis/transition')
    to_price_ = os.path.join(transitions_, 'uv_price', 'tprice.json')
    from_price_ = os.path.join(transitions_, 'uv_price', 'fprice.json')
    climate_ = os.path.join(transitions_, 'uv_price', 'spei.json')

    if sample:
        glb = 'sample_{}'.format(sample)
    else:
        glb = 'sample'.format(sample)

    sample_data_ = os.path.join(transitions_, 'sample_data', '{}.json'.format(glb))
    # data_to_json(climate_, from_price_, to_price_, sample_data_, samples=sample)

    glob = 'model_sft'
    model_dir_ = os.path.join(transitions_, 'models')
    # run_model(sample_data_, glob=glob, model_dir=model_dir_, cores=4)

    validation_data = os.path.join(transitions_, 'validation_data', '{}.json'.format(glb))
    # data_to_json(climate_, from_price_, to_price_, validation_data, samples=sample)

    dev_summary = os.path.join(transitions_, 'summaries', 'deviances.csv')
    coeff_summary_ = os.path.join(transitions_, 'summaries', 'coefficients.csv')
    summaries = os.path.join(transitions_, 'summaries')
    summarize_pymc_model(model_dir_, summaries, validation_data, dev_summary)
# ========================= EOF ====================================================================
