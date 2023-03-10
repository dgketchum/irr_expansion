import os
from pprint import pprint
import multiprocessing as mp

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az

from transition_models import dirichlet_regression
from transition_data import load_data

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]


def chunk_data_(data, observed, n_samp, n_predictors):
    combined_data = np.column_stack((data, observed))
    num_chunks = int(len(combined_data) / n_samp) + 1
    data_chunks = np.array_split(combined_data, num_chunks)
    data_chunk_pairs = []
    for i in range(num_chunks):
        chunk_data = data_chunks[i][:, :n_predictors]
        chunk_observed = data_chunks[i][:, n_predictors:]
        data_chunk_pairs.append((chunk_data, chunk_observed))
    return data_chunk_pairs


def multiproc_pymc_model(climate, from_price, to_price, max_concurrent_models, model_dir, sample_size=5000):
    for fc in KEYS:
        print('\nTransition from Crop: {}\n\n'.format(fc))
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                           from_crop=fc)

        s = pd.Series(labels)
        y = pd.get_dummies(s).values
        predictors = [c_data, fp_data, tp_data]
        x = np.array(predictors).T
        data_chunks = chunk_data_(x, y, sample_size, n_predictors=len(predictors))

        model_paths, model_result = [], []
        pool = mp.Pool(processes=max_concurrent_models)

        for i in range(len(data_chunks)):
            model_name = f'model_{fc}_{i}.trace'
            model_path = os.path.join(model_dir, model_name)
            model_paths.append(model_path)
            r = pool.apply_async(dirichlet_regression, args=(data_chunks[i][1],
                                                             data_chunks[i][0],
                                                             fc, model_path))
            model_result.append(r)

        pool.close()
        pool.join()
        [r.get() for r in model_result]


def concatenate_models(model_dir):
    model_paths = []
    for fc in KEYS:
        print('\nTransition from Crop: {}\n\n'.format(fc))
        combined_model = pm.Model(name=f'combined_model_{fc}')
        combined_trace = None
        with combined_model:
            for i, model_name in enumerate(model_paths):
                trace = az.from_netcdf(model_name)
                if combined_trace is None:
                    combined_trace = trace
                else:
                    combined_trace = pm.backends.base.concat_traces([combined_trace, trace])

        for i in range(len(model_paths)):
            model_name = f'model_{i}.trace'
            model_path = os.path.join(model_dir, model_name)
            os.remove(model_path)

        combo = os.path.join(model_dir, f'combined_{fc}.trace')
        pm.save_trace(combined_trace, combo)


if __name__ == '__main__':

    root = os.path.join('/media', 'research', 'IrrigationGIS', 'expansion')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'expansion')

    transitions_ = os.path.join(root, 'analysis/transition')
    model_dir_ = os.path.join(transitions_, 'models')

    fp = os.path.join(transitions_, 'fprice.json')
    tp = os.path.join(transitions_, 'tprice.json')
    c = os.path.join(transitions_, 'spei.json')

    multiproc_pymc_model(c, fp, tp, 25, model_dir_, 5000)

# ========================= EOF ====================================================================
