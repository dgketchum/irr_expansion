import os
import multiprocessing as mp

import numpy as np
import pymc as pm
import pandas as pd

from transition_modeling.transition_modeling import dirichlet_regression, load_data

KEYS = [1, 12, 21, 23, 24, 28, 36, 37, 41, 42, 43, 47, 49, 53, 56, 57, 58, 59, 66, 68, 69, 71, 77]


def split_data_into_chunks(data, observed, n_samp):
    combined_data = np.column_stack((data, observed))
    num_chunks = int(len(combined_data) / n_samp) + 1
    data_chunks = np.array_split(combined_data, num_chunks)
    data_chunk_pairs = []
    for i in range(num_chunks):
        chunk_data = data_chunks[i][:, 0]
        chunk_observed = data_chunks[i][:, 1]
        data_chunk_pairs.append((chunk_data, chunk_observed))
    return data_chunk_pairs


def multiproc_pymc_model(climate, from_price, to_price, max_concurrent_models, sample_size=5000):
    for fc in KEYS:
        print('\nTransition from Crop: {}\n\n'.format(fc))
        c_data, fp_data, tp_data, labels, keys = load_data(climate, from_price, to_price,
                                                           from_crop=fc, n_samples=sample_size)

        s = pd.Series(labels)
        y = pd.get_dummies(s).values
        y = y.sum(axis=0)
        x = np.array([climate, from_price, to_price]).T

        data_chunks = split_data_into_chunks(x, y, sample_size)
        pool = mp.Pool(max_concurrent_models)
        model_names = []
        for i in range(len(data_chunks)):
            while len(model_names) >= max_concurrent_models:
                for j in range(len(model_names)):
                    if not os.path.exists(model_names[j] + ".trace"):
                        model_names.pop(j)
                        break
            model_name = f'model{i}'
            result = pool.apply_async(dirichlet_regression, args=(model_name, data_chunks[i]))
            model_names.append(result)

        pool.close()
        pool.join()

        combined_model = pm.Model(name='combined_model')
        combined_trace = None
        with combined_model:
            for i in range(len(data_chunks)):
                model_name = f'model{i}'
                trace = pm.load_trace(model_name + ".trace", model_name)
                if combined_trace is None:
                    combined_trace = trace
                else:
                    combined_trace = pm.backends.base.concat_traces([combined_trace, trace])

        for i in range(len(data_chunks)):
            model_name = f'model{i}'
            os.remove(model_name + ".trace")


if __name__ == '__main__':
    transitions_ = '/media/research/IrrigationGIS/expansion/analysis/transition'
    model_dir = os.path.join(transitions_, 'models')

    fp = os.path.join(transitions_, 'fprice.json')
    tp = os.path.join(transitions_, 'tprice.json')
    c = os.path.join(transitions_, 'spei.json')

    multiproc_pymc_model(c, fp, tp, 25, 5000)

# ========================= EOF ====================================================================
