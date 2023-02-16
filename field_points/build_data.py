import os
import json
from itertools import product
from pprint import pprint
from multiprocessing import Pool

import numpy as np
import pandas as pd

from call_ee import BASIN_STATES
from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']

IDX_KWARGS = dict(distribution=indices.Distribution.gamma,
                  data_start_year=1987,
                  calibration_year_initial=1987,
                  calibration_year_final=2009,
                  periodicity=compute.Periodicity.monthly)


def concatenate_field_data(csv_dir, metadata, out, file_check=False, glob=None):
    for s in BASIN_STATES:
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df.sort_index()
        dt_range = ['{}-{}-01'.format(y, m) for y in range(1984, 2022) for m in range(1, 13)]
        data = np.zeros((df.shape[0], len(dt_range), len(COLS)))

        print('\n', s)

        file_list = [os.path.join(csv_dir, 'et',
                                  '{}_{}_{}_{}.csv'.format(glob, s, y, m)) if m in range(4, 11) and y > 1986
                     else os.path.join(csv_dir, 'met', '{}_{}_{}_{}.csv'.format(glob, s, y, m))
                     for y in range(1984, 2022)
                     for m in range(1, 13)]

        if file_check:
            missing = [f for f in file_list if not os.path.exists(f)]
            print('missing {} files in {}'.format(len(missing), s))
            pprint([os.path.basename(f) for f in missing])
            open('missing.txt', 'a').write('\n'.join(missing))
            continue

        for csv in file_list:
            splt = csv.split('.')[0].split('_')
            m, y = int(splt[-1]), int(splt[-2])
            dt_ind = dt_range.index('{}-{}-01'.format(y, m))
            c = pd.read_csv(csv, index_col='OPENET_ID')
            print(c.shape, csv)
            fill_ind = [i for i in df.index if i not in c.index]
            c = c.reindex(df.index)

            if m in range(4, 11) and y > 1986:
                _f = csv.replace('/et/', '/met/')
                gs_met = pd.read_csv(_f, index_col='OPENET_ID')
                c.loc[fill_ind, ['ppt', 'etr']] = gs_met.loc[fill_ind, ['ppt', 'etr']]
            else:
                c[['et', 'cc', 'eff_ppt', 'ietr']] = np.zeros((c.shape[0], 4))

            c = c[COLS]
            data[:, dt_ind, :] = c.values

        out_path = os.path.join(out, '{}.npy'.format(s))
        data.tofile(out_path)
        out_js = out_path.replace('.npy', '_index.json')
        with open(out_js, 'w') as fp:
            json.dump({'index': list(df.index)}, fp, indent=4)
        print(out_path)


def write_indices(arg):
    state, id_, od = arg
    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    ag_periods = list(range(1, 8))
    periods = list(product(met_periods, ag_periods))

    npy = os.path.join(id_, '{}.npy'.format(state))
    print(npy)
    js = npy.replace('.npy', '_index.json')
    data = np.fromfile(npy, dtype=float)

    with open(js, 'r') as fp:
        index = json.load(fp)['index']

    data = data.reshape((len(index), -1, len(COLS)))
    df = pd.DataFrame(index=index)
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(1987, 2022) for m in range(1, 13)]
    months = np.multiply(np.ones((len(index), len(dt_range))), np.array([dt.month for dt in dt_range]))

    for met_p, ag_p in periods:

        kc = data[:, :, COLS.index('et')] / data[:, :, COLS.index('ietr')]
        simi = np.apply_along_axis(lambda x: indices.spi(x, scale=ag_p, **IDX_KWARGS), arr=kc, axis=1)

        # uses locally modified climate_indices package that takes cwb = ppt - pet as input
        cwb = data[:, :, COLS.index('ppt')] - data[:, :, COLS.index('etr')]
        spei = np.apply_along_axis(lambda x: indices.spei(x, scale=met_p, **IDX_KWARGS), arr=cwb, axis=1)

        stack = np.stack([simi[:, -len(dt_range):], spei[:, -len(dt_range):], months])

        for m in range(4, 11):
            if m - ag_p < 3:
                continue
            d = stack[:, stack[2] == float(m)].reshape((3, len(index), -1))
            mx = np.ma.masked_array(np.repeat(np.isnan(d[:1, :, :]), 3, axis=0))
            d = np.ma.MaskedArray(d, mx)
            coref = [np.ma.corrcoef(d[0, i, :], d[1, i, :])[0][1].item() ** 2 for i in range(d.shape[1])]
            col = 'met{}_ag{}_fr{}'.format(met_p, ag_p, m)
            df[col] = coref
            print(state, col)

    ofile = os.path.join(od, '{}.csv'.format(state))
    df.to_csv(ofile)
    return ofile


def main():
    args = [(s, fpd, ind_) for s in BASIN_STATES]
    with Pool(processes=11) as pool:
        result = pool.map(write_indices, args)
        pool.close()
        pool.join()
        print(result)


if __name__ == '__main__':
    root = '/media/nvm'
    if not os.path.exists(root):
        root = '/home/dgketchum/data'

    csv_ = os.path.join(root, 'field_pts/csv')
    fpd = os.path.join(root, 'field_pts/field_pts_data')
    meta_ = os.path.join(root, 'field_pts/usbr_attr/')
    # concatenate_field_data(csv_, meta_, fpd, glob='ietr_fields_13FEB2023', file_check=False)

    ind_ = os.path.join(root, 'field_pts/indices')
    main()
# ========================= EOF ====================================================================
