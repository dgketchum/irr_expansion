import os
import json
from itertools import product
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import linregress as lr

from call_ee import BASIN_STATES
from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']

IDX_KWARGS = dict(distribution=indices.Distribution.gamma,
                  data_start_year=1987,
                  calibration_year_initial=1987,
                  calibration_year_final=2009,
                  periodicity=compute.Periodicity.monthly)


def check_file_(_dir, state, glob):
    fext = 'csv'
    fstr = '{}_{}'.format(glob, state)
    files = ['{}_{}_{}.{}'.format(fstr, y, m, fext) for y in range(1987, 2022) for m in range(1, 13)]
    files = [os.path.join(_dir, f) for f in files]
    return files


def concatenate_field_data(tfr_dir, metadata, out, subset='all', file_check=False, glob=None):
    for s in BASIN_STATES:
        if s != 'CA':
            continue
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df.sort_index()
        dt_range = ['{}-{}-01'.format(y, m) for y in range(1987, 2022) for m in range(1, 13)]
        data = np.zeros((df.shape[0], len(dt_range), len(COLS)))

        print('\n', s)

        if subset == 'usbr':
            df = df[df['usbrid'] > 0]
        elif subset == 'nonusbr':
            df = df[df['usbrid'] == 0]
        else:
            pass

        file_list = [os.path.join(tfr_dir, '{}_{}_{}_{}.csv'.format(glob, s, y, m))
                     for y in range(1987, 2022)
                     for m in range(1, 13)]

        if file_check:
            targets = check_file_(tfr_dir, s, glob=glob)
            missing = [os.path.basename(f) for f in targets if f not in file_list]
            print('missing {} files in {}'.format(len(missing), s))
            pprint([os.path.basename(f) for f in missing])
            open('missing.txt', 'a').write('\n'.join(missing) + '\n')
            continue

        for csv in file_list:
            splt = csv.split('.')[0].split('_')
            m, y = int(splt[-1]), int(splt[-2])
            dt_ind = dt_range.index('{}-{}-01'.format(y, m))
            c = pd.read_csv(csv)
            c.index = c['OPENET_ID']
            c = c.reindex(df.index)
            if not 3 < m < 11:
                c[['et', 'cc', 'eff_ppt', 'ietr']] = np.zeros((c.shape[0], 4))
            c = c[COLS]
            data[:, dt_ind, :] = c.values

        out_path = os.path.join(out, '{}.npy'.format(s))
        if subset in ['usbr', 'nonusbr']:
            out_path = os.path.join(out, '{}_{}.npy'.format(s, subset))
        data.tofile(out_path)
        out_js = out_path.replace('.npy', '_index.json')
        with open(out_js, 'w') as fp:
            json.dump({'index': list(df.index)}, fp, indent=4)
        print(out_path)


def map_indices(npy, out_js):
    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    ag_periods = list(range(1, 8))
    periods = list(product(met_periods, ag_periods))

    data = np.fromfile(npy, dtype=float)
    js = npy.replace('.npy', '.json')

    with open(js, 'r') as fp:
        index = json.load(fp)['index']

    data = data.reshape((len(index), -1, len(COLS)))
    cols_ = ['{}_{}'.format(m, a) for m, a in periods]
    df = pd.DataFrame(columns=cols_, index=index)

    for met_p, ag_p in periods:
        kc = data[:, :, COLS.index('et')] / data[:, :, COLS.index('ietr')]
        simi = np.apply_along_axis(lambda x: indices.spi(x, scale=ag_p, **IDX_KWARGS), arr=kc, axis=1)
        cwb = data[:, :, COLS.index('ppt')] - data[:, :, COLS.index('etr')]
        # uses locally modified climate_indices package that takes cwb = ppt - pet as input
        spei = np.apply_along_axis(lambda x: indices.spei(x, scale=met_p, **IDX_KWARGS), arr=cwb, axis=1)
        for i, ind in enumerate(index):
            _simi, _spei, rng = simi[i, :], spei[i, :], np.arange(len(spei))
            mask = ~np.isnan(np.array(_simi + _spei))
            _simi, _spei, rng = _simi[mask], _spei[mask], rng[mask]
            r = lr()
            pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    tfr_ = '/media/nvm/field_pts/csv'
    out_ = '/media/nvm/field_pts/field_pts_data'
    meta_ = '/media/nvm/field_pts/usbr_attr/'
    # concatenate_field_data(tfr_, meta_, out_, subset='all', glob='ietr_fields_13FEB2023', file_check=False)

    in_ = '/media/nvm/field_pts/field_pts_data/CA.npy'
    out_ = '/media/nvm/field_pts/indices/CA.json'
    map_indices(in_, out_)
# ========================= EOF ====================================================================
