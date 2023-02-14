import os
import json
from pprint import pprint

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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


def concat_tfr(tfr_dir, metadata, out, subset='all', file_check=False, glob=None):
    import pandas_tfrecords as pdtfr
    m, y = None, None

    for s in BASIN_STATES:
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df[~df.index.duplicated()]
        df = df.sort_index()
        dt_range = ['{}-{}-01'.format(y, m) for y in range(1987, 2022) for m in range(1, 13)]
        dt_ind = pd.DatetimeIndex(dt_range)
        midx = pd.MultiIndex.from_product([df.index, dt_ind], names=['idx', 'dt'])
        mdf = pd.DataFrame(index=midx, columns=COLS)

        print('\n', s)

        if subset == 'usbr':
            df = df[df['usbrid'] > 0]
        elif subset == 'nonusbr':
            df = df[df['usbrid'] == 0]
        else:
            pass

        l = [os.path.join(tfr_dir, x) for x in os.listdir(tfr_dir) if s in x]
        l.sort()

        if file_check:
            targets = check_file_(tfr_dir, s, glob=glob)
            missing = [os.path.basename(f) for f in targets if f not in l]
            print('missing {} files in {}'.format(len(missing), s))
            pprint([os.path.basename(f) for f in missing])
            open('missing.txt', 'a').write('\n'.join(missing) + '\n')
            continue

        for csv in l:
            try:
                splt = csv.split('.')[0].split('_')
                m, y = int(splt[-1]), int(splt[-2])
                dt = pd.to_datetime('{}-{}-01'.format(y, m))
                c = pdtfr.tfrecords_to_pandas(file_paths=csv)
                c.index = c['OPENET_ID']
                print(y, m, len(c.index))
                c = c[['rand']]
                match = [i for i in c.index if i in df.index]
                df.loc[match, 'rand'] = c.loc[match, 'rand']
                c = c[~c.index.duplicated()]
                if not 3 < m < 11:
                    c[['et', 'cc', 'eff_ppt', 'ietr']] = np.zeros((c.shape[0], 4))
                c = c[COLS]
                match_idx = [i for i in c.index if i in mdf.index]
                mdf.loc[(match_idx, dt), c.columns] = c.loc[match_idx].values

            except Exception as e:
                print(e, csv, m, y)
                continue

        out_path = os.path.join(out, '{}.csv'.format(s))
        if subset in ['usbr', 'nonusbr']:
            out_path = os.path.join(out, '{}_{}.csv'.format(s, subset))
        mdf.to_csv(out_path)


def map_indices(csv, out_js):
    df = pd.read_csv(csv, index_col=['idx', 'dt'])
    df['kc'] = df['et'] / df['ietr']
    simi = df.groupby(level=0).apply(lambda x: indices.spi(x.kc.values, scale=1, **IDX_KWARGS))
    simi = np.stack(simi.values).flatten()
    spei = df.groupby(level=0).apply(lambda x: indices.spei(x.ppt.values, x.etr.values, scale=11, **IDX_KWARGS))
    spei = np.stack(spei.values).flatten()
    resp_data = np.array([simi, spei]).T
    df = pd.DataFrame(data=resp_data, columns=['simi', 'spei'], index=df.index)
    dct = {level: df.xs(level).to_dict('index') for level in df.index.levels[0]}
    with open(out_js, 'w') as f:
        json.dump(dct, f, indent=4)
        print(out_js)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    tfr_ = '/media/nvm/field_pts/csv'
    out_ = '/media/nvm/field_pts/field_pts_data'
    meta_ = '/media/nvm/field_pts/metadata/'
    concat_tfr(tfr_, meta_, out_, subset='all', glob='ietr_fields_13FEB2023', file_check=False)

    in_ = '/media/nvm/field_pts/field_pts_data/CO.csv'
    out_ = '/media/nvm/field_pts/indices/CO.json'
    # map_indices(in_, out_)
# ========================= EOF ====================================================================
