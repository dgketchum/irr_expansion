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


def concat_tfr(tfr_dir, metadata, out, subset='all'):
    import pandas_tfrecords as pdtfr

    for s in BASIN_STATES:
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df[~df.index.duplicated()]
        df = df.sort_index()
        dt_range = ['{}-{}-01'.format(y, m) for y in range(1987, 1988) for m in range(1, 2)]
        dt_ind = pd.DatetimeIndex(dt_range)
        midx = pd.MultiIndex.from_product([df.index, dt_ind], names=['idx', 'dt'])
        mdf = pd.DataFrame(index=midx, columns=COLS)

        for r in range(10):
            print('\n', s, r)

            if subset == 'usbr':
                df = df[df['usbrid'] > 0]
            elif subset == 'nonusbr':
                df = df[df['usbrid'] == 0]
            else:
                pass

            glb = '{}_{}'.format(s, r)
            l = [os.path.join(tfr_dir, x) for x in os.listdir(tfr_dir) if glb in x]
            l.sort()

            for tfr in l[:1]:
                try:
                    splt = tfr.split('.')[0].split('_')
                    m, y = int(splt[-1]), int(splt[-2])
                    dt = pd.to_datetime('{}-{}-01'.format(y, m))
                    c = pdtfr.tfrecords_to_pandas(file_paths=tfr)
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
                    print(e, tfr, m, y)
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

    tfr_ = '/media/nvm/field_pts/tfr'
    out_ = '/media/nvm/field_pts/field_pts_data'
    meta_ = '/media/nvm/field_pts/metadata/'
    concat_tfr(tfr_, meta_, out_, subset='all')

    in_ = '/media/nvm/field_pts/field_pts_data/CO.csv'
    out_ = '/media/nvm/field_pts/indices/CO.json'
    # map_indices(in_, out_)
# ========================= EOF ====================================================================
