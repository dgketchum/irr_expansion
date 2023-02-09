import os
import json

import numpy as np
import pandas as pd
import pandas_tfrecords as pdtfr
import geopandas as gpd

from call_ee import BASIN_STATES

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']


def concat_tfr(tfr_dir, metadata, out):

    for s in BASIN_STATES:
        if s in ['AZ', 'CA']:
            continue
        print('\n', s)
        mdata = os.path.join(metadata, 'points_metadata_{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df[~df.index.duplicated()]

        dt_range = ['{}-{}-01'.format(y, m) for y in range(1987, 1990) for m in range(1, 13)]
        dt_ind = pd.DatetimeIndex(dt_range)
        midx = pd.MultiIndex.from_product([df.index, dt_ind], names=['idx', 'dt'])
        mdf = pd.DataFrame(index=midx, columns=COLS)
        l = [os.path.join(tfr_dir, x) for x in os.listdir(tfr_dir) if s in x]
        l.reverse()
        for tfr in l:
            splt = tfr.split('.')[0].split('_')
            m, y = int(splt[-1]), int(splt[-2])
            print(y, m)
            dt = pd.to_datetime('{}-{}-01'.format(y, m))
            c = pdtfr.tfrecords_to_pandas(file_paths=tfr)
            c.index = c['OPENET_ID']
            c = c[~c.index.duplicated()]
            if not 3 < m < 11:
                c[['et', 'cc', 'eff_ppt', 'ietr']] = np.zeros((c.shape[0], 4))
            c = c[COLS]
            mdf.loc[(c.index, dt), c.columns] = c.values

        os.path.join(out, '{}.csv'.format(s))
        mdf.to_csv()


def map_indices(csv, out_csv):
    df = pd.read_csv(csv, index_col=['idx', 'dt'])
    pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    tfr_ = '/media/nvm/field_pts/tfr'
    out_ = '/media/nvm/field_pts/field_pts_data'
    meta_ = '/media/nvm/field_pts/metadata/'
    # concat_tfr(tfr_, meta_, out_)

    in_ = '/media/nvm/field_pts/field_pts_data/CA.csv'
    out_ = '/media/nvm/field_pts/indices/CA.csv'
    # map_indices(in_, out_)
# ========================= EOF ====================================================================
