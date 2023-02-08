import os

import pandas as pd
# import findspark
# findspark.init('/home/dgketchum/spark-3.3.1-bin-hadoop3')
# findspark.find()
# import pyspark.pandas as ps
import pandas_tfrecords as pdtfr

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']


# def build_frame(tfr_dir, metadata, out_csv):
#     df = pd.read_csv(metadata)
#     dt_range = pd.date_range('1987-01-01', '1989-12-31', freq='M')
#     midx = pd.MultiIndex.from_product([df.index, dt_range], names=['idx', 'dt'])
#     mdf = ps.DataFrame()
#     l = [os.path.join(tfr_dir, x) for x in os.listdir(tfr_dir)]
#     for tfr in l:
#         c = pdtfr.tfrecords_to_pandas(file_paths=tfr)


def map_indices(tfr_dir, metadata, out_csv):

    df = pd.read_csv(metadata, index_col='OPENET_ID')
    df = df[~df.index.duplicated()]

    dct = {k: {'STUSPS': v['STUSPS'], 'x': v['x'], 'y': v['y'],
               'name': v['name'], 'usbrid': v['usbrid']} for k, v in df.iterrows()}

    dt_range = ['{}-{}-01'.format(y, m) for y in range(1987, 1990) for m in range(1, 13)]
    dt_ind = pd.DatetimeIndex(dt_range)
    midx = pd.MultiIndex.from_product([df.index, dt_ind], names=['idx', 'dt'])
    mdf = pd.DataFrame(index=midx, columns=COLS)
    l = [os.path.join(tfr_dir, x) for x in os.listdir(tfr_dir) if 'CA' in x]
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
        c.columns = ['{}_{}_{}'.format(c, m, y) for c in c.columns]
        mdf.loc[(c.index, dt), c.columns] = c.values
    pass

if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    tfr_ = '/media/nvm/field_pts/tfr'
    out_ = '/media/nvm/field_pts/field_pts_data'
    meta_ = '/media/nvm/field_pts/points_metadata_CA.csv'
    # build_frame(tfr_, meta_, out_)
    map_indices(tfr_, meta_, out_)
# ========================= EOF ====================================================================
