import os
import json

import numpy as np
import pandas as pd

from itype_mapping import itype_integer_mapping
from call_ee import BASIN_STATES

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']


def concatenate_field_data(csv_dir, metadata, out, file_check=False, glob=None, itype_dir=None):
    with open('field_points/tiles.json', 'r') as _file:
        tile_dct = json.load(_file)

    itype_map = itype_integer_mapping()
    for s in BASIN_STATES[7:]:

        tiles = tile_dct[s]
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df.sort_index()

        if s in ['CO', 'MT', 'UT', 'WY'] and itype_dir:
            itype_f = os.path.join(itype_dir, '{}_openet_itype.csv'.format(s))
            idf = pd.read_csv(itype_f, index_col='OPENET_ID')
            idf = idf[~idf.index.duplicated()]
            idf['remap'] = idf['itype'].apply(lambda x: itype_map[x])
            match = [i for i in df.index if i in idf.index]
            idf = idf.loc[match]
            df['itype'] = [0 for _ in range(df.shape[0])]
            df.loc[match, 'itype'] = idf.loc[match, 'remap']
            df['itype'] = df['itype'].values.astype(int)
        else:
            df['itype'] = [0 for _ in range(df.shape[0])]

        dt_range = ['{}-{}-01'.format(y, m) for y in range(1984, 2022) for m in range(1, 13)]
        data = np.zeros((df.shape[0], len(dt_range), len(COLS)))

        print('\n', s)

        file_list = [os.path.join(csv_dir, '{}_{}_{}_{}_{}.csv'.format(glob, s, t, y, m))
                     for y in range(1984, 2022)
                     for m in range(1, 13)
                     for t in tiles]

        file_dct = {'{}_{}'.format(y, m): [f for f in file_list if f.endswith('{}_{}.csv'.format(y, m))]
                    for y in range(1984, 2022)
                    for m in range(1, 13)}

        if file_check:
            dir_list = os.listdir(os.path.join(csv_dir))
            bnames = [os.path.basename(f) for f in file_list]
            missing = [f.strip('.csv') for f in bnames if f not in dir_list]
            open('missing.txt', 'a').write('\n'.join(missing))
            continue

        for dtstr, csv_list in file_dct.items():
            tdf = None
            y, m = int(dtstr[:4]), int(dtstr[5:])
            dt_ind = dt_range.index('{}-{}-01'.format(y, m))
            first = True
            for csv in csv_list:
                c = pd.read_csv(csv, index_col='OPENET_ID')
                match = [i for i in c.index if i in df.index]
                c = c.loc[match]
                print(c.shape, csv)

                if m not in range(4, 11) or y < 1987:
                    c[['et', 'cc', 'eff_ppt', 'ietr']] = np.zeros((c.shape[0], 4))

                if first:
                    tdf = c.copy()
                    first = False
                else:
                    tdf = pd.concat([tdf, c])

            tdf = tdf[COLS]
            data[:, dt_ind, :] = tdf.values

        out_path = os.path.join(out, '{}.npy'.format(s))
        data.tofile(out_path)
        out_js = out_path.replace('.npy', '_index.json')
        dct = {'index': list(df.index), 'usbrid': list(df['usbrid']), 'itype': list(df['itype'])}
        with open(out_js, 'w') as fp:
            json.dump(dct, fp, indent=4)
        print(out_path)


if __name__ == '__main__':
    root = '/media/nvm'
    if not os.path.exists(root):
        root = '/home/dgketchum/data'

    csv_ = os.path.join(root, 'field_pts/csv/fields')
    fpd = os.path.join(root, 'field_pts/fields_data/fields_npy')
    meta_ = os.path.join(root, 'field_pts/fields_data/fields_shp')
    itype_data = '/media/research/IrrigationGIS/expansion/tables/itype'
    concatenate_field_data(csv_, meta_, fpd, glob='ietr_fields_16FEB2023',
                           file_check=False, itype_dir=itype_data)
# ========================= EOF ====================================================================
