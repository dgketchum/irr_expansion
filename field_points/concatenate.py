import os
import json

import numpy as np
import pandas as pd

from call_ee import BASIN_STATES

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']


def concatenate_field_data(csv_dir, metadata, out, file_check=False, glob=None):

    with open('tiles.json', 'r') as _file:
        tile_dct = json.load(_file)

    for s in BASIN_STATES[1:]:
        tiles = tile_dct[s]
        mdata = os.path.join(metadata, '{}.csv'.format(s))
        df = pd.read_csv(mdata, index_col='OPENET_ID')
        df = df.sort_index()
        dt_range = ['{}-{}-01'.format(y, m) for y in range(1984, 2022) for m in range(1, 13)]
        data = np.zeros((df.shape[0], len(dt_range), len(COLS)))

        print('\n', s)

        file_list = [os.path.join(csv_dir, '{}_{}_{}_{}_{}.csv'.format(glob, s, t, y, m))
                     for y in range(1984, 2022)
                     for m in range(1, 13)
                     for t in tiles]

        if file_check:
            dir_list = os.listdir(os.path.join(csv_dir))
            bnames = [os.path.basename(f) for f in file_list]
            missing = [f.strip('.csv') for f in bnames if f not in dir_list]
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


if __name__ == '__main__':
    root = '/media/nvm'
    if not os.path.exists(root):
        root = '/home/dgketchum/data'

    csv_ = os.path.join(root, 'field_pts/csv/fields')
    fpd = os.path.join(root, 'field_pts/field_pts_data')
    meta_ = os.path.join(root, 'field_pts/usbr_attr/')
    concatenate_field_data(csv_, meta_, fpd, glob='ietr_fields_16FEB2023', file_check=True)
# ========================= EOF ====================================================================
