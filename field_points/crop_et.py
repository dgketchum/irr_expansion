import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from field_points.crop_codes import BASIN_STATES
from transition_modeling.transition_data import KEYS

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']


def split(a, n):
    k, m = divmod(len(a), n)
    inds = [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]
    return inds


def cdl_crop_et(npy, shp_dir, out_js):
    COLS.append('cdl')
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(2005, 2022) for m in range(1, 13)]
    vols = {k: 0 for k in KEYS}

    for state in BASIN_STATES[:1]:

        npy_file = os.path.join(npy, '{}.npy'.format(state))

        shp_file = os.path.join(shp_dir, '{}.shp'.format(state))
        shp = gpd.read_file(shp_file)
        shp['area'] = shp['geometry'].apply(lambda x: x.area)
        shp = {r['OPENET_ID']: r['area'] for i, r in shp.iterrows()}

        js = npy_file.replace('.npy', '_index.json')
        with open(js, 'r') as fp:
            d = json.load(fp)
            index = d['index']
            areas = [shp[i] for i in index]

        n = int(np.ceil(len(index) / 5e3))
        indx = split(index, n)

        data_mem = np.fromfile(npy_file)
        data_mem = data_mem.reshape((len(index), -1, len(COLS)))

        for i, (s_ind, e_ind) in enumerate(indx):
            data = data_mem[s_ind:e_ind, :, :]

            months = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                 np.array([dt.month in list(range(4, 11)) for dt in dt_range]))
            classific = data[:, -len(dt_range):, -1]
            area = np.array(areas[s_ind: e_ind]).repeat(months.shape[1]).reshape(months.shape)
            cc = data[:, -len(dt_range):, 1] * area
            stack = np.stack([cc, months, classific])
            unq = np.unique(classific)
            for code in unq:
                if code not in KEYS:
                    continue
                print(state, code, i)
                d = stack[:, stack[1] == 1.0].reshape((3, len(index[s_ind:e_ind]), -1))
                s = d[:, d[2] == code][0, :]
                s = np.nansum(s)
                vols[code] += s

    with open(out_js, 'w') as fp:
        json.dump(vols, fp, indent=4)

    print(out_js)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    indir = os.path.join(root, 'field_pts/fields_cdl_npy')
    part_ = os.path.join(root, 'field_pts/crop_et/et_cdl.json')
    shp_d = os.path.join(root, 'field_pts/fields_shp')
    cdl_crop_et(indir, shp_d, part_)

# ========================= EOF ====================================================================
