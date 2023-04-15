import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from field_points.crop_codes import BASIN_STATES
from transition_modeling.transition_data import KEYS, OLD_KEYS
from field_points.crop_codes import cdl_key

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

    for state in ['MT']:
        print(state)
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
                if code != 36:
                    continue
                print(state, code, i)
                d = stack[:, stack[2] == code].reshape((3, len(index[s_ind:e_ind]), -1))
                s = d[:, d[2] == code][0, :]
                s = np.nansum(s)
                vols[code] += s

    with open(out_js, 'w') as fp:
        json.dump(vols, fp, indent=4)

    print(out_js)


def cdl_mean_cc(areas, ccons, response, out_):
    with open(areas, 'r') as fp:
        areas = json.load(fp)

    resp = pd.read_csv(response, index_col='crop_code')
    resp.index = [str(int(i)) for i in resp.index]

    with open(ccons, 'r') as fp:
        et = json.load(fp)
        et = pd.Series(et)
        et = pd.DataFrame(et)
        et = et.rename(columns={0: 'Irrigation Water Use [m]'})

        match = [i for i in resp.index if i in et.index]
        et.loc[match, 'Response'] = resp.loc[match, 'mean_resp']

        areas = {k: np.array(v).mean() for k, v in areas.items() if k in et.index}
        et['Area [sq km]'] = pd.Series(areas)
        et['Crop Code CDL'] = et.index
        et = et.loc[[i for i in et.index if int(i) in OLD_KEYS]]
        et = et.reindex([str(k) for k in OLD_KEYS])
        et.index = [i for i in range(len(et))]

    cdl = cdl_key()
    set_, classes = OLD_KEYS, [cdl[c][0] for c in OLD_KEYS]
    et['Crop'] = classes
    et['IWU Error [m]'] = et['Irrigation Water Use [m]'] * 0.33
    et['IWU Rank'] = et['Irrigation Water Use [m]'].rank(axis=0, ascending=False)
    et['Area Rank'] = et['Area [sq km]'].rank(axis=0, ascending=False)
    et = et[['Crop Code CDL', 'Crop', 'Area [sq km]', 'Response', 'IWU Rank',
             'Area Rank', 'IWU Rank']]
    et.to_csv(out_)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    indir = os.path.join(root, 'field_pts/fields_cdl_npy')
    part_ = os.path.join(root, 'field_pts/crop_et/et_cdl.json')
    shp_d = os.path.join(root, 'field_pts/fields_shp')
    # cdl_crop_et(indir, shp_d, part_)

    areas_cdl = os.path.join(root, 'tables/cdl/cdl_area_timeseries.json')
    response_cdl = os.path.join(root, 'analysis/cdl_response.csv')
    ccons = '/media/nvm/field_pts/fields_data/cdl_cc.json'
    out_ = os.path.join(root, 'analysis/cdl_summary.csv')
    cdl_mean_cc(areas_cdl, ccons, response_cdl, out_)
# ========================= EOF ====================================================================
