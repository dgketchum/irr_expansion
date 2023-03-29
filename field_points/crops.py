import json
import os
from collections import OrderedDict
from pprint import pprint

import numpy as np
import pandas as pd

from gridded_data import BASIN_STATES
from field_points.crop_codes import cdl_key


def get_openet_cdl(in_dir, join_csv, out_dir):
    with open('tiles.json', 'r') as f_obj:
        tiles_dct = json.load(f_obj)

    crop_counts = {}
    cdl = cdl_key()

    for state in BASIN_STATES:

        tiles = tiles_dct[state]
        l = [os.path.join(join_csv, 'openet_cdl_{}_{}_2021.csv'.format(state, tile)) for tile in tiles]
        first = True
        for f_ in l:
            c = pd.read_csv(f_, index_col='OPENET_ID')
            if first:
                adf = c.copy()
                first = False
            else:
                adf = pd.concat([adf, c])

        f = os.path.join(in_dir, '{}.csv'.format(state))
        df = pd.read_csv(f, index_col='OPENET_ID')
        match = [i for i in df.index if i in adf.index]
        df = df.loc[match]
        df['CROP_2021'] = [0 for _ in range(df.shape[0])]
        df.loc[match, 'CROP_2021'] = adf.loc[match, 'mode'].values.astype(int)
        outf = os.path.join(out_dir, '{}.csv'.format(state))
        counts = np.unique(df.values, return_counts=True)
        for code, ct in zip(counts[0], counts[1]):
            if code not in crop_counts.keys():
                crop_counts[code] = ct
            else:
                crop_counts[code] += ct
        df.to_csv(outf)

    crop_counts = {k: v for k, v in crop_counts.items() if k > 0}
    codes = list(crop_counts.keys())
    l = sorted([(c, (cdl[c][0], crop_counts[c])) for c in codes if len(cdl[c][0]) > 3],
               key=lambda x: x[1][1], reverse=True)
    dct = OrderedDict(l)
    pprint(dct)


def cdl_area_timeseries(cdl_dir, area_json, out_json):
    with open(area_json, 'r') as _file:
        areas = json.load(_file)

    area = pd.DataFrame(index=[k for k, v in areas.items()], data=[v for k, v in areas.items()], columns=['area'])

    l = [os.path.join(cdl_dir, x) for x in os.listdir(cdl_dir)]
    first = True
    for csv in l:
        if first:
            df = pd.read_csv(csv, index_col='OPENET_ID')
            first = False
        else:
            c = pd.read_csv(csv, index_col='OPENET_ID')
            df = pd.concat([df, c])

    match = [i for i in df.index if i in area.index]
    df = df.loc[match]
    df[df.values < 0] = np.nan
    for c in df.columns:
        area[c] = area['area']
    area.drop(columns=['area'], inplace=True)

    dct = {}
    l = [int(c) for c in list(np.unique(df.values, return_counts=True)[0]) if np.isfinite(c) and c > 0]
    for c in l:
        a_vals, c_vals = area.values.copy(), df.values.copy()
        a_vals[c_vals != c] = np.nan
        dct[c] = list(np.nansum(a_vals, axis=0))
        print(c, cdl_key()[c], '{:.3f}'.format(np.array(dct[c]).mean()))

    with open(out_json, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    crops = '/media/research/IrrigationGIS/expansion/tables/cdl/crops'
    areas_ = '/media/research/IrrigationGIS/expansion/tables/cdl/fields_area.json'
    cdl_area_ = '/media/research/IrrigationGIS/expansion/tables/cdl/cdl_area_timesereies.json'
    cdl_area_timeseries(crops, areas_, cdl_area_)

# ========================= EOF ====================================================================
