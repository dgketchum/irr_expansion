import os
import json
from pprint import pprint

import pandas as pd

from call_ee import BASIN_STATES


def get_openet_cdl(in_dir, join_csv, out_dir):
    with open('field_points/tiles.json', 'r') as f_obj:
        tiles_dct = json.load(f_obj)

    all_crop_codes = []

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
        [all_crop_codes.append(crop) for crop in list(set(df.values.flatten())) if crop not in all_crop_codes]
        df.to_csv(outf)
        print(outf)

    all_crop_codes.sort()
    pprint(all_crop_codes)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    fields_openet = '/media/research/IrrigationGIS/openET/OpenET_GeoDatabase_cdl'
    cdl_csv = '/media/nvm/field_pts/fields_data/fields_cdl'
    extract = '/media/nvm/field_pts/csv/cdl'
    get_openet_cdl(fields_openet, extract, cdl_csv)
# ========================= EOF ====================================================================
