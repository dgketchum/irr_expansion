import os
import json
from pprint import pprint
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd

from nasspython.nass_api import nass_data, nass_count, nass_param
from call_ee import BASIN_STATES
from utils.cdl import study_area_crops, cdl_key
from utils.placenames import state_name_abbreviation

kwargs = dict(source_desc='CENSUS', sector_desc='CROPS', group_desc=None, commodity_desc=None, short_desc=None,
              domain_desc=None, agg_level_desc='STATE', domaincat_desc=None, statisticcat_desc=None,
              state_name=None, asd_desc=None, county_name=None, region_desc=None, zip_5=None,
              watershed_desc=None, year=None, freq_desc='ANNUAL', reference_period_desc=None)


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

    crop_counts = {k: v for k, v in crop_counts.items() if v > 10000}
    codes = list(crop_counts.keys())
    l = sorted([(c, (cdl[c][0], crop_counts[c])) for c in codes], key=lambda x: x[1][1], reverse=True)
    dct = OrderedDict(l)
    pprint(dct)


def transition_probability(csv_dir, out_matrix):

    cdl = cdl_key()
    classes = ['Grain', 'Vegetable', 'Forage', 'Orchard', 'Uncultivated']

    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    first = True
    for f_ in l:
        c = pd.read_csv(f_, index_col='OPENET_ID')
        c[c.values < 0] = 0
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    l = list(df.values.flatten())
    l = [cdl[v][1] for v in l]

    set_ = list(set(l))
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    dct = {k: 0 for k in keys}
    for i, v in enumerate(l):
        if i % df.shape[1] == 0:
            continue
        try:
            dct[(l[i], l[i + 1])] += 1
        except IndexError:
            break
    map = np.zeros((5, 5))
    for r, c in dct.keys():
        map[r - 1, c - 1] = dct[r, c]
    prob = np.divide(map, np.sum(map, axis=0))
    tdf = pd.DataFrame(columns=classes, index=classes, data=prob)


def get_crop_values(key):
    state_abv = state_name_abbreviation()

    with open(key, 'r') as fp:
        key = json.load(fp)['auth']

    for state in BASIN_STATES:
        kwargs['state_name'] = state_abv[state].upper()
        kwargs['commodity_desc'] = 'HAY'
        kwargs['year'] = 2017
        kwargs['domain_desc'] = 'AREA HARVESTED'
        kwargs['short_desc'] = 'HAY - ACRES HARVESTED'
        resp = nass_data(key, **kwargs)
        pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    fields_openet = '/media/research/IrrigationGIS/openET/OpenET_GeoDatabase_cdl'
    extract = '/media/nvm/field_pts/csv/cdl'
    cdl_csv = '/media/nvm/field_pts/fields_data/fields_cdl'
    # get_openet_cdl(fields_openet, extract, cdl_csv)

    key_ = '/home/dgketchum/quickstats_token.json'
    # get_crop_values(key_)

    transistion = '/media/research/IrrigationGIS/expansion/analysis/transition'
    transition_probability(cdl_csv, transistion)
# ========================= EOF ====================================================================
