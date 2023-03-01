import os
import json
from pprint import pprint
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from lxml import etree
from io import StringIO

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


def transition_probability(cdl_npy, out_matrix):
    cdl = cdl_key()
    cdl = {k: v[1] for k, v in cdl.items()}
    time_series_length = 17
    classes = ['Grain', 'Vegetable', 'Forage', 'Orchard', 'Uncultivated', 'Fallow']

    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    rec = rec[:, np.random.randint(0, rec.shape[1], int(1e6)), :]
    codes = np.where(rec[3] < 0, np.zeros_like(rec[3]), rec[3])
    codes = np.vectorize(cdl.get)(codes)
    rec[3] = codes

    set_ = list(np.unique(rec[3]))
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = [(int(a), int(b)) for a, b in keys]
    dct = {clime: {k: 0 for k in keys} for clime in ['Wet', 'Normal', 'Dry', 'Driest']}
    rec = np.moveaxis(rec, 0, 1)

    for v in rec:
        for e, c in enumerate(v.T):
            if e == 0:
                continue
            if c[1] >= 0:
                clime = 'Wet'
            elif 0 > c[1] > -1.3:
                clime = 'Normal'
            elif -2 < c[1] < -1.3:
                clime = 'Dry'
            else:
                clime = 'Driest'

            trans = (int(v.T[e - 1, 3]), int(c[3]))
            dct[clime][trans] += 1

    for k, d in dct.items():
        map = np.zeros((len(classes), len(classes)))
        for r, c in d.keys():
            map[r - 1, c - 1] = d[r, c]
        prob = np.divide(map, np.sum(map, axis=0))
        tdf = pd.DataFrame(columns=classes, index=classes, data=prob)
        print(k)
        print(tdf, '\n')
        out_file = os.path.join(out_matrix, '{}.csv'.format(k.lower()))
        tdf.to_csv(out_file, float_format='%.3f')

    pass


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


def cdl_accuracy(out_js):
    dct = {s: [] for s in BASIN_STATES}
    for y in range(2008, 2022):
        for s in BASIN_STATES:
            url = 'https://www.nass.usda.gov/Research_and_Science/' \
                  'Cropland/metadata/metadata_{}{}.htm'.format(s.lower(), str(y)[-2:])
            resp = requests.get(url).content.decode('utf-8')
            for i in resp.split('\n'):
                txt = i.strip('\r')
                if txt.startswith('OVERALL'):
                    l = txt.split(' ')
                    k = float(l[-1])
            dct[s].append(k)
            print(s, y, '{:.3f}'.format(k))

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    fields_openet = '/media/research/IrrigationGIS/openET/OpenET_GeoDatabase_cdl'
    extract = '/media/nvm/field_pts/csv/cdl'
    cdl_csv = '/media/research/IrrigationGIS/expansion/tables/cdl/crops'
    # get_openet_cdl(fields_openet, extract, cdl_csv)

    key_ = '/home/dgketchum/quickstats_token.json'
    # get_crop_values(key_)

    met_cdl = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met4_ag3_fr8.npy'
    transistion = '/media/research/IrrigationGIS/expansion/analysis/transition'
    # transition_probability(met_cdl, transistion)

    cdl_csv = '/media/research/IrrigationGIS/expansion/tables/cdl/accuracy/acc.json'
    cdl_accuracy(cdl_csv)
# ========================= EOF ====================================================================
