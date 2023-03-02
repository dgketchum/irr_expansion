import json
import os
from collections import OrderedDict
from itertools import combinations
from pprint import pprint

import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import process

from gridded_data import BASIN_STATES
from utils.cdl import cdl_key, study_area_crops
from utils.placenames import state_name_abbreviation


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
            if e == 16:
                continue
            if c[1] >= 0:
                clime = 'Wet'
            elif 0 > c[1] > -1.3:
                clime = 'Normal'
            elif -2 < c[1] < -1.3:
                clime = 'Dry'
            else:
                clime = 'Driest'

            # trans = (int(v.T[e - 1, 3]), int(c[3]))
            trans = (int(c[3]), int(v.T[e + 1, 3]))
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


def map_nass_to_csl(values_dir, txt):
    cdl = study_area_crops()
    df = pd.read_table(txt, header=None)
    df.columns = ['index', 'file', 'desc']
    df.drop(columns=['index'], inplace=True)
    df['desc'] = df['desc'].apply(lambda x: x.split('Price per')[0])

    dct_choices = sorted([v[0] for k, v in cdl.items() if len(v[0]) > 3 and v[1] > 1000])
    inv_cdl = {v[0]: k for k, v in cdl.items()}
    files = ['cpvl_p10_t004.csv', 'cpvl_p08_t002.csv', 'cpvl_p13_t007.csv']

    for f in files:

        table = pd.read_csv(os.path.join(values_dir, 'CropValuSu-02-24-2017', f),
                            skiprows=6, encoding_errors='ignore')
        try:
            table.columns = ['4', 'h', 'name', 'unit', 'y_minus_3', 'y_minus_2', 'y_minus_1']
        except ValueError:
            table.columns = ['4', 'h', 'name', 'y_minus_3', 'y_minus_2', 'y_minus_1']

        table = table[['name', 'y_minus_3', 'y_minus_2', 'y_minus_1']]
        table.dropna(subset=['name'], inplace=True)
        table.dropna(subset=['y_minus_3', 'y_minus_2', 'y_minus_1'], how='all', inplace=True)

        for i, r in table.iterrows():
            approx = process.extractOne(r['name'], choices=dct_choices)
            if approx[1] > 85:
                print('match {} == {}'.format(r['name'], approx[0]))
                test_exists = process.extractOne(r['name'], choices=df['desc'])
                if test_exists[1] > 85:
                    print('already in database, skipping')
                else:
                    print('appending')
                    df = df.append({'file': f, 'desc': approx[0]}, ignore_index=True)
            else:
                print('nonmatch {} == {}'.format(r['name'], approx[0]))

    df['cdl_name'] = df['desc'].apply(lambda x: process.extractOne(x, choices=dct_choices)[0])
    df['cdl_code'] = df['desc'].apply(lambda x: inv_cdl[process.extractOne(x, choices=dct_choices)[0]])
    df.to_csv('values_.csv')
    pass


def get_price_timeseries(dir_, mapping, out_csv):
    st_abv = state_name_abbreviation()
    full_states = [st_abv[k] for k in BASIN_STATES]
    cdl = study_area_crops()
    m = pd.read_csv(mapping)
    m.index = m['cdl_code']
    m.drop(columns=['Unnamed: 0', 'cdl_code'], inplace=True)
    files = m.T.to_dict()
    dct = {}
    for code, crop in cdl.items():
        if code not in files.keys():
            continue
        for year in range(2009, 2022):
            d = [x for x in os.listdir(dir_) if x.endswith(str(year))][0]
            d = os.path.join(dir_, d)
            t = [x for x in os.listdir(d) if files[code]['file'].split('_')[-1].replace('t', '') in x][0]
            f = os.path.join(d, t)
            c = pd.read_csv(f, skiprows=4, encoding_errors='ignore', header=None)
            c['state'] = c[2]
            c['yr_col'] = c[5]
            if 'AZ' in c['state'].values:
                c = c.loc[c['state'].apply(lambda x: True if x in BASIN_STATES else False)]
            else:
                c['state'] = [x.strip() if str(x).strip() in full_states else 'None' for x in list(c['state'])]
                c = c.loc[c['state'].apply(lambda x: True if x in full_states else False)]
            c = c[['state', 'yr_col']]
            print(list(c.columns))


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
    # cdl_accuracy(cdl_csv)

    price_data = '/media/research/IrrigationGIS/expansion/tables/crop_value/nass_data'
    price_ts = '/media/research/IrrigationGIS/expansion/tables/crop_value/time_series.csv'
    map_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/values.csv'
    get_price_timeseries(price_data, map_, price_ts)
# ========================= EOF ====================================================================
