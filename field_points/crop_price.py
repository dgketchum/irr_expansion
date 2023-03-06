import json
import os

import numpy as np
import pandas as pd
from fuzzywuzzy import process
from nasspython.nass_api import nass_data

from gridded_data import BASIN_STATES
from utils.cdl import study_area_crops, nass_price_queries
from utils.placenames import state_name_abbreviation


def get_annual_price_timeseries(dir_, mapping, out_js):
    st_abv = state_name_abbreviation()
    inv_st_abv = {v: k for k, v in st_abv.items()}
    cdl = study_area_crops()
    m = pd.read_csv(mapping)
    m['table'] = m['file'].apply(lambda x: x.split('.')[0][-3:])
    m.index = m['cdl_code']
    m.drop(columns=['Unnamed: 0', 'cdl_code', 'file', 'desc'], inplace=True)
    tables = m.T.to_dict()
    dct, target_year = {}, {}
    for i, (code, (crop, count)) in enumerate(cdl.items()):
        if code not in tables.keys():
            continue
        table = tables[code]['table']
        dct[code] = {s: (crop, []) for s in BASIN_STATES}
        years = list(range(2003, 2023))
        for year in years:
            if year > 2019:
                target_col = 5
            elif table == '004':
                target_col = 4
            else:
                target_col = 3
            d = [x for x in os.listdir(dir_) if x.endswith(str(year))][0]
            d = os.path.join(dir_, d)

            try:
                t = [x for x in os.listdir(d) if table in x][0]
            except IndexError:
                print(i, year, crop, count, 'no table')
                continue

            f = os.path.join(d, t)

            try:

                if table in ['004', '007', '002']:
                    c = pd.read_csv(f, skiprows=9, encoding_errors='ignore', header=None)
                    ind = list(c[2]).index(process.extractOne(crop, choices=c[2])[0])
                    p = float(c.loc[ind, 4])
                    for s in BASIN_STATES:
                        dct[code][s][1].append((year, float(p)))
                        dct[code]['national'] = True
                else:
                    c = pd.read_csv(f, skiprows=7, encoding_errors='ignore', header=None)
                    c['state'] = c[2]
                    c['yr_col'] = c[target_col]
                    if 'AZ' in c['state'].values:
                        c = c.loc[c['state'].apply(lambda x: True if x in BASIN_STATES else False)]
                    else:
                        c['state'] = [inv_st_abv[x.strip()] if str(x).strip() in inv_st_abv.keys()
                                      else 'None' for x in list(c['state'])]
                        c = c.loc[c['state'].apply(lambda x: True if x in BASIN_STATES else False)]
                    c = c[['state', 'yr_col']]

                    for s, p in zip(c['state'], c['yr_col']):
                        try:
                            dct[code][s][1].append((year, float(p)))
                        except ValueError:
                            dct[code][s][1].append((year, np.nan))
                        dct[code]['national'] = False

            except pd.errors.ParserError as e:
                print(i, year, crop)
                pass

        print(code, crop, 'success')

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


def get_monthly_price_timeseries(key, out_js):
    state_abv = state_name_abbreviation()
    queries = nass_price_queries()

    with open(key, 'r') as fp:
        key = json.load(fp)['auth']

    states = [s.upper() for abv, s in state_abv.items() if abv in BASIN_STATES]
    dct = {c: {s: [] for s in states} for c in queries.keys()}

    for code, query in queries.items():
        for year in range(2008, 2022):
            query['year'] = year
            resp = nass_data(key, **query)
            if isinstance(resp, str):
                break
            for i in resp['data']:
                try:
                    dct[code][i['location_desc']].append(('{}-{}-01'.format(year, i['begin_code']), i['Value']))
                except KeyError:
                    continue

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    key_ = '/home/dgketchum/quickstats_token.json'
    nass_price_monthly = '/media/research/IrrigationGIS/expansion/analysis/nass_price_monthly.json'
    # get_monthly_price_timeseries(key_, nass_price_monthly)

    price_data = '/media/research/IrrigationGIS/expansion/tables/crop_value/nass_data'
    map_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/values.csv'
    nass_price_annual = '/media/research/IrrigationGIS/expansion/analysis/nass_price_annual.json'
    get_annual_price_timeseries(price_data, map_, nass_price_annual)

# ========================= EOF ====================================================================
