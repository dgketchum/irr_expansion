import json
import os

import numpy as np
import pandas as pd
from fuzzywuzzy import process
from nasspython.nass_api import nass_data
from climate_indices.indices import spi, Distribution, compute

from gridded_data import BASIN_STATES
from field_points.crop_codes import nass_monthly_price_queries, nass_annual_price_queries
from field_points.crop_codes import study_area_crops, ppi_to_cdl_crop
from utils.placenames import state_name_abbreviation

KWARGS = dict(scale=9,
              distribution=Distribution.gamma,
              data_start_year=2006,
              calibration_year_initial=2006,
              calibration_year_final=2021,
              periodicity=compute.Periodicity.monthly)


def get_annual_price_timeseries(dir_, mapping, out_js, key):
    st_abv, nass_queries = state_name_abbreviation(), nass_annual_price_queries()
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
        dct[code] = {s: [] for s in BASIN_STATES}
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
                        dct[code][s].append(('{}-07-01'.format(year), float(p)))
                        dct[code]['national'] = True

                # elif crop in ['Rye', 'Mint']:
                #     # TODO: re-run when re-authorized
                #     query = nass_queries[crop]
                #     query['year'] = year
                #     resp = nass_data(key, **query)

                else:
                    c = pd.read_csv(f, skiprows=7, encoding_errors='ignore', header=None)
                    c['state'] = c[2]
                    c['yr_col'] = c[target_col]
                    if 'CA' in c['state'].values:
                        c = c.loc[c['state'].apply(lambda x: True if x in BASIN_STATES else False)]
                    else:
                        c['state'] = [inv_st_abv[x.strip()] if str(x).strip() in inv_st_abv.keys()
                                      else 'None' for x in list(c['state'])]
                        c = c.loc[c['state'].apply(lambda x: True if x in BASIN_STATES else False)]
                    c = c[['state', 'yr_col']]
                    for s, p in zip(c['state'], c['yr_col']):
                        try:
                            dct[code][s].append(('{}-07-01'.format(year), float(p)))
                        except ValueError:
                            dct[code][s].append(('{}-07-01'.format(year), np.nan))
                        dct[code]['national'] = False

            except pd.errors.ParserError as e:
                print(i, year, crop)
                pass

        print(code, crop, 'success')

    dct = {cdl[k][0]: v for k, v in dct.items()}
    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


def get_monthly_price_timeseries(key, out_js):
    state_abv = state_name_abbreviation()
    queries = nass_monthly_price_queries()

    with open(key, 'r') as fp:
        key = json.load(fp)['auth']

    states = [s.upper() for abv, s in state_abv.items() if abv in BASIN_STATES]
    dct = {c: {s: [] for s in states} for c in queries.keys()}

    for code, query in queries.items():
        for year in range(2006, 2022):
            print(code, year)
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


def _from_series(tseries, state):
    try:
        yrs = [int(x[0][:4]) for x in tseries[state]]
        data = [float(x[1]) for x in tseries[state]]
        ind = [pd.to_datetime('{}-07-01'.format(y)) for y in yrs]
    except ValueError:
        ind = [pd.to_datetime('{}-07-01'.format(y)) for y in range(2008, 2022)]
        data = [np.nan for x in ind]
    series = pd.Series(index=ind, data=data)
    return series


def nominal_price_data(monthly_price, annual_price, nominal):
    st_names = state_name_abbreviation()
    st_abv = {v.upper(): k for k, v in st_names.items() if k in BASIN_STATES}

    with open(monthly_price, 'r') as fp:
        month_price = json.load(fp)
    month_price = {k: {st_abv[kk]: vv for kk, vv in month_price[k].items()} for k, v in month_price.items()}

    with open(annual_price, 'r') as fp:
        annual_price = json.load(fp)

    counts = study_area_crops()
    counts = {k: v for k, v in counts.items() if v[0] in month_price.keys() or v[0] in annual_price.keys()}
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(2006, 2022) for m in range(1, 13)]
    start, end = dt_range[0], dt_range[-1]
    missing_state = None

    for code, (crop, count) in counts.items():

        if crop in ['Rye']:
            continue

        print(crop)

        try:
            ts = month_price[crop]
            monthly, annual, national = True, False, False
        except KeyError:
            ts = annual_price[crop]
            monthly, annual = False, True
            national = ts['national']

        c = pd.DataFrame(columns=[s for s in BASIN_STATES])
        c = c.reindex(dt_range)
        c = c.loc['2006-01-01': '2021-12-31']

        if annual and national:
            series = _from_series(ts, 'CA')
            for s in BASIN_STATES:
                c[s] = series

        missing_state = []
        for state in BASIN_STATES:

            st_data = ts[state]

            if annual and not national:
                pass

            for dt, val in st_data:
                pdt = pd.to_datetime(dt)
                if end < pdt or pdt < start:
                    continue
                try:
                    c.loc[pdt, state] = float(val)
                except ValueError:
                    c.loc[pdt, state] = np.nan

            missing = [i for i, r in c[state].items() if np.isnan(r)]
            if missing:
                ts = annual_price[crop]
                series = _from_series(ts, state)
                series = series[~series.index.duplicated()]
                series = series.reindex(dt_range).interpolate()
                series = series.dropna()

                try:
                    c.loc[missing, state] = series.loc[missing].values
                except KeyError:
                    c[state] = c[state].values.astype(float)
                    c[state] = c[state].interpolate(method='linear').ffill().bfill()
                    if np.all(np.isnan(c[state].values)):
                        missing_state.append(state)

        if missing_state:
            mean_ = c.mean(axis=1)
            for state in missing_state:
                c[state] = mean_
        print(c.shape, '\n')

        if 'Other' in crop:
            crop = crop.replace(' ', '_').replace('/', '_')
        o_file = os.path.join(nominal, '{}.csv'.format(crop))
        c.to_csv(o_file)


def normalized_price_data(nominal, ppi, deflated):
    code_table = {v: k for k, v in ppi_to_cdl_crop().items()}

    df = pd.read_csv(ppi, index_col='DATE', infer_datetime_format=True, parse_dates=True)
    df = df.loc['2006-01-01': '2021-12-31']
    scale = df.loc['2021-12-01'] / 100
    df = df / scale
    df.columns = [code_table[k] if k in code_table.keys() else k for k in df.columns]
    df = df.interpolate(method='linear', axis=0, limit=5)
    df = df.dropna(how='any', axis=1)

    l = [os.path.join(nominal, x) for x in os.listdir(nominal)]
    for csv in l:

        crop = os.path.basename(csv).strip('.csv')
        c = pd.read_csv(csv, infer_datetime_format=True, parse_dates=True, index_col=0)

        if crop in df.columns:
            c = c.div(df[crop], axis=0)
        else:
            c = c.div(df['Farm Products'].div(100, axis=0), axis=0)

        for col in c.columns:
            c[col] = spi(c[col], **KWARGS)

        o_file = os.path.join(deflated, '{}.csv'.format(crop))
        print(o_file)
        c = c.loc['2006-01-01':]
        c.to_csv(o_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    key_ = '/home/dgketchum/quickstats_token.json'
    nass_price_monthly = '/media/research/IrrigationGIS/expansion/analysis/nass_price_monthly.json'
    # get_monthly_price_timeseries(key_, nass_price_monthly)

    price_data = '/media/research/IrrigationGIS/expansion/tables/crop_value/nass_data'
    map_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/nass_data/values.csv'
    nass_price_annual = '/media/research/IrrigationGIS/expansion/analysis/nass_price_annual.json'
    # get_annual_price_timeseries(price_data, map_, nass_price_annual, key_)

    nominal_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/nominal'
    # nominal_price_data(nass_price_monthly, nass_price_annual, nominal_)

    ppi_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/ppi/ppi_cdl_monthly.csv'
    deflated = '/media/research/IrrigationGIS/expansion/tables/crop_value/deflated'
    normalized_price_data(nominal_, ppi_, deflated)
# ========================= EOF ====================================================================
