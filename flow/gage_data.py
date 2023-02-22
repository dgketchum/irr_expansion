import os
import json
from datetime import date
from copy import deepcopy
from calendar import monthrange
from itertools import groupby, islice, tee
from operator import itemgetter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import dataretrieval.nwis as nwis


class MissingValues(Exception):
    pass


def hydrograph(c):
    df = pd.read_csv(c)

    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = pd.to_datetime(df['dt'])
    except KeyError:
        df['dt'] = pd.to_datetime(df['datetimeUTC'])
    df = df.set_index('dt')
    try:
        df.drop(columns='datetimeUTC', inplace=True)
    except KeyError:
        pass
    try:
        df = df.tz_convert(None)
    except:
        pass

    if 'USGS' in list(df.columns)[0]:
        df = df.rename(columns={list(df.columns)[0]: 'q'})

    return df


def get_station_daily_data(start, end, stations, out_dir, plot_dir=None, overwrite=False):
    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    with open(stations, 'r') as f_obj:
        stations = json.load(f_obj)

    for sid, data in stations.items():

        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        df = nwis.get_record(sites=sid, service='dv', start=start, end=end, parameterCd='00060')
        df = df.tz_convert(None)

        if df.empty:
            print(sid, ' is empty')
            continue

        q_col = '00060_Mean'
        df = df.rename(columns={q_col: 'q'})
        df = df.reindex(pd.DatetimeIndex(dt_range), axis=0)

        df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
        nan_count = np.count_nonzero(np.isnan(df['q']))

        # exclude months without complete data
        if nan_count > 0:
            df['q'] = df['q'].dropna(axis=0)
            record_ct = df['q'].groupby([df.index.year, df.index.month]).agg('count')
            records = [r for i, r in record_ct.items()]
            mask = [0] + [int(a == b) for a, b in zip(records, counts)]
            missing_mo = len(counts) - sum(mask)
            resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)
            mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(resamp_start, end, freq='M')),
                             data=mask).resample('D').bfill()
            mask = mask[1:]
            df = df.loc[mask[mask == 1].index, 'q']
            print('write {:.1f}'.format(data['AREA']), sid, 'missing {} months'.format(missing_mo), data['STANAME'])
        else:
            df = df['q']
            print('write {:.1f}'.format(data['AREA']), sid, data['STANAME'])

        df.to_csv(out_file)

        if plot_dir:
            plt.plot(df.index, df)
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()


def get_station_daterange_data(daily_q_dir, aggregate_q_dir, resample_freq='A', convert_to_mcube=True, plot_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    out_records, short_records = [], []
    for sid, c in zip(sids, q_files):

        df = pd.read_csv(c, index_col=0, infer_datetime_format=True, parse_dates=True)

        # cfs to m ^3 d ^-1
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(pd.DataFrame.sum, skipna=False)
        dates = deepcopy(df.index)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

        if plot_dir:
            pdf = pd.DataFrame(data={'Date': dates, 'q': df.values})
            pdf.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()

    print('{} processed'.format(len(out_records)))
    print(out_records)


def find_lowest_flows(shapes, monthly_q, out_shape):
    gdf = gpd.read_file(shapes)
    gdf.index = gdf['STAID']

    years = range(1987, 2022)
    months = list(range(4, 11))

    periods = [(m,) for m in months] + [list(consecutive_subseq(months, x)) for x in range(2, 7)]
    periods = [item for sublist in periods for item in sublist]
    periods = [i if isinstance(i, tuple) else (i, i) for i in periods]
    str_pers = ['{}-{}'.format(q_win[0], q_win[-1]) for q_win in periods]
    nan_data = [np.nan for i, r in gdf.iterrows()]
    for p in str_pers:
        gdf[p] = nan_data

    for sid, data in gdf.iterrows():
        _file = os.path.join(monthly_q, '{}.csv'.format(sid))

        try:
            df = hydrograph(_file)
        except FileNotFoundError:
            print(sid, 'not found')
            continue

        df['ppt'], df['etr'] = df['gm_ppt'], df['gm_etr']
        df = df[['q', 'etr', 'ppt', 'cc']]
        df[df == 0.0] = np.nan
        for q_win in periods:
            key_ = '{}-{}'.format(q_win[0], q_win[-1])
            q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
            q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
            finite = np.isfinite(q)

            if sum(finite) != len(q):
                df.loc[sid, key_] = int(0)

            q = q[finite]
            y = np.array(years)[finite]
            _min_year = y[np.argmin(q)].item()
            gdf.loc[sid, key_] = int(_min_year)
            print(_min_year, sid, key_, 'of {} years'.format(len(y)))

    vals = gdf[str_pers]
    vals[np.isnan(vals)] = 0.0
    vals = vals.astype(int)
    gdf[str_pers] = vals
    gdf.drop(columns=['STAID'], inplace=True)
    gdf.to_file(out_shape)
    df = pd.DataFrame(gdf)
    df.drop(columns=['geometry'], inplace=True)
    df.to_csv(out_shape.replace('.shp', '.csv'))


def consecutive_subseq(iterable, length):
    for _, consec_run in groupby(enumerate(iterable), lambda x: x[0] - x[1]):
        k_wise = tee(map(itemgetter(1), consec_run), length)
        for n, it in enumerate(k_wise):
            next(islice(it, n, n), None)  # consume n items from it
        yield from zip(*k_wise)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
    figures = os.path.join(root, 'figures')

    daily_q = os.path.join(root, 'tables', 'hydrographs', 'daily_q')
    daily_q_fig = os.path.join(figures, 'hydrographs', 'daily_hydrograph_plots')

    monthly_q = os.path.join(root, 'tables', 'hydrographs', 'monthly_q')
    monthly_q_fig = os.path.join(figures, 'hydrographs', 'monthly_hydrograph_plots')

    start_yr, end_yr = 1984, 2021
    months = list(range(1, 13))
    get_station_daily_data('{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr), gages_metadata,
                           daily_q, plot_dir=daily_q_fig, overwrite=True)
    get_station_daterange_data(daily_q, monthly_q, convert_to_mcube=True, resample_freq='M', plot_dir=monthly_q_fig)

    d = os.path.join(root, 'expansion')
    data_ = os.path.join(d, 'irrigated_gage_metadata.json')
    _shape_out = os.path.join(d, 'analysis', 'min_flows.shp')
    _points = os.path.join(d, 'gages', 'selected_gages.shp')
    data_tables = os.path.join(root, 'impacts', 'tables',
                               'input_flow_climate_tables',
                               'IrrMapperComp_21OCT2022')
    # find_lowest_flows(_points, data_tables, _shape_out)

# ========================= EOF ====================================================================
