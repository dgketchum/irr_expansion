import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union
import warnings
from urllib.error import HTTPError

from flow.gage_lists import EXCLUDE_RESERVOIRS

warnings.filterwarnings("ignore", category=FutureWarning)


def read_gp_res70(url):
    skip = list(range(10)) + [11]
    txt = requests.get(url).content.decode('utf-8')
    lines = [l for i, l in enumerate(txt.splitlines()) if i not in skip]
    table = []
    for l in lines:
        if l.startswith(' Year'):
            table.append(l.split()[:13])
        try:
            _ = int(l.strip()[:4])
            table.append(l.split()[:13])
        except ValueError:
            continue
    try:
        df = pd.DataFrame(columns=table[0], data=table[1:])
        df = df.melt(id_vars=['Year'], var_name='Month', value_name='storage')
        df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
        df = df.set_index('date').drop(['Year', 'Month'], axis=1)
        df = df.sort_index()
        df['storage'] = pd.to_numeric(df['storage'], errors='coerce')
    except IndexError:
        return False

    return df


def read_pnw(url):
    try:
        df = pd.read_csv(url, infer_datetime_format=True, parse_dates=True, index_col='DateTime')
    except ValueError:
        return False

    df.columns = ['storage']
    df['storage'] = df['storage'].values.astype(float)
    if np.all(np.isnan(df['storage'].values)):
        return False
    return df


def get_reservoir_data(csv, bounds, out_shp, out_dir, resops_dir, start, end):
    gdf = gpd.read_file(bounds)
    bounds = unary_union(gdf['geometry'])

    eom = pd.date_range(start, end, freq='M')

    ucrb_url = 'https://www.usbr.gov/uc/water/hydrodata/reservoir_data/{}/csv/{}.csv'
    ucrb_keys = {'inflow': '29', 'outflow': '42', 'storage': '17',
                 'columns': {'datetime': 'date', 'storage': 'storage'}}

    gp_url = 'https://www.usbr.gov/gp-bin/res070.pl?station={}&parameters={}&byear=1995&eyear=2015'
    gp_keys = {'units': 'af', 'storage': 'AF.EOM', 'inflow': 'AF.IN', 'outflow': 'AF.QD',
               'columns': {'date': 'date', 'storage': 'storage'}}

    pnw_url = 'https://www.usbr.gov/pn-bin/daily.pl?station={}&' \
              'format=csv&year=1987&month=1&day=1&year=2021&month=12&day=31&pcode={}'
    pnw_keys = {'inflow': '29', 'outflow': '42', 'storage': 'af',
                'columns': {'datetime': 'date', 'storage': 'storage'}}

    regions = {'UCRB': (ucrb_url, ucrb_keys), 'GP': (gp_url, gp_keys),
               'CPN': (pnw_url, pnw_keys)}

    stations = gpd.read_file(csv)
    shp_cols = list(stations.columns)
    shp = pd.DataFrame(columns=shp_cols)

    ct = 0

    for i, r in stations.iterrows():
        has_updated, series = True, None
        sid, _name, region, dam_id = r['DAM_ID'], r['DAM_NAME'], r['REGION'], r['DAM_ID']

        # if sid != 413:
        #     continue

        if not _name:
            _name = r['DAM_NAME']

        if not r['geometry'].within(bounds):
            continue

        if region:

            map = regions[region]
            url, key = map[0], map[1]['storage']
            url = url.format(sid, key)

            if region == 'GP':
                df = read_gp_res70(url)
                if df is False:
                    has_updated = False

            elif region == 'UCRB':
                try:
                    df = pd.read_csv(url, index_col='datetime', infer_datetime_format=True, parse_dates=True)
                except HTTPError:
                    has_updated = False
            else:
                df = read_pnw(url)
                if df is False:
                    has_updated = False

            if not has_updated:
                _file = os.path.join(resops_dir, '{}.csv'.format(sid))
                df = pd.read_csv(_file, infer_datetime_format=True, parse_dates=True, index_col=0)

            series = df['storage'] * 1233.48
            series = series.reindex(eom)
            series = series.loc['1982-01-01': '2021-12-31']

        else:
            has_updated = False

        if not has_updated or series.isna().sum() > 0:
            resfile = os.path.join(resops_dir, '{}.csv'.format(r['DAM_ID']))
            resop_df = pd.read_csv(resfile, index_col=0, infer_datetime_format=True, parse_dates=True)
            resop_df = resop_df.reindex(eom)

            if isinstance(series, pd.Series):
                series = resop_df['storage'].fillna(series)
            else:
                series = resop_df['storage']

        na_fraction = series.isna().sum() / series.shape[0]
        ofile = os.path.join(out_dir, '{}.csv'.format(dam_id))
        series.to_csv(ofile)

        try:
            q95, q05 = np.nanpercentile(series, [95, 5])
            s_range = q95 - q05
            dct = r.to_dict()
            dct['s_rng'] = s_range
            dct['s_95'] = q95
            dct['s_05'] = q05
            dct['na_fraction'] = na_fraction
            shp = shp.append(dct, ignore_index=True)
            ct += 1
            print(sid, _name, '{:.1f}'.format(s_range / 1e6))

        except Exception as e:
            print(e, sid, _name)
            continue

    shp = gpd.GeoDataFrame(shp, crs='EPSG:5071')
    shp.to_file(out_shp)


def process_resops_hydrographs(reservoirs, time_series, out_dir, start, end):
    adf = gpd.read_file(reservoirs)
    eom = pd.date_range(start, end, freq='M')
    for i, r in adf.iterrows():
        d = r.to_dict()
        sid = d['DAM_ID']

        ts_file = os.path.join(time_series, 'ResOpsUS_{}.csv'.format(sid))
        df = pd.read_csv(ts_file, index_col='date', infer_datetime_format=True, parse_dates=True)
        df = df.loc[start: end]
        series = df['storage']
        series = series.reindex(eom)
        series.dropna(inplace=True)
        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        series *= 1e6
        series.to_csv(ofile, float_format='%.3f')
        print(sid, d['DAM_NAME'], d['STATE'])


def join_reservoirs_to_basins(basins, reservoirs, ibts, out_json):
    basins = gpd.read_file(basins)
    reservoirs = gpd.read_file(reservoirs)
    ibts = gpd.read_file(ibts)

    res_geo = [r['geometry'] for i, r in reservoirs.iterrows() if r['DAM_ID'] not in EXCLUDE_RESERVOIRS]
    res_id = [r['DAM_ID'] for i, r in reservoirs.iterrows() if r['DAM_ID'] not in EXCLUDE_RESERVOIRS]

    ibt_geo = [r['geometry'] for i, r in ibts.iterrows()]
    ibt_id = [r['STAID'] for i, r in ibts.iterrows()]

    dct = {r['STAID']: {'ibt': [], 'res': []} for i, r in basins.iterrows()}

    for i, r in basins.iterrows():

        g = r['geometry']

        for j, res_g in enumerate(res_geo):
            if res_g.within(g):
                dct[r['STAID']]['res'].append(res_id[j])

        for j, ibt_g in enumerate(ibt_geo):
            if ibt_g.within(g):
                dct[r['STAID']]['ibt'].append(ibt_id[j])

    with open(out_json, 'w') as f:
        json.dump(dct, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'

    s, e = '1982-01-01', '2020-12-31'
    csv_ = os.path.join(root, 'reservoirs/resopsus/attributes/reservoir_flow_summary.shp')
    res_gages = os.path.join(root, 'reservoirs/resopsus/time_series_all')
    processed = os.path.join(root, 'reservoirs/resopsus/time_series_processed')
    # process_resops_hydrographs(csv_, res_gages, processed, s, e)

    s, e = '1982-01-01', '2021-12-31'
    resops_ = os.path.join(root, 'reservoirs/resopsus/time_series_processed')
    sites = os.path.join(root, 'reservoirs/resopsus/attributes/resevoirs_study_area.shp')
    oshp = os.path.join(root, 'reservoirs/usbr/reservoir_sites.shp')
    hyd = os.path.join(root, 'reservoirs/hydrographs')
    study_area = '/media/research/IrrigationGIS/boundaries/expansion_study_area/study_area.shp'
    # get_reservoir_data(sites, bounds=study_area, resops_dir=resops_, out_shp=oshp, out_dir=hyd, start=s, end=e)

    ibt = '/media/research/IrrigationGIS/impacts/canals/ibt_exports_wgs.shp'
    res_ = os.path.join(root, 'reservoirs/resopsus/attributes/resevoirs_study_area_wgs.shp')
    basins = os.path.join(root, 'gages/matched_basins_wgs.shp')
    js = '/media/research/IrrigationGIS/impacts/gages/metadata_res_ibt.json'

    # join_reservoirs_to_basins(basins, res_, ibt, js)

    meta = os.path.join(root, 'metadata', 'irrigated_gage_metadata_26OCT2023.json')
    out = os.path.join(root, 'metadata', 'res_ibt_metadata_26OCT2023.json')

    with open(js, 'r') as f:
        res = json.load(f)

    with open(meta, 'r') as f:
        stations = json.load(f)

    dct = stations.copy()
    for k, v in stations.items():
        dct[k].update(res[k])

    with open(out, 'w') as f:
        json.dump(dct, f, indent=4, sort_keys=False)
# ========================= EOF ====================================================================
