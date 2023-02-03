import os
from copy import deepcopy
from collections import OrderedDict
from pprint import pprint

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, shape, mapping
from rasterstats import zonal_stats

from gage_data import hydrograph
from call_ee import PROPS

DROP = ['left', 'right', 'top', 'bottom']
UNCULT_CDL = [62,
              63,
              64,
              65,
              131,
              141,
              142,
              143,
              152,
              176]

NLCD_UNCULT = [31,
               41,
               42,
               43,
               52,
               71]

STATICS = ['aspect', 'awc', 'clay', 'ksat', 'lat', 'lon', 'sand', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']


def join_csv_shapefile(in_shp, csv, out_shape, join_feat='OPENET_ID'):
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[join_feat]
    gdf.drop(columns=[join_feat], inplace=True)
    if os.path.isdir(csv):
        l = [os.path.join(csv, x) for x in os.listdir(csv) if x.endswith('.csv')]
        first = True
        for c in l:
            if first:
                df = pd.read_csv(c, index_col=join_feat)
                first = False
            else:
                d = pd.read_csv(c, index_col=join_feat)
                df = pd.concat([df, d])

    else:
        df = pd.read_csv(csv, index_col=join_feat)
    gdf = gdf.loc[df.index]
    cols = list(df.columns)
    gdf[cols] = df[cols]
    gdf = gdf[gdf['slope'] < 8]
    gdf = gdf[gdf['nlcd'].apply(lambda x: True if x in NLCD_UNCULT else False)]
    gdf = gdf[gdf['cdl'].apply(lambda x: True if x in UNCULT_CDL else False)]
    gdf = gdf[gdf['uncult'].apply(lambda x: True if x >= 30 else False)]
    gdf.to_file(out_shape)
    print(out_shape)


def dataset_stats(csv_dir, out_dir, icol='OPENET_ID', glob=None):
    mdf = pd.DataFrame()
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if glob in x and x.endswith('.csv')]

    for p in PROPS:
        print(p)
        sdf, first = None, True
        for i, c in enumerate(l):
            try:
                df = pd.read_csv(c, index_col=icol)
            except ValueError:
                continue
            if first:
                sdf = pd.DataFrame(df[p] * 0.001)
                if p in STATICS:
                    break
                first = False
            else:
                s = df[p] * 0.001
                sdf = pd.concat([sdf, s], axis=1, ignore_index=False)

        mdf[p] = (sdf.min()[p], sdf.mean()[p], sdf.max()[p])
    all_file = os.path.join(out_dir, '{}.csv'.format(glob))
    print(sdf.shape)
    mdf.to_csv(all_file, index=False)


def prep_extracts(in_dir, out_dir, clamp_et=False, indirect=False):
    ldata, years = [], list(range(1987, 2021))
    years.reverse()

    for yr in years:
        l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('{}.csv'.format(yr))]
        sdf, first = None, True

        if not len(l) == 10:
            print(yr, 'short')

        for c in l:
            df = pd.read_csv(c)
            et_cols = ['et_{}_{}'.format(yr, mm) for mm in range(4, 11)]
            df['season'] = df[et_cols].sum(axis=1) * 0.00001

            if clamp_et:
                df = df[df['season'] < df['ppt_wy_et'] * 0.001]
                DROP.append('season')

            ppt_wy_et = df['ppt_wy_et'] * 0.001
            if indirect:
                for etc in et_cols:
                    df[etc] = df[etc].values.astype(float) * 0.00001 / ppt_wy_et

            df.dropna(inplace=True)
            try:
                d = ['.geo', 'system:index']
                df.drop(columns=d, inplace=True)
            except KeyError:
                d = ['.geo', 'system:index', 'STUSPS', 'season']
                df.drop(columns=d, inplace=True)

            if first:
                sdf = deepcopy(df)
                first = False
            else:
                sdf = pd.concat([sdf, df], axis=0, ignore_index=True)

        fname = '{}_{}.csv'.format(os.path.basename(out_dir), yr)
        all_file = os.path.join(out_dir, fname)
        print(sdf.shape, fname)
        sdf.to_csv(all_file, index=False)
        ldata.append(sdf.shape[0])

    print('av len data: {:.3f}'.format(np.mean(ldata)))


def initial_cdl_filter(cdl_dir, in_shape, out_shape):
    gdf = gpd.read_file(in_shape, index_col='id')
    gdf['cdl'] = [0 for _ in gdf.index]
    gdf['keep'] = [False for _ in gdf.index]
    idx = [i for i, r in gdf.iterrows() if isinstance(r['STUSPS'], str)]
    gdf = gdf.loc[idx]
    states = sorted(list(set(gdf['STUSPS'])))
    for st in states:
        df = gdf[gdf['STUSPS'] == st]
        print(st, df.shape[0])
        in_raster = os.path.join(cdl_dir, 'CDL_2017_{}.tif'.format(st))
        geos = [Point(r.coords) for r in df['geometry']]
        stats = zonal_stats(geos, in_raster, stats=['majority'], nodata=0.0)
        gdf.loc[df.index, 'cdl'] = [int(d['majority']) if d['majority'] else 0 for d in stats]
        d_name = os.path.dirname(out_shape)
        [os.remove(os.path.join(d_name, x)) for x in os.listdir(d_name) if 'temp' in x]
    gdf['keep'] = gdf['cdl'].apply(lambda x: True if x in UNCULT_CDL else False)
    gdf = gdf[gdf['keep']]
    gdf.to_file(out_shape)


def centroid_strip_attr(in_shp, out_shp):
    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        feats = []
        for f in src:
            try:
                centroid = shape(f['geometry']).centroid
            except AttributeError:
                continue
            feat = {'type': 'Feature', 'properties': {'OPENET_ID': f['properties']['OPENET_ID']},
                    'geometry': mapping(centroid)}
            feats.append(feat)
    meta['schema']['properties'] = OrderedDict([('OPENET_ID', 'str:80')])
    meta['schema'].update({'geometry': 'Point'})

    with fiona.open(out_shp, 'w', **meta) as dst:
        for f in feats:
            dst.write(f)

    print(out_shp)


def merge_gridded_flow_data(gridded_dir, out_dir, flow_dir=None, start_year=1987, end_year=2021, glob='glob',
                            join_key='STAID'):
    missing, missing_ct, processed_ct = [], 0, 0

    l = [os.path.join(gridded_dir, x) for x in os.listdir(gridded_dir) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])

        try:
            if first:
                df = pd.read_csv(csv, index_col=join_key)
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = pd.read_csv(csv, index_col=join_key)
                if y < start_year:
                    c['irr'] = [np.nan for _ in range(c.shape[0])]
                    c['et'] = [np.nan for _ in range(c.shape[0])]
                    c['ept'] = [np.nan for _ in range(c.shape[0])]
                    c['ietr'] = [np.nan for _ in range(c.shape[0])]
                    c['cc'] = [np.nan for _ in range(c.shape[0])]
                cols = list(c.columns)
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
                df = pd.concat([df, c], axis=1)

        except pd.errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df = df.copy()
    df['{}_STR'.format(join_key)] = [str(x).rjust(8, '0') for x in list(df.index.values)]

    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(start_year), '{}-12-31'.format(end_year)
    idx = pd.DatetimeIndex(pd.date_range(s, e, freq='M'))

    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]

    for d in dfd:
        try:
            sta = d['{}_STR'.format(join_key)]

            irr, cc, et, ietr, ept = [], [], [], [], []
            for y, m in months:
                try:
                    cc_, et_ = d['cc_{}_{}'.format(y, m)], d['et_{}_{}'.format(y, m)]
                    ietr_, ept_ = d['ietr_{}_{}'.format(y, m)], d['eff_ppt_{}_{}'.format(y, m)]
                    irr_ = d['irr_{}_{}'.format(y, m)]
                    cc.append(cc_)
                    et.append(et_)
                    ietr.append(ietr_)
                    ept.append(ept_)
                    irr.append(irr_)
                except KeyError:
                    cc.append(np.nan)
                    et.append(np.nan)
                    ietr.append(np.nan)
                    ept.append(np.nan)
                    irr.append(np.nan)

            irr = irr, 'irr'
            cc = cc, 'cc'
            et = et, 'et'
            ept = ept, 'ept'
            ietr = ietr, 'ietr'

            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue

            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'

            recs = pd.DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, ietr, ept]]), index=idx)

            if flow_dir:
                q_file = os.path.join(flow_dir, '{}.csv'.format(sta))
                qdf = hydrograph(q_file)
                h = pd.concat([qdf, recs], axis=1)
            else:
                h = recs

            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            processed_ct += 1

            print(file_name)

        except FileNotFoundError:
            missing_ct += 1
            print(sta, 'not found')
            missing.append(sta)

    print(processed_ct, 'processed')
    print(missing_ct, 'missing')
    pprint(missing)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    basin_extracts = os.path.join(root, 'tables', 'gridded_tables', 'ietr_nonreclamation_24JAN2023')
    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'ietr_nonreclamation_24JAN2023')
    # hydrographs_ = os.path.join(root, 'tables', 'hydrographs', 'monthly_q')
    # merge_gridded_flow_data(basin_extracts, merged, flow_dir=None,
    #                         glob='ietr_nonreclamation_24JAN2023', join_key='FID')

    d = os.path.join(root, 'shapefiles', 'uncult_addition_2FEB2023')
    data = os.path.join(d, 'uncult_data')
    oshp = os.path.join(d, 'uncult_pts_mod_2FEB2023.shp')
    ishp = os.path.join(d, 'uncult_pts_mod_30JAN2023.shp')
    join_csv_shapefile(ishp, data, oshp, join_feat='id')


# ========================= EOF ====================================================================
