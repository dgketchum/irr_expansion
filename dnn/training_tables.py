import os
from collections import OrderedDict
from copy import deepcopy

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from shapely.geometry import shape, mapping

from training_data import PROPS, BASIN_STATES

DROP = ['left', 'right', 'top', 'bottom']
UNCULT_CDL = [152,
              176]

NLCD_UNCULT = [31,
               41,
               42,
               43,
               52,
               71]

STATICS = ['aspect', 'awc', 'clay', 'ksat', 'lat', 'lon', 'sand', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']


def join_csv_shapefile(in_shp, csv, out_shape, join_feat='OPENET_ID', find='uncult'):
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
    if find == 'uncult':
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


def validate_predictions(csv_dir, out_csv, join_key='id', glob='None'):

    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if glob in x]
    file_dct = {s: [c for c in l if s in c] for s in BASIN_STATES}
    cols = ['eff_ppt_{}'.format(n) for n in range(4, 11)]
    cols = cols + ['et_{}'.format(n) for n in range(4, 11)]
    target_cols = ['{}_{}'.format(c, y) for c in cols for y in range(1987, 2022)]

    mdf = pd.DataFrame(columns=['ept_sum', 'et_sum'])
    for st, l in file_dct.items():
        first = True
        for csv in l:
            splt = os.path.basename(csv).split('_')
            y = int(splt[-1].split('.')[0])

            print(os.path.basename(csv))

            try:
                if first:
                    df = pd.read_csv(csv)
                    df = df[df['slope'] < 3]
                    df.index = df['id'].apply(lambda x: '{}_{}'.format(st, x))
                    df.drop(columns=[join_key], inplace=True)
                    df = df[cols]
                    df.columns = ['{}_{}'.format(col, y) for col in list(df.columns)]
                    dummy = pd.DataFrame(columns=['ept_sum', 'et_sum'], index=df.index)
                    mdf = pd.concat([mdf, dummy], ignore_index=False)
                    first = False
                else:
                    c = pd.read_csv(csv)
                    c = c[c['slope'] < 3]
                    c.index = c['id'].apply(lambda x: '{}_{}'.format(st, x))
                    c.drop(columns=[join_key], inplace=True)
                    c = c[cols]
                    c.columns = ['{}_{}'.format(col, y) for col in list(c.columns)]
                    df = pd.concat([df, c], axis=1)

            except pd.errors.EmptyDataError:
                print('{} is empty'.format(csv))
                pass

        assert np.all([c in df.columns for c in target_cols])
        ept_cols, et_cols = [x for x in df.columns if 'eff_ppt' in x], [x for x in df.columns if 'et_' in x]
        df['ept_sum'] = df[ept_cols].sum(axis=1)
        df['et_sum'] = df[et_cols].sum(axis=1)
        match = [i for i in mdf.index if i in df.index]
        mdf.loc[match, ['ept_sum', 'et_sum']] = df.loc[match, ['ept_sum', 'et_sum']] / 35
        mdf['diff'] = (mdf['et_sum'] - mdf['ept_sum']) / mdf['et_sum']

    rmse = mean_squared_error(mdf['et_sum'], mdf['ept_sum'], squared=False)
    mean_ = mdf['et_sum'].mean()
    print(mean_, rmse)
    df.to_csv(out_csv)


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


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    c_ = os.path.join(root, 'tables/validation/initial_filter')
    s_ = os.path.join(root, 'shapefiles/training_data/validation', 'validation_pts_initial_28FEB2023.shp')
    oshp = os.path.join(root, 'shapefiles/training_data/validation', 'validation_pts_28FEB2023.shp')
    # join_csv_shapefile(s_, c_, oshp, join_feat='id', find='uncult')

    v_csv = '/media/research/IrrigationGIS/expansion/tables/validation/csv/'
    o_csv = '/media/research/IrrigationGIS/expansion/tables/validation/results/'
    validate_predictions(v_csv, o_csv, join_key='id', glob='validation_28FEB2023')
# ========================= EOF ====================================================================
