import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, shape, mapping
from rasterstats import zonal_stats

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


def join_csv_shapefile(in_shp, csv, out_shape, min_irr_years=20):
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf['id']
    gdf.drop(columns=['id'], inplace=True)
    df = pd.read_csv(csv, index_col='id')
    gdf = gdf.loc[df.index]
    cols = list(df.columns)
    gdf[cols] = df[cols]
    gdf = gdf[gdf['slope'] < 8]
    gdf = gdf[gdf['nlcd'].apply(lambda x: True if x in NLCD_UNCULT else False)]
    gdf = gdf[gdf['cdl'].apply(lambda x: True if x in UNCULT_CDL else False)]
    gdf = gdf[gdf['uncult'].apply(lambda x: True if x >= 30 else False)]
    gdf.to_file(out_shape)
    print(out_shape)


def prep_extracts(in_dir, out_dir, clamp_et=False):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    for c in l:
        print(os.path.basename(c))
        df = pd.read_csv(c)
        print(df.shape)
        yr = int(os.path.basename(c).split('.')[0][-4:])
        df['ppt'] = df[['ppt_{}_{}'.format(yr, m) for m in range(1, 10)] +
                       ['ppt_{}_{}'.format(yr - 1, m) for m in range(10, 13)]].sum(axis=1)
        et_cols = ['et_{}_{}'.format(yr, mm) for mm in range(4, 11)]
        df['season'] = df[et_cols].sum(axis=1) * 0.00001

        if clamp_et:
            df = df[df['season'] < df['ppt']]

        df.dropna(inplace=True)
        print(df.shape)
        df.to_csv(os.path.join(out_dir, os.path.basename(c)), index=False)


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


if __name__ == '__main__':
    root = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    pts = os.path.join(root, 'expansion', 'shapefiles', 'points_29DEC2022')
    ishp = os.path.join(pts, 'random_points.shp')
    c = os.path.join(pts, 'sample_pts_29DEC2022.csv')
    oshp = os.path.join(pts, 'sample_pts_29DEC2022.shp')
    join_csv_shapefile(ishp, c, oshp)

    extracts = os.path.join(root, 'expansion', 'tables', 'band_extracts', 'bands_28DEC2022')
    bands_out = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_28DEC2022')
    # prep_extracts(extracts, bands_out, clamp_et=True)

# ========================= EOF ====================================================================
