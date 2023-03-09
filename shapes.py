import os
import json
import random

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from gridded_data import BASIN_STATES


def state_csv(shape, out_dir):
    gdf = gpd.read_file(shape)
    for s in BASIN_STATES:
        df = gdf[gdf['STUSPS'] == s].copy()
        df.loc[pd.isna(df['name']), 'name'] = 'none'
        df.loc[pd.isna(df['usbrid']), 'usbrid'] = 0
        df['usbrid'] = df['usbrid'].values.astype(np.int64)
        df = df[['OPENET_ID', 'usbrid', 'name']]
        out_ = os.path.join(out_dir, '{}.csv'.format(s))
        df.to_csv(out_, index=False)
        print(out_, df.shape[0])


def get_state_fields(points, attrs, fields, out_dir):
    dct = {}
    for s in BASIN_STATES:

        cfile = os.path.join(points, '{}_openet_centr_16FEB2023.csv'.format(s))
        try:
            df = pd.read_csv(cfile, index_col='OPENET_ID')
        except FileNotFoundError:
            print(cfile, 'does not exist')
            continue

        afile = os.path.join(attrs, '{}.csv'.format(s))
        adf = pd.read_csv(afile, index_col='OPENET_ID')
        match = [i for i in df.index if i in adf.index]

        if len(match) > 0:
            df.loc[match, 'usbrid'] = adf.loc[match, 'usbrid']
            df['usbrid'] = df['usbrid'].fillna(0)
        else:
            df['usbrid'] = [0 for _ in range(df.shape[0])]

        print(df.shape[0], 'points', s)
        df = df[df['irr'] > 9]
        print(df.shape[0], 'irr points', s)
        pfile = os.path.join(fields, '{}.shp'.format(s))
        poly = gpd.read_file(pfile)
        poly.index = poly['OPENET_ID']
        print(poly.shape[0], 'fields')
        match = [i for i in df.index if i in poly.index]
        df.loc[match, 'geometry'] = poly.loc[match, 'geometry']
        df.loc[match, 'MGRS_TILE'] = poly.loc[match, 'MGRS_TILE']
        print(df.shape[0], 'target fields')
        print(len(match), 'index match')
        tiles = [x for x in list(set(df['MGRS_TILE'])) if isinstance(x, str)]
        print(tiles)
        dct[s] = tiles
        df = gpd.GeoDataFrame(df, geometry=poly.geometry)
        df.loc[pd.isna(df['usbrid']), 'usbrid'] = 0
        df['usbrid'] = df['usbrid'].values.astype(np.int64)
        df = df[['usbrid', 'MGRS_TILE', 'geometry']]
        out_ = os.path.join(out_dir, '{}.shp'.format(s))
        df.to_file(out_, crs='EPSG:5071')
        df = df[['usbrid', 'MGRS_TILE']]
        df.to_csv(out_.replace('.shp', '.csv'))
        print(out_, df.shape[0], '\n')


def field_areas(fields, out_json):
    dct = {}
    for s in BASIN_STATES:
        print(s)
        cfile = os.path.join(fields, '{}.shp'.format(s))
        df = gpd.read_file(cfile, index_col='OPENET_ID')
        for i, r in df.iterrows():
            dct[r['OPENET_ID']] = r['geometry'].area / 1e6

    with open(out_json, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    inshp = os.path.join(root, 'expansion/shapefiles/openet_centr/irr_tables/')
    fields_ = os.path.join(root, 'openET/OpenET_GeoDatabase_5071')
    fields_shp = '/media/nvm/field_pts/fields_data/fields_shp'
    usbr_attr = '/media/nvm/field_pts/usbr_attr'
    # get_state_fields(inshp, usbr_attr, fields_, fields_shp)

    fields_area = '/media/research/IrrigationGIS/expansion/analysis/fields_area.json'
    field_areas(fields_shp, fields_area)

# ========================= EOF ====================================================================
