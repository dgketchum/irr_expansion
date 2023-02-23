import os
import json
import random

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from field_points.itype_mapping import itype

BASIN_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def generate_random(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    a = polygon.area
    p = (4 * np.pi * polygon.area) / (polygon.boundary.length ** 2.)

    if a < 1e5 or p < 0.6:
        buff = 0.
        centr = polygon.centroid
        if polygon.contains(centr):
            return centr
    else:
        buff = -100

    ct = 0
    while True:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.buffer(buff).contains(pnt):
            return pnt
        ct += 1
        if ct > 10000:
            print('{:.3f}, {:.3f}, {}'.format(a, p, ct))
            return None


def random_point(poly_dir, irr_cnt, out_rand):
    c = gpd.read_file(irr_cnt)
    c['STATE'] = c['OPENET_ID'].apply(lambda x: x[:2])
    for s in BASIN_STATES:
        d = c[c['STATE'] == s].copy()
        print(s, d.shape[0])
        ids = d['OPENET_ID'].values
        p_file = os.path.join(poly_dir, '{}.shp'.format(s))
        polys = gpd.read_file(p_file)
        polys = polys.loc[polys['geometry'].is_valid, :]
        polys = polys[['OPENET_ID', 'geometry']]
        polys['INCL'] = polys['OPENET_ID'].apply(lambda x: x in ids)
        polys = polys[polys['INCL']]
        polys['point'] = polys['geometry'].apply(lambda x: generate_random(x))
        polys = polys.dropna()
        polys['geometry'] = polys['point']
        polys['rand'] = np.random.rand(polys.shape[0], 1)
        polys = polys[['OPENET_ID', 'rand', 'geometry']]
        polys.to_file(out_rand.format(s), crs='epsg:5071', geometry='geometry')
        print(polys.shape[0], out_rand.format(s))


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


def reclassify_itype(in_dir, out_shp):
    l = [os.path.join(in_dir, '{}_itype.shp'.format(s)) for s in ['co', 'mt', 'ut', 'wa']]
    cols = ['IRRIG_TYPE', 'IType', 'IRR_TYPE', 'Irrigation']
    itype_map = itype()
    df = gpd.GeoDataFrame(columns=['itype', 'geometry'])
    for f, col in zip(l, cols):
        c = gpd.read_file(f)
        c = c[[col, 'geometry']]
        c.dropna(inplace=True)
        c['itype'] = c[col].apply(lambda x: itype_map[x])
        print(c.shape[0], 'feaures in ', os.path.basename(f))
        df = pd.concat([df, c[['itype', 'geometry']]], ignore_index=True)

    df.crs = 'EPSG:4326'
    df.to_crs(epsg=5071)
    df.to_file(out_shp)
    print(df.shape[0], 'features', out_shp)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    inshp = os.path.join(root, 'expansion/shapefiles/openet_centr/irr_tables/')
    fields_ = os.path.join(root, 'openET/OpenET_GeoDatabase_5071')
    odir = '/media/nvm/field_pts/fields_data/fields_shp'
    usbr_attr = '/media/nvm/field_pts/usbr_attr'
    # get_state_fields(inshp, usbr_attr, fields_, odir)

    in_itype = '/media/hdisk/itype/field_boundaries/fields_wgs'
    out_itype = '/media/hdisk/itype/field_boundaries/fields_aea/itype_join.shp'
    reclassify_itype(in_itype, out_itype)
# ========================= EOF ====================================================================
