import os
import random

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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
    for s in BASIN_STATES[2:]:
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
        df = gdf[gdf['STUSPS'] == s]
        out_ = os.path.join(out_dir, '{}.csv'.format(s))
        df.to_csv(out_)
        print(out_, df.shape[0])


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    poly_dir_ = os.path.join(root, 'openET', 'OpenET_GeoDatabase_5071')

    centr_dir = os.path.join(root, 'expansion', 'shapefiles', 'openet_field_centr_irr_gt19',
                             'study_basins_openet_irr_centr_wKlamath.shp')

    out_randn = os.path.join(root, 'expansion', 'shapefiles', 'openet_field_centr_irr_gt19', 'state_openet_points',
                             'study_basins_openet_irr_randr_wKlamath_{}.shp')

    random_point(poly_dir_, centr_dir, out_randn)
    #
    # inshp = os.path.join(root, 'expansion/shapefiles/openet_field_centr_irr_gt19/field_pts_attr_8FEB2023.shp')
    # odir = '/media/nvm/field_pts/metadata'
    # state_csv(inshp, odir)
# ========================= EOF ====================================================================
