import os
import random

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASIN_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def generate_random(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(10):
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.buffer(-100).contains(pnt):
            return pnt


def random_point(poly_dir, irr_cnt, out_rand):
    c = gpd.read_file(irr_cnt)
    c['STATE'] = c['OPENET_ID'].apply(lambda x: x[:2])
    for s in BASIN_STATES:
        print(s)
        d = c[c['STATE'] == s].copy()
        ids = d['OPENET_ID'].values
        p_file = os.path.join(poly_dir, '{}.shp'.format(s))
        polys = gpd.read_file(p_file)
        polys = polys[['OPENET_ID', 'geometry']]
        polys['INCL'] = polys['OPENET_ID'].apply(lambda x: x in ids)
        polys = polys[polys['INCL']]
        polys['point'] = polys['geometry'].apply(lambda x: generate_random(x))
        polys['geometry'] = polys['point']
        polys = polys[['OPENET_ID', 'geometry']]
        polys.to_file(out_rand.format(s), crs='epsg:5071', geometry='geometry')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    poly_dir_ = os.path.join(root, 'openET', 'OpenET_GeoDatabase_5071')
    centr_dir = os.path.join(root, 'expansion', 'shapefiles', 'openet_field_centr_irr_gt19',
                             'study_basins_openet_irr_centr_wKlamath.shp')
    out_randn = os.path.join(root, 'expansion', 'shapefiles', 'openet_field_centr_irr_gt19',
                             'study_basins_openet_irr_randr_wKlamath_{}.shp')
    random_point(poly_dir_, centr_dir, out_randn)
# ========================= EOF ====================================================================
