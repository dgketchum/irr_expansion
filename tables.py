import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

drop = ['left', 'right', 'top', 'bottom']


def join_shp_csv(in_shp, csv_dir, out_shp, join_on='id', glob='.csv', **rename_map):
    gdf = gpd.read_file(in_shp)
    gdf.index = [int(i) for i in gdf['id']]
    gdf.drop(columns=drop, inplace=True)

    csv_l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith(glob)]
    first = True
    for csv in csv_l:
        y = int(csv.split('.')[0][-4:])
        try:
            if first:
                df = pd.read_csv(csv, index_col=join_on)
                if rename_map:
                    df = df.rename(columns={'mean': 'irr_{}'.format(y)})
                print(df.shape, csv)
                first = False
            else:
                c = pd.read_csv(csv, index_col=join_on)
                if rename_map:
                    c = c.rename(columns={'mean': 'irr_{}'.format(y)})
                df = pd.concat([df, c['irr_{}'.format(y)]], axis=1)
                print(c.shape, csv)
        except pd.errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    geo = [gdf.loc[i].geometry for i in df.index]
    df = gpd.GeoDataFrame(df, crs=gdf.crs, geometry=geo)
    df.to_file(out_shp)


def select_grid_points(fields_dir, study_area, out_shp):
    western_11 = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    first, df = True, None
    study_area = gpd.read_file(study_area)
    ct = 1
    if not os.path.exists(out_shp):
        for s in western_11:
            print(s)
            gdf = gpd.read_file(os.path.join(fields_dir, '{}.shp'.format(s)), crs=study_area.crs)
            if gdf.shape[0] > 1e5:
                frac = 0.01
            else:
                frac = 0.1
            gdf = gdf.sample(frac=frac)
            gdf['id'] = [x for x in range(ct, ct + gdf.shape[0])]
            gdf = gdf[['id', 'geometry']]
            gdf['geometry'] = [x.centroid for x in gdf['geometry']]
            within = gpd.sjoin(gdf, study_area, predicate='within')
            if first:
                df = within[['id', 'geometry']]
                ct += df.shape[0]
                first = False
            else:
                c = within[['id', 'geometry']]
                df = gpd.GeoDataFrame(pd.concat([df, c], ignore_index=True), crs=gdf.crs)
        df = df.set_geometry('geometry').to_crs('epsg:102008')
        df.to_file(out_shp)


def prep_extracts(in_dir, out_dir):

    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    for c in l:
        print(os.path.basename(c))
        df = pd.read_csv(c)
        print(df.shape)
        df.dropna(inplace=True)
        print(df.shape)
        df.to_csv(os.path.join(out_dir, os.path.basename(c)), index=False)


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS/expansion')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS/expansion'

    grids = os.path.join(gis, 'grid')
    shp = os.path.join(grids, 'grid_5km_uinta.shp')
    out_shp = os.path.join(grids, 'grid_5km.shp')
    tables = os.path.join(gis, 'tables', 'points_extracts')
    csv = os.path.join(tables, 'uinta_2020.csv')
    join_shp_csv(shp, grids, out_shp, glob='study_uncult.csv', rename_map=None)

    grid_pts_ = os.path.join(grids, 'grid_5km_uncult_attr_wgs.shp')
    fields_dir_ = '/media/research/IrrigationGIS/openET/OpenET_GeoDatabase'
    study_area_ = '/media/research/IrrigationGIS/expansion/shapefiles/study_area_wgs.shp'
    out_shp_ = '/media/research/IrrigationGIS/expansion/shapefiles/study_area_field_centroids.shp'
    # select_grid_points(fields_dir_, study_area_, out_shp_)

# ========================= EOF ====================================================================
