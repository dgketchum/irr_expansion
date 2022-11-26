import os

import pandas as pd
import geopandas as gpd


drop = ['left', 'right', 'top', 'bottom']


def join_shp_csv(in_shp, csv_dir, out_shp, join_on='id'):

    gdf = gpd.read_file(in_shp)
    gdf.index = [int(i) for i in gdf['id']]
    gdf.drop(columns=drop, inplace=True)

    csv_l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith('.csv')]
    first = True
    for csv in csv_l:
        y = int(csv.split('.')[0][-4:])
        try:
            if first:
                df = pd.read_csv(csv, index_col=join_on)
                df = df.rename(columns={'mean': 'irr_{}'.format(y)})
                print(df.shape, csv)
                first = False
            else:
                c = pd.read_csv(csv, index_col=join_on)
                c = c.rename(columns={'mean': 'irr_{}'.format(y)})
                df = pd.concat([df, c['irr_{}'.format(y)]], axis=1)
                print(c.shape, csv)
        except pd.errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    geo = [gdf.loc[i].geometry for i in df.index]
    df = gpd.GeoDataFrame(df, crs=gdf.crs, geometry=geo)
    df.to_file(out_shp)


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS/expansion')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS/expansion'

    grids = os.path.join(gis, 'grid')
    shp = os.path.join(grids, 'grid_5km_uinta.shp')
    out_shp = os.path.join(grids, 'grid_5km_uinta_data.shp')
    tables = os.path.join(gis, 'tables', 'points_extracts')
    join_shp_csv(shp, tables, out_shp)
# ========================= EOF ====================================================================
