import os

import geopandas as gpd
import pandas as pd

from field_points.itype_mapping import itype


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

    in_itype = '/media/hdisk/itype/field_boundaries/fields_wgs'
    out_itype = '/media/hdisk/itype/field_boundaries/fields_aea/itype_join.shp'
    # reclassify_itype(in_itype, out_itype)

# ========================= EOF ====================================================================
