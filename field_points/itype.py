import os
import json

import geopandas as gpd
import pandas as pd
import numpy as np
from pandarallel import pandarallel

from field_points.itype_mapping import itype, itype_integer_mapping


def get_openet_itype(in_dir, out_dir):
    df = None
    with open('tiles.json', 'r') as f_obj:
        tiles_dct = json.load(f_obj)
    ct = 0
    for state in ['CO', 'MT', 'NM', 'UT', 'WA', 'WY']:

        tiles = tiles_dct[state]
        l = [os.path.join(in_dir, 'openet_itype_{}_{}_2021.csv'.format(state, tile)) for tile in tiles]
        first = True
        for f_ in l:
            c = pd.read_csv(f_, index_col='OPENET_ID')
            c.dropna(subset=['mode'], inplace=True)
            if first:
                df = c.copy()
                first = False
            else:
                df = pd.concat([df, c])

        o_file = os.path.join(out_dir, '{}.csv'.format(state))
        df['itype'] = df['mode'].values.astype(int)
        df = df[['itype']]
        df.to_csv(o_file)
        print(df.shape[0], state, os.path.basename(o_file))
        ct += df.shape[0]
        print(np.unique(df['itype'].values, return_counts=True), '\n')

    print('{} fields'.format(ct))


def reclassify_itype(in_dir, out_shp):
    l = [os.path.join(in_dir, '{}.shp'.format(s)) for s in ['co', 'mt', 'nm', 'ut', 'wa', 'wy']]
    cols = ['IRRIG_TYPE', 'IType', 'Source_i_1', 'IRR_TYPE', 'Irrigation', 'Source_i_1']
    itype_map = itype()
    itype_int_map = itype_integer_mapping()
    df = gpd.GeoDataFrame(columns=['itype', 'geometry'])
    first = True
    for f, col in zip(l, cols):
        c = gpd.read_file(f)
        c = c[[col, 'geometry']]
        c.dropna(inplace=True)
        c['itype'] = c[col].apply(lambda x: itype_map[x])
        c['itype'] = c['itype'].apply(lambda x: itype_int_map[x])
        print(c.shape[0], col, 'feaures in ', os.path.basename(f))
        if first:
            c.drop(columns=[col], inplace=True)
            df = c.copy()
            first = False
        else:
            df = gpd.GeoDataFrame(pd.concat([df, c[['itype', 'geometry']]], ignore_index=True), crs=c.crs)

    df.to_file(out_shp)
    print(df.shape[0], 'features', out_shp)


def bearing(a, b):
    lat1 = np.radians(a[0])
    lat2 = np.radians(b[0])

    diffLong = np.radians(b[1] - a[1])

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
                                       * np.cos(lat2) * np.cos(diffLong))

    return np.arctan2(x, y)


def popper(geometry):
    p = (4 * np.pi * geometry.area) / (geometry.boundary.length ** 2.)
    return p


def find_arcs(g):
    min_arc = 8
    tol = 0.2

    verts = g.exterior.coords
    arc_ct, b_prev = 0, np.pi
    signs = []
    for i, v in enumerate(verts):
        try:
            next = verts[i + 1]
        except IndexError:
            break
        b = bearing(v, next)
        diff = b - b_prev
        sign = diff < 0
        if diff < tol:
            arc_ct += 1
            signs.append(sign)
            single_sign = len(np.unique(signs[-min_arc:])) == 1
            if arc_ct >= min_arc and single_sign:
                return True
        else:
            arc_ct = 0

        b_prev = b

    return False


def pivot_test(in_shp, out_shp):
    pandarallel.initialize(use_memory_fs=False, progress_bar=True)

    l = [os.path.join(in_shp, '{}.shp'.format(s)) for s in ['co', 'nm', 'ut', 'wy']]
    cols = ['IRRIG_TYPE', 'Source_i_1', 'IRR_TYPE', 'Source_i_1']
    for f, col in zip(l, cols):
        if 'nm' not in f:
            continue
        o = os.path.join(out_shp, os.path.basename(f))
        df = gpd.read_file(f)
        df['geo'] = df['geometry'].apply(lambda x: 1 if x is not None else 0)
        df = df[df['geo'] == 1]
        df.drop(columns=['geo'], inplace=True)
        df = df.explode()
        df.index = range(df.shape[0])
        print('{} features'.format(df.shape[0]))
        # df['arc'] = df.geometry.apply(find_arcs)
        df['arc'] = df.geometry.parallel_apply(find_arcs)
        df['popper'] = df.geometry.parallel_apply(popper)
        df.loc[(df['arc'] == 1) & (df['Source_i_1'] == 'Sprinkler'), col] = 'P'
        df.loc[df['popper'] > 0.9, col] = 'P'
        df.to_file(o, crs='epsg:5071')
        print('{} of {} features have an arc'.format(np.count_nonzero(df['arc']), df.shape[0]))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    extract_ = '/media/research/IrrigationGIS/expansion/tables/itype/extracts'
    out_ = '/media/research/IrrigationGIS/expansion/tables/itype'
    get_openet_itype(extract_, out_)

    in_itype = '/media/research/IrrigationGIS/expansion/shapefiles/itype/raw_src'
    out_itype = '/media/research/IrrigationGIS/expansion/shapefiles/itype/int_labels/itype_int.shp'
    # reclassify_itype(in_itype, out_itype)

    out_arc = '/media/research/IrrigationGIS/expansion/shapefiles/itype/inferred_pivot'
    # pivot_test(in_itype, out_arc)
# ========================= EOF ====================================================================
