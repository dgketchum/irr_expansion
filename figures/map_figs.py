import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

from gage_lists import EXCLUDE_STATIONS


def basin_response(sensitivity_d, in_shape, glob=None, out_shape=None, basins=True, key='STAID'):
    feats = gpd.read_file(in_shape)
    geo = {f[key]: shape(f['geometry']) for i, f in feats.iterrows()}
    areas = {f[key]: f['AREA'] for i, f in feats.iterrows()}
    if basins:
        names = {f[key]: f['STANAME'] for i, f in feats.iterrows()}
    else:
        names = {f[key]: f['Name'] for i, f in feats.iterrows()}

    trends_dct = {}

    l = [os.path.join(sensitivity_d, x) for x in os.listdir(sensitivity_d) if glob in x]

    for f in l:
        m = int(os.path.basename(f).split('.')[0].split('_')[-1])
        with open(f, 'r') as _file:
            dct = json.load(_file)
            dct = {k: v for k, v in dct.items() if k not in EXCLUDE_STATIONS}

        trends_dct.update({m: dct})

    if basins:
        combos = [('SPI', 'SFI'), ('SPEI', 'SFI'), ('SFI', 'SCUI'), ('SFI', 'SIMI')]
    else:
        combos = [('SPI', 'SCUI'), ('SPI', 'SIMI'), ('SPEI', 'SCUI'), ('SPEI', 'SIMI')]

    months = list(range(5, 11))
    sids = trends_dct[5].keys()

    for m in months[-1:]:
        df = pd.DataFrame(index=sids)

        d = trends_dct[m]

        for sid, lr in d.items():
            seq = [(k, v['b'], v['r2']) for k, v in lr.items() if not np.isnan(v['b']) and v['p'] < 0.05]

            for met, use in combos:

                r_type = [(k, b, r) for k, b, r in seq if met in k and use in k]

                if not r_type:
                    continue

                srt = sorted(r_type, key=lambda x: x[2], reverse=True)[0]
                met_ts, use_ts = srt[0].split('_')[1], srt[0].split('_')[3]
                df.loc[sid, '{}_{}_{}_mt'.format(met, use, m)] = int(met_ts)
                df.loc[sid, '{}_{}_{}_ut'.format(met, use, m)] = int(use_ts)
                df.loc[sid, '{}_{}_{}_b'.format(met, use, m)] = srt[1]
                df.loc[sid, '{}_{}_{}_r2'.format(met, use, m)] = srt[2]
                pass

        gdf = gpd.GeoDataFrame(df)
        gdf.geometry = [geo[_id] for _id in gdf.index]
        area_arr = np.array([areas[_id] for _id in gdf.index])
        areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}
        gdf['AREA'] = [areas[_id] for _id in gdf.index]
        if basins:
            gdf['STANAME'] = [names[i] for i in gdf.index]
        _file = os.path.join(out_shape, '{}_{}.gpkg'.format(glob, m))
        gdf = gdf.fillna(0)
        gdf.to_file(_file, driver='GPKG', crs='epsg:5071', geometry=geo)
        print(_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    basins = False
    if basins:
        glob_ = 'basin'
        key_ = 'STAID'
        inshp = os.path.join(root, 'gages', 'selected_gages.shp')
    else:
        glob_ = 'huc8'
        key_ = 'huc8'
        inshp = os.path.join(root, 'shapefiles', 'study_area_huc8.shp')

    oshp_dir = os.path.join('/home/dgketchum/Downloads', 'figures', 'sensitivity_maps')
    js_ = os.path.join(root, 'analysis', 'basin_sensitivities')
    basin_response(js_, inshp, glob=glob_, out_shape=oshp_dir, basins=basins, key=key_)
# ========================= EOF ====================================================================
