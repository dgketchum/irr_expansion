import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd


def write_response(csv, attrs, shp, out_shp, o_json, out_fig):
    first = True
    bnames = [x for x in os.listdir(csv) if x.strip('.csv')]
    csv = [os.path.join(csv, x) for x in bnames]
    attrs = [os.path.join(attrs, x) for x in os.listdir(attrs) if x in bnames]

    for m, f in zip(attrs, csv):
        print(f)
        c = pd.read_csv(f, index_col=0)
        meta = pd.read_csv(m, index_col='OPENET_ID')
        match = [i for i in c.index if i in meta.index]

        if match:
            c.loc[match, 'usbrid'] = meta['usbrid']
        else:
            c['usbrid'] = [0 for _ in range(c.shape[0])]

        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    cols = [c for c in df.columns if c.startswith('met')]
    print(df.shape)
    df[df[cols] == 0] = np.nan
    df.dropna(subset=cols, inplace=True)
    ocsv = out_shp.replace('.shp', '.csv')
    df.to_csv(ocsv)

    for c in cols:
        ojs = os.path.join(o_json, '{}.json'.format(c))
        dct = {c: list(df[c].values)}
        with open(ojs, 'w') as fp:
            json.dump(dct, fp, indent=4)
            print(c)

    print(df.shape)
    gdf = gpd.read_file(shp)
    gdf.index = gdf['OPENET_ID']
    gdf = gdf.loc[df.index]
    gdf['resp'] = df['met11_ag5_fr10'].values
    gdf.to_file(out_shp, crs='EPSG:5071')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = '/media/nvm/field_pts/indices/simi'
    meta = '/media/nvm/field_pts/usbr_attr'
    out_ = os.path.join(root, 'figures', 'partitions', 'fields.png')
    s = os.path.join(root, 'shapefiles', 'openet_field_centr_irr_gt19',
                     'state_openet_points/field_null_attr_13FEB2023.shp')
    oshp = '/media/nvm/field_pts/indices/shapefiles/response.shp'
    ojs_ = '/media/nvm/field_pts/indices/response_json'
    write_response(in_, meta, s, oshp, ojs_, out_)
# ========================= EOF ====================================================================
