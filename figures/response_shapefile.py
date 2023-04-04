import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


def write_response_shapefile(csv, attrs, shp, out_shp, o_json=None):
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

    if o_json:
        for c in cols:
            ojs = os.path.join(o_json, '{}.json'.format(c))
            dct = {c: list(df[c].values)}
            with open(ojs, 'w') as fp:
                json.dump(dct, fp, indent=4)
                print(c)

    print(df.shape)
    gdf = gpd.read_file(shp)
    gdf.index = gdf['OPENET_ID']
    match = [i for i in gdf.index if i in df.index]
    gdf = gdf.loc[match]
    gdf['resp'] = df.loc[match, 'met4_ag3_fr8']
    gdf.drop(columns=['OPENET_ID'], inplace=True)
    gdf.to_file(out_shp, crs='EPSG:5071')


def write_response_histogram(csv, attrs, out_fig):
    first = True
    bnames = [x for x in os.listdir(csv) if x.strip('.csv')]
    csv = [os.path.join(csv, x) for x in bnames]
    attrs = [os.path.join(attrs, x) for x in os.listdir(attrs) if x in bnames]
    df = None

    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    use_ints = list(range(4, 11))
    use_periods = [datetime(2001, m, 1).strftime('%b') for m in use_ints]

    for m, f in zip(attrs, csv):
        print(f)
        c = pd.read_csv(f, index_col=0)

        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    cols = [c for c in df.columns if c.startswith('met')]
    df = df[cols].idxmax(axis=1)
    df = pd.DataFrame(df, columns=['str'])
    df['met'] = df['str'].apply(lambda x: int(x.split('_')[0].strip('met')))
    df['ag'] = df['str'].apply(lambda x: int(x.split('_')[1].strip('ag')))
    df['fr'] = df['str'].apply(lambda x: int(x.split('_')[2].strip('fr')))

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), dpi=80)

    sns.countplot(data=df, ax=ax[0], x='met', palette=['blue'])
    ax[0].set(xlabel='Standardised Precipitation-Evapotranspiration Index, Time Scale (Months)')
    ax[0].set_ylabel(None)

    sns.countplot(data=df, ax=ax[1], x='ag', palette=['red'])
    ax[1].set(xlabel='Standardized Irrigation Management Index Time Scale (Months)')
    ax[1].set_ylabel('Fields Count')

    sns.countplot(data=df, ax=ax[2], x='fr', palette=['green'])
    ax[2].set(xlabel='From Month')
    ax[2].set_ylabel(None)
    ax[2].set_xticklabels(use_periods)

    plt.tight_layout()
    # plt.show()
    plt.savefig(out_fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = os.path.join(root, 'field_pts/indices/simi')
    meta = os.path.join(root, 'field_pts/usbr_attr')

    out_ = os.path.join(root, 'figures', 'partitions', 'fields_response_hist.png')
    s = os.path.join(root, 'shapefiles', 'openet_field_centr_irr_gt19',
                     'state_openet_points/field_null_attr_13FEB2023.shp')

    oshp = os.path.join(root, 'shapefiles/field_pts/response.shp')
    ojs_ = os.path.join(root, 'field_pts/indices/response_json')

    # write_response_shapefile(in_, meta, s, oshp, None)
    write_response_histogram(in_, meta, out_)
# ========================= EOF ====================================================================
