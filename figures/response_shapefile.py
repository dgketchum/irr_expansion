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

    cols = [c for c in df.columns if c.startswith('met')]
    max_ = df[cols].max(axis=1)
    df = df[cols].idxmax(axis=1)
    df = pd.DataFrame(df, columns=['str'])
    df['met'] = df['str'].apply(lambda x: int(x.split('_')[0].strip('met')))
    df['ag'] = df['str'].apply(lambda x: int(x.split('_')[1].strip('ag')))
    df['fr'] = df['str'].apply(lambda x: int(x.split('_')[2].strip('fr')))
    df['max'] = max_

    gdf['met'] = df.loc[match, 'met']
    gdf['ag'] = df.loc[match, 'ag']
    gdf['fr'] = df.loc[match, 'fr']
    gdf['max'] = df.loc[match, 'max']
    gdf.drop(columns=['OPENET_ID'], inplace=True)
    print(gdf.shape)
    gdf.to_file(out_shp, crs='EPSG:5071')


def write_response_histogram(csv, areas, out_fig):
    first = True
    bnames = [x for x in os.listdir(csv) if x.strip('.csv')]
    csv = [os.path.join(csv, x) for x in bnames]

    with open(areas, 'r') as f_obj:
        areas = json.load(f_obj)
    areas = pd.Series(areas)

    df = None

    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    use_ints = list(range(4, 11))
    use_periods = [datetime(2001, m, 1).strftime('%b') for m in use_ints]

    for f in csv:

        print(f)
        c = pd.read_csv(f, index_col=0)
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    cols = [c for c in df.columns if c.startswith('met')]
    resp = df[cols].max(axis=1)
    df = df[cols].idxmax(axis=1)
    df = pd.DataFrame(df, columns=['str'])
    df['met'] = df['str'].apply(lambda x: int(x.split('_')[0].strip('met')))
    df['ag'] = df['str'].apply(lambda x: int(x.split('_')[1].strip('ag')))
    df['fr'] = df['str'].apply(lambda x: int(x.split('_')[2].strip('fr')))
    match = [k for k, v in areas.items() if k in df.index]
    df.loc[match, 'Area'] = areas.loc[match]
    df['Response'] = resp

    palette = sns.color_palette("rocket", as_cmap=True)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), dpi=80)

    sns.countplot(data=df, ax=ax[0], x='met', color=palette.colors[1])
    ax[0].set(xlabel='Standardised Precipitation-Evapotranspiration Index, Time Scale (Months)')
    ax[0].set_ylabel(None)

    sns.countplot(data=df, ax=ax[1], x='ag', color=palette.colors[86])
    ax[1].set(xlabel='Standardized Irrigation Management Index Time Scale (Months)')
    ax[1].set_ylabel('Fields Count')

    sns.countplot(data=df, ax=ax[2], x='fr', color=palette.colors[172])
    ax[2].set(xlabel='From Month')
    ax[2].set_ylabel(None)
    ax[2].set_xticklabels(use_periods)

    plt.tight_layout()
    # plt.show()
    plt.savefig(out_fig, dpi=5 * plt.gcf().dpi)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = os.path.join(root, 'field_pts/indices/simi')
    fields_area = os.path.join(root, 'tables/cdl/fields_area.json')
    meta = os.path.join(root, 'field_pts/usbr_attr')
    s = os.path.join(root, 'shapefiles', 'openet_field_centr_irr_gt19',
                     'state_openet_points/field_null_attr_13FEB2023.shp')

    oshp = os.path.join(root, 'shapefiles/field_pts/response.shp')
    ojs_ = os.path.join(root, 'field_pts/indices/response_json')

    # write_response_shapefile(in_, meta, s, oshp, None)

    out_ = os.path.join(root, 'figures', 'field_pts', 'fields_response_hist_5DEC2023.png')
    write_response_histogram(in_, fields_area, out_)

    park = '/media/research/IrrigationGIS/expansion/figures/park_fields'
    indices_ = os.path.join(park, 'indices', 'simi')
    csv_ = os.path.join(park, 'fields_shp')
    ishp_ = os.path.join(park, 'fields_shp', 'park_select.shp')
    oshp_ = os.path.join(park, 'fields_shp', 'park_response.shp')
    # write_response_shapefile(indices_, csv_, ishp_, oshp_, o_json=None)
# ========================= EOF ====================================================================
