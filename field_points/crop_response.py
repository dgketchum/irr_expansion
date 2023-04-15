import os
import warnings

import numpy as np
import pandas as pd

from gridded_data import BASIN_STATES
from transition_modeling.transition_data import KEYS, OLD_KEYS

warnings.filterwarnings("ignore", category=DeprecationWarning)
COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']


def cdl_response(csv, cdl, out_):
    summary_cols = ['crop_code', 'count', 'mean_resp', 'min_resp', 'max_resp', 'std_resp']
    summary = pd.DataFrame(columns=summary_cols)

    bnames = [x for x in os.listdir(csv) if x.strip('.csv')]
    csv = sorted([os.path.join(csv, x) for x in bnames])
    first, ct = True, 0

    for f in csv:
        print(f)
        c = pd.read_csv(f, index_col=0)
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    cols = [c for c in df.columns if c.startswith('met')]
    print(df.shape)
    df[df[cols] == 0] = np.nan
    max_ = df[cols].max(axis=1)
    df = df[cols].idxmax(axis=1)
    df = pd.DataFrame(df, columns=['str'])
    df.dropna(inplace=True)
    df['met'] = df['str'].apply(lambda x: int(x.split('_')[0].strip('met')))
    df['ag'] = df['str'].apply(lambda x: int(x.split('_')[1].strip('ag')))
    df['fr'] = df['str'].apply(lambda x: int(x.split('_')[2].strip('fr')))
    df['max'] = max_

    for s in BASIN_STATES:
        cdl_f = os.path.join(cdl, '{}.csv'.format(s.lower()))
        cdf = pd.read_csv(cdl_f, index_col='OPENET_ID')
        cols = [c for c in cdf.columns]
        cdf = cdf[~cdf.index.duplicated()]
        match = [i for i in cdf.index if i in df.index]
        df.loc[match, cols] = cdf.loc[match, cols]

    df['mode'] = [int(m) for m in df[cols].mode(axis=1)[0]]
    df['count'] = df.apply(lambda x: (x[cols] == x['mode']).sum(), axis=1)
    df.drop(columns=cols, inplace=True)

    for k in OLD_KEYS:
        if k in [41, 49, 53]:
            min_yrs = 4
        else:
            min_yrs = 10
        cdf = df.loc[(df['mode'] == k) & (df['count'] > min_yrs)]
        data = cdf['max'].values
        if data.size == 0:
            continue
        summary.loc[ct] = [k, cdf.shape[0], data.mean(), data.min(), data.max(), data.std()]
        ct += 1
    summary.to_csv(out_)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = '/media/nvm/field_pts/indices/simi'
    meta = '/media/nvm/field_pts/fields_data/fields_shp'
    cdl_data = '/media/research/IrrigationGIS/expansion/tables/cdl/crops'
    summary_out = '/media/research/IrrigationGIS/expansion/analysis/cdl_response_ID.csv'
    cdl_response(in_, cdl_data, summary_out)
# ========================= EOF ====================================================================
