import os
from pprint import pprint

import numpy as np
import pandas as pd

from flow.gage_data import hydrograph


def merge_gridded_flow_data(gridded_dir, out_dir, flow_dir=None, start_year=1987, end_year=2021, glob='glob',
                            join_key='STAID'):
    missing, missing_ct, processed_ct = [], 0, 0

    l = [os.path.join(gridded_dir, x) for x in os.listdir(gridded_dir) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])

        try:
            if first:
                df = pd.read_csv(csv, index_col=join_key)
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = pd.read_csv(csv, index_col=join_key)
                if y < start_year:
                    c['irr'] = [np.nan for _ in range(c.shape[0])]
                    c['et'] = [np.nan for _ in range(c.shape[0])]
                    c['ept'] = [np.nan for _ in range(c.shape[0])]
                    c['ietr'] = [np.nan for _ in range(c.shape[0])]
                    c['cc'] = [np.nan for _ in range(c.shape[0])]
                cols = list(c.columns)
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
                df = pd.concat([df, c], axis=1)

        except pd.errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df = df.copy()
    df['{}_STR'.format(join_key)] = [str(x).rjust(8, '0') for x in list(df.index.values)]

    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(start_year), '{}-12-31'.format(end_year)
    idx = pd.DatetimeIndex(pd.date_range(s, e, freq='M'))

    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]

    for d in dfd:
        try:
            sta = d['{}_STR'.format(join_key)]

            irr, cc, et, ietr, ept = [], [], [], [], []
            for y, m in months:
                try:
                    cc_, et_ = d['cc_{}_{}'.format(y, m)], d['et_{}_{}'.format(y, m)]
                    ietr_, ept_ = d['ietr_{}_{}'.format(y, m)], d['eff_ppt_{}_{}'.format(y, m)]
                    irr_ = d['irr_{}_{}'.format(y, m)]
                    cc.append(cc_)
                    et.append(et_)
                    ietr.append(ietr_)
                    ept.append(ept_)
                    irr.append(irr_)
                except KeyError:
                    cc.append(np.nan)
                    et.append(np.nan)
                    ietr.append(np.nan)
                    ept.append(np.nan)
                    irr.append(np.nan)

            irr = irr, 'irr'
            cc = cc, 'cc'
            et = et, 'et'
            ept = ept, 'ept'
            ietr = ietr, 'ietr'

            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue

            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'

            recs = pd.DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, ietr, ept]]), index=idx)

            if flow_dir:
                q_file = os.path.join(flow_dir, '{}.csv'.format(sta))
                qdf = hydrograph(q_file)
                h = pd.concat([qdf, recs], axis=1)
            else:
                h = recs

            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            processed_ct += 1

            print(file_name)

        except FileNotFoundError:
            missing_ct += 1
            print(sta, 'not found')
            missing.append(sta)

    print(processed_ct, 'processed')
    print(missing_ct, 'missing')
    pprint(missing)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    bname = 'ietr_basins_21FEB2023'

    basin_extracts = os.path.join(root, 'tables', 'gridded_tables', bname)
    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', bname)
    hydrographs_ = os.path.join(root, 'tables', 'hydrographs', 'monthly_q')

    merge_gridded_flow_data(basin_extracts, merged, flow_dir=hydrographs_,
                            glob=bname, join_key='STAID')

# ========================= EOF ====================================================================
