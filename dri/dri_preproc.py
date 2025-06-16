import os
import json
import pandas as pd
import numpy as np

REMAP_COLS = {'ETa_final_acre_ft': 'et', 'NetET_final_acre_ft': 'cc', 'PPT_MM': 'ppt',
              'ETO_MM': 'eto', 'Eff_PPT_Adjusted_acre_ft': 'eff_ppt'}

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']


def preproc_csv_to_npy(in_csv, outdir):
    df = pd.read_csv(in_csv, index_col='DATE', parse_dates=True)

    fids = df['OPENET_ID'].unique()

    dates = pd.DatetimeIndex(list(set(df.index)))
    dates = dates.sort_values()

    first, idxes, array = True, [], None

    for i, fid in enumerate(fids):

        subarray = df[df['OPENET_ID'] == fid].copy()

        subarray = subarray.rename(columns=REMAP_COLS)[COLS]
        subarray = subarray.reindex(dates)

        if first:
            array = np.zeros((len(fids), len(dates), len(REMAP_COLS))) * np.nan
            idxes.append(fid)
            first = False

        array[i, :, :] = subarray.values.reshape((1, len(dates), len(REMAP_COLS)))

    out_json = os.path.join(outdir, os.path.basename(csv).replace('.csv', '_index.json'))
    with open(out_json, 'w') as f:
        json.dump({'index': idxes}, f, indent=4)

    out_npy = os.path.join(outdir, os.path.basename(csv).replace('.csv', '.npy'))
    np.save(out_npy, array)
    print(f'saved {out_json}')
    print(f'saved {out_npy}')


if __name__ == '__main__':
    csv = '/media/nvm/dri_field_pts/fields_data/field_summaries_EToF_final.csv'
    outdir_ = '/media/nvm/dri_field_pts/fields_data/fields_npy'
    preproc_csv_to_npy(csv, outdir_)
# ========================= EOF ====================================================================
