import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

REMAP_COLS = {'ETa_final_acre_ft': 'et', 'NetET_final_acre_ft': 'cc', 'PPT_MM': 'ppt',
              'ETO_MM': 'eto', 'Eff_PPT_Adjusted_acre_ft': 'eff_ppt'}

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

FUTURE_SCENARIO_LIST = ['rcp45', 'rcp85']

MODEL_LIST = ['bcc-csm1-1',
              'bcc-csm1-1-m',
              'BNU-ESM',
              'CanESM2',
              'CCSM4',
              'CNRM-CM5',
              'CSIRO-Mk3-6-0',
              'GFDL-ESM2G',
              'GFDL-ESM2M',
              'HadGEM2-CC365',
              'HadGEM2-ES365',
              'inmcm4',
              'IPSL-CM5A-MR',
              'IPSL-CM5A-LR',
              'IPSL-CM5B-LR',
              'MIROC5',
              'MIROC-ESM',
              'MIROC-ESM-CHEM',
              'MRI-CGCM3',
              'NorESM1-M']

GRIDMET_RESAMPLE_MAP = {'year': 'first',
                        'month': 'first',
                        'day': 'first',
                        'centroid_lat': 'first',
                        'centroid_lon': 'first',
                        'elev_m': 'first',
                        'eto_mm': 'sum',
                        'prcp_mm': 'sum',
                        'eto_mm_uncorr': 'sum'}


def preproc_csv_to_npy(in_pqt, gridmet, gridmet_gfid, outdir):
    fields = pd.read_csv(gridmet_gfid, index_col='OPENET_ID')

    df = pd.read_parquet(in_pqt)

    fids = df['OPENET_ID'].unique()
    fields = fields.loc[[i for i in fields.index if i in fids]].copy()

    first, idxes, array = True, [], None

    for i, (fid, v) in enumerate(tqdm(fields.iterrows(), desc='Processing Field Data Arrays', total=df.shape[0])):

        g_fid = str(int(v['GFID']))

        file_ = os.path.join(gridmet, 'gridmet_{}.csv'.format(g_fid))

        met_df = pd.read_csv(file_, index_col='date', parse_dates=True)
        met_new_cols = {c: f'{c}_gm' for c in met_df.columns}
        met_df = met_df.resample('MS').agg(GRIDMET_RESAMPLE_MAP)
        met_df = met_df.rename(columns=met_new_cols)

        subarray = df[df['OPENET_ID'] == fid].copy()

        subarray = subarray.rename(columns=REMAP_COLS)[COLS]
        subarray = subarray.reindex(met_df.index)

        subarray['eto'] = met_df['eto_mm_gm'].copy()
        subarray['ppt'] = met_df['prcp_mm_gm'].copy()

        if first:
            array = np.zeros((len(fids), len(met_df.index), len(REMAP_COLS))) * np.nan
            first = False

        idxes.append(fid)
        array[i, :, :] = subarray.values.reshape((1, len(met_df.index), len(REMAP_COLS)))
        nan_ct = [(c, np.count_nonzero(np.isnan(subarray[c]))) for c in COLS]
        print(f'{fid} nan values: {nan_ct}')

    out_json = os.path.join(outdir, os.path.basename(in_pqt).replace('.parquet', '_index.json'))
    with open(out_json, 'w') as f:
        json.dump({'index': idxes}, f, indent=4)

    out_npy = os.path.join(outdir, os.path.basename(in_pqt).replace('.parquet', '.npy'))
    np.save(out_npy, array)
    print(f'saved {out_json}, len {idxes}')
    print(f'saved {out_npy}, shape: {array.shape}')


def split_projections(dir_, split_out):
    for scenario in FUTURE_SCENARIO_LIST:
        for model in MODEL_LIST:
            c = os.path.join(dir_, f'{scenario}_{model}.csv')
            df = pd.read_csv(c, skiprows=10000)
            a = 1


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    fields_data = os.path.join(nv_data, 'fields_data')
    pqt = os.path.join(fields_data, 'field_summaries_EToF_final.parquet')
    outdir_ = os.path.join(fields_data, 'fields_npy')

    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gridmet_factors_ = os.path.join(nv_fields_boundaries,
                                    'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    met = os.path.join(fields_data, 'gridmet')

    preproc_csv_to_npy(pqt, met, gridmet_factors_, outdir_)

    csv_dir = os.path.join(root, 'Nevada/projections/exports')
    splits = os.path.join(root, 'Nevada/projections/splits')
    # split_projections(csv_dir, splits)
# ========================= EOF ====================================================================
