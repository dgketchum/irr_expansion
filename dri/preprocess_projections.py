import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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


def preprocess_projections(gridmet, gridmet_gfid, outdir, projections_extracts, processed_projections,
                           target_areas=None):
    fields = pd.read_csv(gridmet_gfid, index_col='OPENET_ID')

    for i, (fid, v) in enumerate(tqdm(fields.iterrows(),
                                      desc=f'Processing Projections Data',
                                      total=fields.shape[0])):

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

        for model in MODEL_LIST:
            for scenario in FUTURE_SCENARIO_LIST:
                if not (model == 'IPSL-CM5A-MR' and scenario == 'rcp45'):
                    continue
                proj_file = os.path.join(projection_raw, f'{scenario}_{model}')

        if first:
            array = np.zeros((len(zone_fids), len(subarray.index), len(REMAP_COLS))) * np.nan
            first = False

        idxes.append(fid)
        array[i, :, :] = subarray.values.reshape((1, len(subarray.index), len(REMAP_COLS)))

    out_json = os.path.join(outdir, os.path.basename(in_pqt).replace('.parquet', '_index.json'))
    with open(out_json, 'w') as f:
        json.dump({'index': idxes}, f, indent=4)

    out_npy = os.path.join(outdir, os.path.basename(in_pqt).replace('.parquet', '.npy'))
    np.save(out_npy, array)
    print(f'saved {out_json}, len {len(idxes)}')
    print(f'saved {out_npy}, shape: {array.shape}')


def split_projections(fields, raw_exports, outdir):
    fields = pd.read_csv(fields)
    gfids = fields['GFID'].unique()

    gfid_data_collector = defaultdict(lambda: defaultdict(list))

    proj_files = []
    for model in MODEL_LIST:
        for scenario in FUTURE_SCENARIO_LIST:

            if model not in ['bcc-csm1-1', 'BNU-ESM'] or scenario != 'rcp45':
                continue

            add_files = []
            for yr in range(2006, 2056):
                file_ = os.path.join(raw_exports, f'{scenario}_{model}_{yr}.csv')
                if not os.path.exists(file_):
                    print(f'{os.path.basename(file_)} does not exist')
                    break
                add_files.append(file_)

            proj_files.extend(add_files)

    for proj_file in tqdm(proj_files, desc="Reading projection files"):
        if not os.path.exists(proj_file):
            continue

        basename = os.path.basename(proj_file).replace('.csv', '')
        parts = basename.split('_')
        scenario = parts[0]
        model = '_'.join(parts[1:-1])

        df = pd.read_csv(proj_file, usecols=['GFID', 'datenum', 'pr'], engine='c')

        df['date'] = pd.to_datetime(df['datenum'], format='%Y%m%d')

        df_monthly = df.groupby('GFID').resample('MS', on='date')[['pr']].sum()

        for gfid, group_df in df_monthly.groupby(level='GFID'):
            gfid_data_collector[gfid][(scenario, model)].append(group_df.droplevel(0))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, gfid in enumerate(tqdm(gfids, desc=f'Processing Projections', total=len(gfids))):

        if gfid not in gfid_data_collector:
            continue

        processed_projections = []
        for (scenario, model), df_list in gfid_data_collector[gfid].items():
            full_ts_df = pd.concat(df_list, axis=0)
            full_ts_df.rename(columns={'pr': f'{scenario}_{model}_ppt'}, inplace=True)
            processed_projections.append(full_ts_df)

        if not processed_projections:
            continue

        final_df = pd.DataFrame().join(processed_projections, how='outer')

        out_file = os.path.join(outdir, f'{gfid}.parquet')
        final_df.to_parquet(out_file)


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    fields_data = os.path.join(nv_data, 'fields_data')

    npy_dir = os.path.join(fields_data, 'fields_npy')
    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gfid_fields = os.path.join(nv_fields_boundaries,
                                    'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    projections_extracts_ = os.path.join(fields_data, 'projections', 'exports')
    projections_processed_ = os.path.join(fields_data, 'projections', 'processed')
    met = os.path.join(fields_data, 'gridmet')
    split_projections(gfid_fields, projections_extracts_, projections_processed_)

    projection_raw = os.path.join(fields_data, 'exports')
    projection_processed = os.path.join(fields_data, 'processed')
    # preprocess_projections(met, gridmet_factors_, projection_raw, projection_processed, target_areas=None)

# ========================= EOF ====================================================================
