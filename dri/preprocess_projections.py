import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
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


def _read_and_process_proj_file(proj_file):
    if not os.path.exists(proj_file):
        return None

    basename = os.path.basename(proj_file).replace('.csv', '')
    parts = basename.split('_')
    scenario = parts[0]
    model = '_'.join(parts[1:-1])

    file_data_collector = defaultdict(lambda: defaultdict(list))

    try:
        df = pd.read_csv(proj_file, usecols=['GFID', 'datenum', 'pr'], engine='c')
        df['date'] = pd.to_datetime(df['datenum'], format='%Y%m%d')
        df_monthly = df.groupby('GFID').resample('MS', on='date')[['pr']].sum()

        for gfid, group_df in df_monthly.groupby(level='GFID'):
            file_data_collector[gfid][(scenario, model)].append(group_df.droplevel(0))

        return dict(file_data_collector)
    except Exception as e:
        print(f"Error processing file {proj_file}: {e}")
        return None


def _process_and_write_gfid(args):
    gfid, gfid_data, outdir = args

    processed_projections = []
    for (scenario, model), df_list in gfid_data.items():
        full_ts_df = pd.concat(df_list, axis=0)
        full_ts_df.rename(columns={'pr': f'{scenario}_{model}_ppt'}, inplace=True)
        processed_projections.append(full_ts_df)

    if not processed_projections:
        return

    final_df = pd.DataFrame(index=pd.to_datetime([]))
    for df in processed_projections:
        final_df = final_df.join(df, how='outer')

    out_file = os.path.join(outdir, f'{gfid}.parquet')
    final_df.to_parquet(out_file)


def split_projections(fields, raw_exports, outdir, num_workers=None):
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

    if num_workers == 1:
        results = []
        for proj_file in proj_files:
            result = _read_and_process_proj_file(proj_file)
            results.append(result)

    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_read_and_process_proj_file, proj_files),
                                total=len(proj_files),
                                desc="Reading projection files"))

    for file_result in results:
        if file_result is None:
            continue
        for gfid, data in file_result.items():
            for (scenario, model), df_list in data.items():
                gfid_data_collector[gfid][(scenario, model)].extend(df_list)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    tasks = []
    for gfid in gfids:
        if gfid in gfid_data_collector:
            tasks.append((gfid, gfid_data_collector[gfid], outdir))

    if num_workers == 1:
        for task in tasks:
            _process_and_write_gfid(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(_process_and_write_gfid, tasks),
                      total=len(tasks),
                      desc='Processing and writing projections'))


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
    split_projections(gfid_fields, projections_extracts_, projections_processed_, num_workers=6)

    projection_raw = os.path.join(fields_data, 'exports')
    projection_processed = os.path.join(fields_data, 'processed')
    # preprocess_projections(met, gridmet_factors_, projection_raw, projection_processed, target_areas=None)

# ========================= EOF ====================================================================
