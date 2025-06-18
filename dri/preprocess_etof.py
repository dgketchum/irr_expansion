import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

REMAP_COLS = {'ETa_final_acre_ft': 'et', 'NetET_final_acre_ft': 'cc', 'PPT_MM': 'ppt',
              'ETO_MM': 'eto', 'Eff_PPT_Adjusted_acre_ft': 'eff_ppt'}

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']
GRIDMET_RESAMPLE_MAP = {'year': 'first',
                        'month': 'first',
                        'day': 'first',
                        'centroid_lat': 'first',
                        'centroid_lon': 'first',
                        'elev_m': 'first',
                        'eto_mm': 'sum',
                        'prcp_mm': 'sum',
                        'eto_mm_uncorr': 'sum'}

def preprocess_historical(in_pqt, gridmet, gridmet_gfid, outdir, target_areas=None):
    fields = pd.read_csv(gridmet_gfid, index_col='OPENET_ID')

    hyd_areas = [(f.split('.')[0], os.path.join(in_pqt, f)) for f in os.listdir(in_pqt) if f.endswith('parquet')]

    for hydro_area, etof_file in hyd_areas:

        if target_areas and hydro_area not in target_areas:
            continue

        df = pd.read_parquet(etof_file)
        df.index = pd.DatetimeIndex(df['DATE'])
        zone_fids = list(set(df['OPENET_ID'].values))
        zone_fields = fields.loc[[i for i in fields.index if i in zone_fids]]

        first, idxes, array = True, [], None

        for i, (fid, v) in enumerate(tqdm(zone_fields.iterrows(),
                                          desc=f'Processing {hydro_area}',
                                          total=zone_fields.shape[0])):

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


def split_etof_input(pqt, split_out, hyd_areas_file):
    df = pd.read_parquet(pqt)

    hyd_areas = df['HYD_AREA'].unique()
    zones = []

    for hyd in hyd_areas:
        a = df[df['HYD_AREA'] == hyd].copy()
        out_file = os.path.join(split_out, f'{hyd}.parquet')
        a.to_parquet(out_file)
        zones.append(hyd)
        print(out_file)

    with open(hyd_areas_file, 'w') as f:
        json.dump({'tiles': zones}, f, indent=4)

    print(hyd_areas_file)



if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    fields_data = os.path.join(nv_data, 'fields_data')
    pqt_ = os.path.join(fields_data, 'NV_field_summaries_EToF_final_large.parquet')
    js_ = os.path.join(fields_data, 'NV_field_summaries_EToF_tiles.json')
    pqt_dir = os.path.join(fields_data, 'fields_pqt')
    split_etof_input(pqt_, pqt_dir, js_)

    npy_dir = os.path.join(fields_data, 'fields_npy')
    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gridmet_factors_ = os.path.join(nv_fields_boundaries,
                                    'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    met = os.path.join(fields_data, 'gridmet')

    preprocess_historical(pqt_dir, met, gridmet_factors_, npy_dir, target_areas=None)

# ========================= EOF ====================================================================
