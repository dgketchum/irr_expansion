import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

FUTURE_SCENARIO_LIST = ['rcp45', 'rcp85']

MODEL_LIST = ['bcc-csm1-1', 'bcc-csm1-1-m', 'BNU-ESM', 'CanESM2', 'CCSM4',
              'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M',
              'HadGEM2-CC365', 'HadGEM2-ES365', 'inmcm4', 'IPSL-CM5A-MR',
              'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM',
              'MIROC-ESM-CHEM', 'MRI-CGCM3', 'NorESM1-M']


def unstandardize_simi(future_simi_series, gamma_params):
    shape, loc, scale = gamma_params
    probabilities = stats.norm.cdf(future_simi_series)
    projected_net_et = stats.gamma.ppf(probabilities, a=shape, loc=loc, scale=scale)
    return projected_net_et


def project_net_et(historical_desc, correlations_csv_path, historical_npy_dir, calculation,
                   future_parquet_dir, out_dir, gfid_csv, from_month=12, ag_scale=12):
    correlations_df = pd.read_csv(correlations_csv_path, index_col=0)

    fields_gridmap = pd.read_csv(gfid_csv, index_col='OPENET_ID')
    fields_gridmap = {i: r['GFID'] for i, r in fields_gridmap.iterrows()}

    corr_cols = [c for c in correlations_df.columns if '_corr' in c]
    s_best_corr_col = correlations_df[corr_cols].abs().idxmax(axis=1)

    best_models = {}
    for field_id, best_corr_col in s_best_corr_col.items():
        parts = best_corr_col.split('_')
        met_p = int(parts[0].replace('met', ''))

        slope_col = best_corr_col.replace('_corr', '_slope')
        intercept_col = best_corr_col.replace('_corr', '_intercept')

        best_models[field_id] = {
            'met_p': met_p,
            'slope': correlations_df.loc[field_id, slope_col],
            'intercept': correlations_df.loc[field_id, intercept_col]
        }

    historical_npy_path = os.path.join(historical_npy_dir, f'{historical_desc}.npy')
    historical_data = np.load(historical_npy_path)

    with open(historical_npy_path.replace('.npy', '_index.json'), 'r') as fp:
        field_index = json.load(fp)['index']

    hist_dt_range = pd.to_datetime([f'{y}-{m}-01' for y in range(1980, 2025) for m in range(1, 13)])
    projection_dt_range = pd.to_datetime([f'{y}-{m}-01' for y in range(2006, 2056) for m in range(1, 13)])
    projection_mask = [True if dt.year > 2025 else False for dt in projection_dt_range]
    future_dt_range = [dt for i, dt in zip(projection_mask, projection_dt_range) if i]

    model, scenario = None, None

    for i, field_id in enumerate(field_index):

        print(f'field {field_id}: {i} of {len(field_index)}')

        if field_id not in best_models:
            continue

        field_gfid = fields_gridmap[field_id]
        field_parquet_path = os.path.join(future_parquet_dir, f'{field_gfid}.parquet')
        if not os.path.exists(field_parquet_path):
            continue

        future_field_df = pd.read_parquet(field_parquet_path)

        field_projections = []

        for model in MODEL_LIST:
            for scenario in FUTURE_SCENARIO_LIST:

                if model not in ['bcc-csm1-1', 'BNU-ESM'] or scenario != 'rcp45':
                    continue

                future_col_name = f'{scenario}_{model}'

                model = best_models[field_id]
                met_p = model['met_p']

                if calculation == 'simi':
                    et = historical_data[i, :, COLS.index('et')]
                    eto = historical_data[i, :, COLS.index('eto')]
                    et_metric = et / eto
                else:
                    et_metric = historical_data[i, :, COLS.index('cc')]

                s_net_et = pd.Series(et_metric, index=hist_dt_range)
                s_net_et_agg = s_net_et.rolling(window=ag_scale, min_periods=ag_scale).mean()
                historical_et_for_month = s_net_et_agg[s_net_et_agg.index.month == from_month].dropna()

                gamma_params = stats.gamma.fit(historical_et_for_month)

                historical_ppt = historical_data[i, :, COLS.index('ppt')]
                future_ppt = future_field_df[f'{future_col_name}_ppt'].values[projection_mask]

                full_ppt = np.concatenate([historical_ppt, future_ppt])
                full_dt_range = hist_dt_range.union(future_dt_range)

                spi = indices.spi(full_ppt, scale=met_p, distribution=indices.Distribution.gamma,
                                  data_start_year=hist_dt_range.year[0],
                                  calibration_year_initial=hist_dt_range.year[0],
                                  calibration_year_final=hist_dt_range.year[-1],
                                  periodicity=compute.Periodicity.monthly)

                s_spi = pd.Series(spi, index=full_dt_range)
                future_spi_for_month = s_spi[s_spi.index.month == from_month].loc[future_dt_range[0]:]

                future_simi = model['slope'] * future_spi_for_month + model['intercept']

                projected_net_et = unstandardize_simi(future_simi.values, gamma_params)
                et_proj = pd.Series(projected_net_et, index=future_spi_for_month.index)
                field_projections.append(et_proj)

        projection_df = pd.concat(field_projections, ignore_index=False)
        projection_df.index = projection_df.index.year

        output_filename = os.path.join(out_dir, f'projected_et_{field_id}_{model}_{scenario}.csv')
        projection_df.to_csv(output_filename)
        print(f"Successfully saved projections to {output_filename}")


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir = os.path.join(root, 'Nevada', 'dri_field_pts')
    historical_npy_dir = os.path.join(nv_data_dir, 'fields_data', 'fields_npy')
    results_dir = os.path.join(nv_data_dir, 'fields_data', 'indices')

    calculation_type = 'cc'
    correlations_csv = os.path.join(results_dir, calculation_type, 'field_summaries_EToF_final_wLRcoeffs_annualAg.csv')

    future_data_dir = os.path.join(nv_data_dir, 'fields_data', 'projections', 'processed')

    fields_gis = os.path.join(nv_data_dir, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gfid_fields = os.path.join(nv_fields_boundaries,
                               'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    projection_out_dir = os.path.join(results_dir, 'projections')
    os.makedirs(projection_out_dir, exist_ok=True)

    project_net_et(
        historical_desc='field_summaries_EToF_final',
        correlations_csv_path=correlations_csv,
        historical_npy_dir=historical_npy_dir,
        future_parquet_dir=future_data_dir,
        out_dir=projection_out_dir,
        calculation=calculation_type,
        gfid_csv=gfid_fields)

# ========================= EOF ====================================================================
