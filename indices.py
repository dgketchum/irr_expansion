import json
import os

import numpy as np
from climate_indices import compute, indices
from scipy.stats import linregress

from flow.gage_data import hydrograph
from field_points.crop_codes import cdl_key


def mode(a):
    vals, cts = np.unique(a, return_counts=True)
    mode = vals[cts.argmax()]
    return mode


def huc_simi_response(meta, gridded_data, out_meta):
    with open(meta, 'r') as f_obj:
        basins = json.load(f_obj)

    for month_end in range(5, 11):
        dct = {}
        out_json = os.path.join(out_meta, 'huc_{}.json'.format(month_end))
        for huc_id, sdata in basins.items():

            print('\n{} {}: {}'.format(month_end, huc_id, sdata['NAME']))

            _file = os.path.join(gridded_data, '{}.csv'.format(huc_id))
            try:
                df = hydrograph(_file)
            except FileNotFoundError:
                print(_file, 'does not exist, skipping')
                continue

            df['ppt'] = (df['ppt'] - df['ppt'].min()) / (df['ppt'].max() - df['ppt'].min()) + 0.001
            df['etr'] = (df['etr'] - df['etr'].min()) / (df['etr'].max() - df['etr'].min()) + 0.001
            df['kc'] = df['et'] / df['ietr']
            df = df[['irr', 'cc', 'ppt', 'etr', 'kc']]

            met_periods = list(range(1, 13)) + [18, 24, 30, 36]

            for x in met_periods:
                df['SPI_{}'.format(x)] = indices.spi(df['ppt'].values,
                                                     scale=x,
                                                     distribution=indices.Distribution.gamma,
                                                     data_start_year=1984,
                                                     calibration_year_initial=1984,
                                                     calibration_year_final=2021,
                                                     periodicity=compute.Periodicity.monthly)

                df['SPEI_{}'.format(x)] = indices.spei(df['ppt'].values,
                                                       df['etr'].values,
                                                       scale=x,
                                                       distribution=indices.Distribution.pearson,
                                                       data_start_year=1984,
                                                       calibration_year_initial=1984,
                                                       calibration_year_final=2021,
                                                       periodicity=compute.Periodicity.monthly)

            for x in range(1, 8):
                df['SIMI_{}'.format(x)] = indices.spi(df['kc'].values,
                                                      scale=x,
                                                      distribution=indices.Distribution.gamma,
                                                      data_start_year=1984,
                                                      calibration_year_initial=1984,
                                                      calibration_year_final=2021,
                                                      periodicity=compute.Periodicity.monthly)

            combos = [('SPI', 'SIMI'), ('SPEI', 'SIMI')]

            df = df.loc['1990-01-01':]
            pdf = df.loc[[x for x in df.index if x.month == month_end]]
            dct[huc_id] = {'irr_area': np.nanmean(df['irr'].values) / 1e6}

            for met, use in combos:
                rmax = 0.0
                corr_ = [met, use, rmax]
                for met_ts in met_periods:
                    use_periods = range(1, 8)
                    for use_ts in use_periods:
                        msel = '{}_{}'.format(met, met_ts)
                        usel = '{}_{}'.format(use, use_ts)
                        met_ind = pdf[msel].values
                        use_ind = pdf[usel].values

                        nan_check = met_ind + use_ind
                        if np.all(np.isnan(nan_check)):
                            continue
                        elif np.any(np.isnan(nan_check)):
                            isna = np.isnan(nan_check)
                            if np.count_nonzero(~isna) < 20:
                                continue
                            met_ind = met_ind[~isna]
                            use_ind = use_ind[~isna]

                        lr = linregress(met_ind, use_ind)
                        r2 = lr.rvalue ** 2
                        if r2 > rmax:
                            rmax = r2
                            corr_ = [msel, usel, rmax]
                        dct[huc_id]['{}_{}'.format(msel, usel)] = {'b': lr.slope, 'p': lr.pvalue,
                                                                   'r2': r2, 'int': lr.intercept}
                print('{} to {}, r2: {:.3f}'.format(*corr_))

        with open(out_json, 'w') as f:
            json.dump(dct, f, indent=4)


def huc_cdl_response(meta, gridded_data, cdl_area, out_js):

    cdl_k = cdl_key()

    with open(meta, 'r') as f_obj:
        basins = json.load(f_obj)

    with open(cdl_area, 'r') as f_obj:
        crops = json.load(f_obj)

    month_analog = list(range(-6, 6))
    months = list(range(6, 13)) + list(range(1, 6))

    dct = {crop: None for crop in crops.keys()}

    for crop, time_series in crops.items():
        ts, lb, area = [], [], []
        huc_ids = list(time_series.keys())
        for huc_id in huc_ids:

            cdl_ts = time_series[huc_id]
            # print('\n{}: {}'.format(huc_id, basins[huc_id]['NAME']))

            _file = os.path.join(gridded_data, '{}.csv'.format(huc_id))
            try:
                df = hydrograph(_file)
            except FileNotFoundError:
                print(_file, 'does not exist, skipping')
                continue

            df['ppt'] = (df['ppt'] - df['ppt'].min()) / (df['ppt'].max() - df['ppt'].min()) + 0.001
            df['etr'] = (df['etr'] - df['etr'].min()) / (df['etr'].max() - df['etr'].min()) + 0.001
            df = df[['irr', 'cc', 'ppt', 'etr']]
            met_periods = list(range(1, 36))

            for x in met_periods:
                df[x] = indices.spei(precips_mm=df['ppt'].values,
                                     pet_mm=df['etr'].values,
                                     scale=x,
                                     distribution=indices.Distribution.pearson,
                                     data_start_year=1984,
                                     calibration_year_initial=1984,
                                     calibration_year_final=2021,
                                     periodicity=compute.Periodicity.monthly)

            rmax, corr_ = 0.0, [None, 0.0, None]

            for i, month in enumerate(month_analog):

                if month < 1:
                    pdf = df.loc['2007-01-01': '2021-01-01'].copy()
                else:
                    pdf = df.loc['2008-01-01': '2022-01-01'].copy()

                pdf = pdf.loc[[x for x in pdf.index if x.month == months[i]]].copy()

                for met_ts in met_periods:
                    met_ind = pdf[met_ts].values.copy()

                    if np.any(np.isnan(met_ind)):
                        raise NotImplementedError

                    lr = linregress(met_ind, cdl_ts)
                    r = lr.rvalue
                    if abs(r) > abs(rmax) and lr.pvalue < 0.05:
                        rmax = r
                        corr_ = [met_ts, rmax, month]

            if corr_[1] == 0.0:
                continue
            else:
                ts.append(corr_[0])
                lb.append(corr_[2])
                area.append(np.mean(cdl_ts))
                # print('{} from {} r: {:.3f}'.format(corr_[0], corr_[2], corr_[1]))

        print('\n\n\n\n{}'.format(cdl_k[int(crop)][0]))
        area = np.array(area) / sum(area)
        lb = (np.array(lb) * area).sum()
        ts = (np.array(ts) * area).sum()
        dct[crop] = {'lb': months[month_analog.index(int(np.rint(lb)))], 'ts': int(np.rint(ts))}
        print(dct[crop])

    with open(out_js, 'w') as f:
        json.dump(dct, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    data_ = os.path.join(root, 'metadata', 'huc8_metadata.json')
    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'ietr_huc8_21FEB2023')
    out_js = os.path.join(root, 'analysis', 'huc8_sensitivities')
    # huc_simi_response(data_, merged, out_js)

    out_js = os.path.join(root, 'analysis', 'cdl_spei_timescales.json')
    cdl_area_ = os.path.join(root, 'tables/cdl/cdl_huc_area_timeseries.json')
    huc_cdl_response(data_, merged, cdl_area_, out_js)

# ========================= EOF ====================================================================
