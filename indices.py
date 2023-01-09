import os
import json

from scipy.stats import linregress

from climate_indices import compute, indices, utils

from gage_data import hydrograph
from plot_indices import plot_indices_panel


def write_indices(meta, gridded_data, out_meta, plot_dir=None, month_end=10):
    with open(meta, 'r') as f_obj:
        stations = json.load(f_obj)

    dct = {}
    for sid, sdata in stations.items():

        print('\n{}: {}'.format(sid, sdata['STANAME']))

        _file = os.path.join(gridded_data, '{}.csv'.format(sid))
        try:
            df = hydrograph(_file)
        except FileNotFoundError:
            print(_file, 'does not exist, skipping')
            continue

        df['ppt'] = (df['ppt'] - df['ppt'].min()) / (df['ppt'].max() - df['ppt'].min()) + 0.001
        df['etr'] = (df['etr'] - df['etr'].min()) / (df['etr'].max() - df['etr'].min()) + 0.001
        df['kc'] = df['et'] / df['ietr']
        df = df[['q', 'cc', 'ppt', 'etr', 'kc']]

        met_periods = [1, 6, 12, 18, 24, 30, 36]

        for x in met_periods:
            df['SPI_{}'.format(x)] = indices.spi(df['ppt'].values,
                                                 scale=x,
                                                 distribution=indices.Distribution.gamma,
                                                 data_start_year=1987,
                                                 calibration_year_initial=1987,
                                                 calibration_year_final=2021,
                                                 periodicity=compute.Periodicity.monthly)

            df['SPEI_{}'.format(x)] = indices.spei(df['ppt'].values,
                                                   df['etr'].values,
                                                   scale=x,
                                                   distribution=indices.Distribution.pearson,
                                                   data_start_year=1987,
                                                   calibration_year_initial=1987,
                                                   calibration_year_final=2021,
                                                   periodicity=compute.Periodicity.monthly)

            df['SSFI_{}'.format(x)] = indices.spi(df['q'].values,
                                                  scale=x,
                                                  distribution=indices.Distribution.gamma,
                                                  data_start_year=1987,
                                                  calibration_year_initial=1987,
                                                  calibration_year_final=2021,
                                                  periodicity=compute.Periodicity.monthly)

        for x in range(1, 8):
            df['SCUI_{}'.format(x)] = indices.spi(df['cc'].values,
                                                  scale=x,
                                                  distribution=indices.Distribution.gamma,
                                                  data_start_year=1987,
                                                  calibration_year_initial=1987,
                                                  calibration_year_final=2021,
                                                  periodicity=compute.Periodicity.monthly)

            df['SIMI_{}'.format(x)] = indices.spi(df['kc'].values,
                                                  scale=x,
                                                  distribution=indices.Distribution.gamma,
                                                  data_start_year=1987,
                                                  calibration_year_initial=1987,
                                                  calibration_year_final=2021,
                                                  periodicity=compute.Periodicity.monthly)

        if plot_dir:
            for met_param in ['SPI_12', 'SPEI_12']:
                use_timescale = 2
                fig_file = os.path.join(plot_dir, '{}_{}.png'.format(sid, met_param))
                plot_indices_panel(df, met_param, use_timescale, month_end, fig_file, title_str=sdata['STANAME'])

        dct[sid] = {}
        combos = [('SPI', 'SCUI'), ('SPI', 'SIMI'), ('SPEI', 'SCUI'), ('SPEI', 'SIMI')]
        df = df.loc['1990-01-01':]
        pdf = df.loc[[x for x in df.index if x.month == month_end]]
        for met, use in combos:
            rmax = 0.0
            corr_ = [met, use, rmax]
            for met_ts in met_periods:
                for use_ts in range(1, 8):
                    msel = '{}_{}'.format(met, met_ts)
                    usel = '{}_{}'.format(use, use_ts)
                    met_ind = pdf[msel].values
                    use_ind = pdf[usel].values
                    lr = linregress(met_ind, use_ind)
                    r2 = lr.rvalue ** 2
                    if r2 > rmax:
                        rmax = r2
                        corr_ = [msel, usel, rmax]
                    dct[sid]['{}_{}'.format(msel, usel)] = {'b': lr.slope, 'p': lr.pvalue,
                                                            'r2': r2, 'int': lr.intercept}
            print('{} to {}, r2: {:.3f}'.format(*corr_))

    with open(out_meta, 'w') as f:
        json.dump(dct, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'extracts_ietr_nater_8JAN2022')
    data_ = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
    figs = os.path.join(root, 'figures', 'panel')
    out_js = os.path.join(root, 'analysis', 'basin_sensitivities', 'irrigated_indices.json')
    write_indices(data_, merged, out_js, plot_dir=None, month_end=10)

# ========================= EOF ====================================================================
