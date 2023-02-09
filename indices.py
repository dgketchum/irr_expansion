import os
import json

from scipy.stats import linregress

from climate_indices import compute, indices

from sfi.gage_data import hydrograph
from figures.plot_indices import plot_indices_panel, plot_indices_panel_nonstream


def write_indices(meta, gridded_data, out_meta, plot_dir=None, basins=False, desc_str='basin'):
    with open(meta, 'r') as f_obj:
        stations = json.load(f_obj)

    for month_end in range(5, 11):
        dct = {}
        out_json = os.path.join(out_meta, '{}_{}.json'.format(desc_str, month_end))
        for sid, sdata in stations.items():
            if sid != '00000230':
                continue

            if basins:
                print('\n{} {}: {}'.format(month_end, sid, sdata['STANAME']))
            else:
                print('\n{} {}: {}'.format(month_end, sid, sdata['NAME']))

            _file = os.path.join(gridded_data, '{}.csv'.format(sid))
            try:
                df = hydrograph(_file)
            except FileNotFoundError:
                print(_file, 'does not exist, skipping')
                continue

            df['ppt'] = (df['ppt'] - df['ppt'].min()) / (df['ppt'].max() - df['ppt'].min()) + 0.001
            df['etr'] = (df['etr'] - df['etr'].min()) / (df['etr'].max() - df['etr'].min()) + 0.001
            df['kc'] = df['et'] / df['ietr']

            if basins:
                df = df[['q', 'cc', 'ppt', 'etr', 'kc']]
            else:
                df = df[['cc', 'ppt', 'etr', 'kc']]

            met_periods = list(range(1, 13)) + [18, 24, 30, 36]

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
                if basins:
                    df['SFI_{}'.format(x)] = indices.spi(df['q'].values,
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
                ssfi_key = 'SFI_{}'.format(x)
                if ssfi_key not in df.columns and basins:
                    df[ssfi_key] = indices.spi(df['q'].values,
                                               scale=x,
                                               distribution=indices.Distribution.gamma,
                                               data_start_year=1987,
                                               calibration_year_initial=1987,
                                               calibration_year_final=2021,
                                               periodicity=compute.Periodicity.monthly)

            if plot_dir and basins:
                for met_param in ['SPI_12', 'SPEI_12']:
                    use_timescale = 2
                    fig_file = os.path.join(plot_dir, '{}_{}.png'.format(sid, met_param))
                    plot_indices_panel(df, met_param, use_timescale, month_end, fig_file, title_str=sdata['STANAME'])

            elif plot_dir and month_end == 10:
                for met_param in ['SPI_12', 'SPEI_12']:
                    use_timescale = 2
                    fig_file = os.path.join(plot_dir, '{}_{}_{}.png'.format(sid, month_end, met_param))
                    plot_indices_panel_nonstream(df, met_param, use_timescale, month_end, fig_file,
                                                 title_str=sdata['NAME'])

            dct[sid] = {}
            combos = [('SPI', 'SCUI'), ('SPI', 'SIMI'), ('SPEI', 'SCUI'), ('SPEI', 'SIMI')]
            if basins:
                combos = [('SPI', 'SFI'), ('SPEI', 'SFI'), ('SFI', 'SCUI'), ('SFI', 'SIMI')]
            df = df.loc['1990-01-01':]
            pdf = df.loc[[x for x in df.index if x.month == month_end]]
            for met, use in combos:
                rmax = 0.0
                corr_ = [met, use, rmax]
                for met_ts in met_periods:
                    if use == 'SFI':
                        use_periods = range(1, 13)
                    else:
                        use_periods = range(1, 8)
                    for use_ts in use_periods:
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

        with open(out_json, 'w') as f:
            json.dump(dct, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'extracts_ietr_huc8_natet_9JAN2022')
    data_ = os.path.join(root, 'gages', 'huc8_metadata.json')
    figs = os.path.join(root, 'figures', 'panel')
    # write_indices(data_, merged, out_js, plot_dir=None, basins=False, desc_str='huc8')

    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'ietr_reclamation_6FEB2023')
    data_ = os.path.join(root, 'gages', 'usbr_metadata_north.json')
    out_js = os.path.join(root, 'analysis', 'basin_sensitivities')
    figs = os.path.join(root, 'figures', 'panel', 'ietr_reclamation_6FEB2023')
    write_indices(data_, merged, out_js, plot_dir=figs, basins=False, desc_str='usbr')

    merged = os.path.join(root, 'tables', 'input_flow_climate_tables', 'ietr_nonreclamation_6FEB2023')
    data_ = os.path.join(root, 'gages', 'nonreclamation_metadata_north.json')
    out_js = os.path.join(root, 'analysis', 'basin_sensitivities')
    figs = os.path.join(root, 'figures', 'panel', 'ietr_nonreclamation_6FEB2023')
    write_indices(data_, merged, out_js, plot_dir=figs, basins=False, desc_str='nonusbr')

# ========================= EOF ====================================================================
