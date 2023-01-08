import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from climate_indices import compute, indices, utils

from gage_data import hydrograph


def plot_indices(meta, gridded_data, out_figs, overwrite=False):
    with open(meta, 'r') as f_obj:
        stations = json.load(f_obj)

    for sid, station in stations.items():

        if sid != '06052500':
            continue

        fig_ = os.path.join(out_figs, '{} {}.png'.format(sid, station['STANAME']))
        if os.path.exists(fig_) and not overwrite:
            continue

        _file = os.path.join(gridded_data, '{}.csv'.format(sid))
        try:
            df = hydrograph(_file)
        except FileNotFoundError:
            print(_file, 'does not exist, skipping')
            continue

        scale = 12

        df['ppt'] = (df['gm_ppt'] - df['gm_ppt'].min()) / (df['gm_ppt'].max() - df['gm_ppt'].min()) + 0.001
        df['etr'] = (df['gm_etr'] - df['gm_etr'].min()) / (df['gm_etr'].max() - df['gm_etr'].min()) + 0.001
        df = df[['q', 'cc', 'ppt', 'etr']]

        for x in range(1, 13):
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

        df['SSFI'] = indices.spi(df['q'].values,
                                 scale=scale,
                                 distribution=indices.Distribution.gamma,
                                 data_start_year=1987,
                                 calibration_year_initial=1987,
                                 calibration_year_final=2021,
                                 periodicity=compute.Periodicity.monthly)

        df['SCUI'] = indices.spi(df['cc'].values,
                                 scale=7,
                                 distribution=indices.Distribution.gamma,
                                 data_start_year=1987,
                                 calibration_year_initial=1987,
                                 calibration_year_final=2021,
                                 periodicity=compute.Periodicity.monthly)

        df = df.loc['1987-01-01':]
        oct_df = df.loc[[x for x in df.index if x.month == 10]]
        plt.figure(figsize=(12, 6))

        ax1 = plt.subplot(2, 3, 1)
        ax1.set(xlabel='SPI - 12 Month', ylabel='SSFI - 12 Month')
        ax1.title.set_text('Standardized Streamflow Index')
        ax1.scatter(oct_df['SPI_12'], oct_df['SSFI'], s=15, marker='.', c='b')

        ax2 = plt.subplot(2, 3, 2)
        ax2.set(xlabel='SPEI - 12 Month', ylabel='SSFI - 12 Month')
        ax2.title.set_text('Standardized Streamflow Index')
        ax2.scatter(oct_df['SPEI_12'], oct_df['SSFI'], s=15, marker='.', c='b')

        ax3 = plt.subplot(2, 3, 3)
        ax3.set(xlabel='SPI - 12 Month', ylabel='SCUI')
        ax3.title.set_text('Standardized Consumptive Use Index')
        ax3.scatter(oct_df['SPI_12'], oct_df['SCUI'], s=15, marker='.', c='b')

        lim = [-3, 3]
        [ax.set_ylim(lim) for ax in [ax1, ax2, ax3]]
        [ax.set_xlim(lim) for ax in [ax1, ax2, ax3]]

        ax4 = plt.subplot(2, 1, 2)
        df[['SPI_12', 'SPEI_12', 'SSFI']].plot(ax=ax4, color=['g', 'purple', 'b'])
        ax4.scatter(df.index, df['SCUI'], label='SCUI', s=5, marker='+', c='r')
        ax4.legend(loc=2)
        ax4.set_ylim(lim)
        plt.suptitle(station['STANAME'])
        plt.tight_layout()
        plt.savefig(fig_)
        print(os.path.basename(fig_))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    data_tables = os.path.join(root, 'impacts', 'tables',
                               'input_flow_climate_tables',
                               'IrrMapperComp_21OCT2022')
    sid_ = '06192500'
    d = os.path.join(root, 'expansion')
    data_ = os.path.join(d, 'gages', 'irrigated_gage_metadata.json')
    figs = os.path.join(d, 'figures', 'indices')
    plot_indices(data_, data_tables, figs, overwrite=True)

# ========================= EOF ====================================================================
