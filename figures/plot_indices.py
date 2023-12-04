import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

box_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)


def plot_indices_panel(df, end_mo, fig_, title_str=None):
    df = df.loc['1987-01-01':]
    oct_df = df.loc[[x for x in df.index if x.month == end_mo]]
    plt.figure(figsize=(8, 12))

    ax1 = plt.subplot(4, 1, 1)
    ax1.title.set_text('Surface Water Drought Index')
    ax1.scatter(oct_df['SIMI_3'], oct_df['SDI_3'], s=15, marker='.', c='k')
    slope, intercept, r_value, p_value, std_err = linregress(oct_df['SIMI_3'], oct_df['SDI_3'])
    string = f'r$^2$ = {r_value ** 2:.2f}\n$\\beta$ = {slope:.2f}\nn = {oct_df.shape[0]}'
    ax1.annotate(string, xy=(0.015, 0.68), xycoords='axes fraction', fontsize=10, bbox=box_props)

    ax1.set(xlabel='SIMI - 3 Months', ylabel='SDI - 3 Months')

    ax2 = plt.subplot(4, 1, 2)
    ax2.set(xlabel='SIMI - 3 Months', ylabel='SPI - 4 Months')
    ax2.title.set_text('Standardized Precipitation Index')
    ax2.scatter(oct_df['SIMI_3'], oct_df['SPI_4'], s=15, marker='.', c='b')
    slope, intercept, r_value, p_value, std_err = linregress(oct_df['SIMI_3'], oct_df['SPI_4'])
    string = f'r$^2$ = {r_value ** 2:.2f}\n$\\beta$ = {slope:.2f}\nn = {oct_df.shape[0]}'
    ax2.annotate(string, xy=(0.015, 0.68), xycoords='axes fraction', fontsize=10, bbox=box_props)

    ax3 = plt.subplot(4, 1, 3)
    ax3.set(xlabel='SIMI - 3 Months', ylabel='SPEI - 4 Months')
    ax3.title.set_text('Standardized Precipitation and Evapotranspiration Index')
    ax3.scatter(oct_df['SIMI_3'], oct_df['SPEI_4'], s=15, marker='.', c='r')
    slope, intercept, r_value, p_value, std_err = linregress(oct_df['SIMI_3'], oct_df['SPEI_4'])
    string = f'r$^2$ = {r_value ** 2:.2f}\n$\\beta$ = {slope:.2f}\nn = {oct_df.shape[0]}'
    ax3.annotate(string, xy=(0.015, 0.68), xycoords='axes fraction', fontsize=10, bbox=box_props)

    lim = [-3, 3]
    [ax.set_ylim(lim) for ax in [ax1, ax2, ax3]]
    [ax.set_xlim(lim) for ax in [ax1, ax2, ax3]]

    ax4 = plt.subplot(4, 1, 4)
    df = df.loc['2010-01-01': '2021-12-31']
    df = df[['SPEI_4', 'SPI_4', 'SDI_3']]
    df.columns = ['SPEI - 4 Months', 'SPI - 4 Months', 'SDI - 3 Months']
    df.plot(ax=ax4, color=['g', 'purple', 'k'], ylabel='Date', xlabel='Index Value', linewidth=1)
    ax4.legend(loc=2)
    lim = [-3.2, 3.2]
    ax4.set_ylim(lim)
    plt.suptitle(title_str)
    plt.tight_layout()
    plt.savefig(fig_)
    # plt.show()
    # print(os.path.basename(fig_))
    pass


def plot_indices_panel_nonstream(df, met_param, cu_timescale, end_mo, fig_, title_str=None):
    df = df.loc['1987-01-01':]
    oct_df = df.loc[[x for x in df.index if x.month == end_mo]]
    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 3, 1)
    ax1.set(xlabel='{} - 12 Month'.format(met_param), ylabel='SCUI - {} Months'.format(cu_timescale))
    ax1.title.set_text('Standardized Consumptive Use Index')
    ax1.scatter(oct_df['{}'.format(met_param)], oct_df['SCUI_{}'.format(cu_timescale)], s=15, marker='.', c='b')

    ax2 = plt.subplot(2, 3, 2)
    ax2.set(xlabel='{} - 12 Month'.format(met_param), ylabel='SIMI - {} Months'.format(cu_timescale))
    ax2.title.set_text('Standardized Irrigation Management Index')
    ax2.scatter(oct_df['{}'.format(met_param)], oct_df['SIMI_{}'.format(cu_timescale)], s=15, marker='.', c='r')

    lim = [-3, 3]
    [ax.set_ylim(lim) for ax in [ax1, ax2]]
    [ax.set_xlim(lim) for ax in [ax1, ax2]]

    plt.suptitle(title_str)
    plt.tight_layout()
    plt.savefig(fig_)
    # print(os.path.basename(fig_))


if __name__ == '__main__':
    f = '/home/dgketchum/Downloads/SIMI_06192500.csv'
    df = pd.read_csv(f, index_col='dt', infer_datetime_format=True, parse_dates=True)
    plot_indices_panel(df, 8, f.replace('csv', 'png'), 'Standardized Irrigation Management Index - August\n'
                                                       'above USGS Gage 06192500:\n'
                                                       ' Yellowstone River near Livingston, MT\n')

# ========================= EOF ====================================================================
