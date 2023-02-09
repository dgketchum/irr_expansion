import os

import matplotlib.pyplot as plt


def plot_indices_panel(df, met_param, cu_timescale, end_mo, fig_, title_str=None):
    df = df.loc['1987-01-01':]
    oct_df = df.loc[[x for x in df.index if x.month == end_mo]]
    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 3, 1)
    ax1.set(xlabel='{} - 12 Month'.format(met_param), ylabel='SFI - 12 Month')
    ax1.title.set_text('Standardized Streamflow Index')
    ax1.scatter(oct_df['{}'.format(met_param)], oct_df['SFI_12'], s=15, marker='.', c='k')

    ax2 = plt.subplot(2, 3, 3)
    ax2.set(xlabel='{} - 12 Month'.format(met_param), ylabel='SCUI - {} Months'.format(cu_timescale))
    ax2.title.set_text('Standardized Consumptive Use Index')
    ax2.scatter(oct_df['{}'.format(met_param)], oct_df['SCUI_{}'.format(cu_timescale)], s=15, marker='.', c='b')

    ax3 = plt.subplot(2, 3, 2)
    ax3.set(xlabel='{} - 12 Month'.format(met_param), ylabel='SIMI - {} Months'.format(cu_timescale))
    ax3.title.set_text('Standardized Irrigation Management Index')
    ax3.scatter(oct_df['{}'.format(met_param)], oct_df['SIMI_{}'.format(cu_timescale)], s=15, marker='.', c='r')

    lim = [-3, 3]
    [ax.set_ylim(lim) for ax in [ax1, ax2, ax3]]
    [ax.set_xlim(lim) for ax in [ax1, ax2, ax3]]

    ax4 = plt.subplot(2, 1, 2)
    df[['SPI_12', 'SPEI_12', 'SFI_12']].plot(ax=ax4, color=['g', 'purple', 'k'])
    ax4.scatter(oct_df.index, oct_df['SCUI_{}'.format(cu_timescale)], label='SCUI', s=5, marker='+', c='b')
    ax4.scatter(oct_df.index, oct_df['SIMI_{}'.format(cu_timescale)], label='SIMI', s=5, marker='+', c='r')
    ax4.legend(loc=2)
    ax4.set_ylim(lim)
    plt.suptitle(title_str)
    plt.tight_layout()
    plt.savefig(fig_)
    print(os.path.basename(fig_))


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
    print(os.path.basename(fig_))


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
