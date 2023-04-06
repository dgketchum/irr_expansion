import os
import json
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (12, 12),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': True,
          'ytick.left': True,
          'ytick.right': True,
          }
plt.rcParams.update(params)
plt.style.use('seaborn-darkgrid')
sns.set_style("dark", {'axes.linewidth': 0.5})


def aggregated_heatmap(data_dir, fig_d, param='r2', basins=True, desc_str='basins', weighted=False):
    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    target_timescales = []
    for month in range(5, 11):
        use_periods = list(range(1, month - 2))

        data = os.path.join(data_dir, '{}_{}.json'.format(desc_str, month))

        with open(data, 'r') as f_obj:
            data = json.load(f_obj)

        if basins:
            combos = [('SPI', 'SFI'), ('SPEI', 'SFI'), ('SFI', 'SCUI'), ('SFI', 'SIMI')]
        else:
            combos = [('SPEI', 'SIMI'), ('SPI', 'SIMI'), ('SPEI', 'SCUI'), ('SPI', 'SCUI')]

        keys = [list(v.keys()) for k, v in data.items()]
        keys = list(set([item for sublist in keys for item in sublist]))

        param_vals = {k: [] for k in keys}
        area_vals = []

        for sid, v in data.items():
            for key in keys:
                if key == 'irr_area':
                    area_vals.append(v[key])
                    continue
                try:
                    param_vals[key].append(v[key][param])
                except KeyError:
                    param_vals[key].append(np.nan)

        for met, use in combos:
            grid = np.zeros((len(use_periods), len(met_periods)))
            for i, u in enumerate(use_periods):
                for j, m in enumerate(met_periods):
                    slice_ = param_vals['{}_{}_{}_{}'.format(met, m, use, u)]
                    if weighted:
                        slice_ = np.array(slice_)
                        nan = np.isnan(slice_)
                        area_sum_, area_vals_ = np.sum(np.array(area_vals)[~nan]), np.array(area_vals)[~nan]
                        weight = area_vals_ / area_sum_
                        grid[i, j] = np.sum(slice_[~nan] * weight)
                    else:
                        grid[i, j] = np.mean(slice_)

            grid = pd.DataFrame(index=use_periods, columns=met_periods, data=grid).values

            ax = sns.heatmap(grid, square=True, annot=True, cmap='magma', cbar=False,
                             xticklabels=met_periods, yticklabels=use_periods)

            y, x = np.unravel_index(np.argmax(np.abs(grid), axis=None), grid.shape)
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='green', lw=4, clip_on=False))
            plt.xlabel('{} - Months'.format(met))
            plt.ylabel('{}\nMonths'.format(use))
            plt.yticks(rotation=0)
            ax.tick_params(axis='y', which='major', pad=30)
            # plt.subplots_adjust(left=0.5)
            plt.tight_layout()
            fig_file = os.path.join(fig_d, param, '{}_{}_{}_heatmap.png'.format(met.lower(), use.lower(), month))
            plt.savefig(fig_file, bbox_inches='tight')
            plt.close()
            target_timescales.append((month, x + 1, y + 1))
            print('{:.3f} {}'.format(grid[y, x], os.path.basename(fig_file)))

    print('max correlation timescales: \n', target_timescales)


def fields_heatmap(csv, attrs, fig_d):
    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    first = True
    bnames = [x for x in os.listdir(csv) if x.strip('.csv') in BASIN_STATES]
    bnames.sort()
    csv = [os.path.join(csv, x) for x in bnames]
    attrs = [os.path.join(attrs, x) for x in os.listdir(attrs) if x in bnames]
    attrs.sort()
    for m, f in zip(attrs, csv):
        c = pd.read_csv(f, index_col=0)
        meta = pd.read_csv(m, index_col='OPENET_ID')
        match = [i for i in c.index if i in meta.index]
        c.loc[match, 'usbrid'] = meta.loc[match, 'usbrid']

        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    for i in range(3):
        if i == 0:
            part = 'all'
            tdf = df.copy()
        elif i == 1:
            part = 'usbr'
            tdf = df[df['usbrid'] > 0]
        else:
            part = 'nonusbr'
            tdf = df[df['usbrid'] == 0]

        for month in range(4, 11):
            use_periods = list(range(1, month - 2))

            title = 'Correlation'

            combos = [('SPEI', 'SIMI')]

            for met, use in combos:
                grid = np.zeros((len(use_periods), len(met_periods)))
                for i, u in enumerate(use_periods):
                    for j, m in enumerate(met_periods):
                        slice_ = tdf['met{}_ag{}_fr{}'.format(m, u, month)].values
                        grid[i, j] = np.mean(slice_)

                grid = pd.DataFrame(index=use_periods, columns=met_periods, data=grid).values
                ax = sns.heatmap(grid, square=True, annot=True, cmap='magma', fmt='.3g')
                y, x = np.unravel_index(np.argmax(np.abs(grid), axis=None), grid.shape)
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='green', lw=4, clip_on=False))
                plt.xlabel('{} - Months'.format(met))
                plt.ylabel('{}\nMonths'.format(use))
                plt.yticks(rotation=0)
                ax.tick_params(axis='y', which='major', pad=30)
                mstr = datetime(1991, month, 1).strftime('%B')
                plt.title('Mean Field {}\n{}'.format(title, mstr))
                plt.subplots_adjust(left=0.5)
                plt.tight_layout()
                fig_file = os.path.join(fig_d, part, '{}_{}_{}_heatmap.png'.format(met.lower(), use.lower(), month))
                plt.savefig(fig_file, bbox_inches='tight')
                plt.close()
                print('{:.3f} {}'.format(grid[y, x], os.path.basename(fig_file)))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    figs = os.path.join(root, 'figures', 'heatmaps', 'huc8')
    js_ = os.path.join(root, 'analysis', 'huc8_sensitivities')
    aggregated_heatmap(js_, figs, param='r2', basins=False, desc_str='huc8', weighted=True)

    p = 'scui'
    in_ = '/media/nvm/field_pts/indices/{}'.format(p)
    meta = '/media/nvm/field_pts/fields_data/fields_shp'
    figs = os.path.join(root, 'figures', 'heatmaps', 'fields', '{}'.format(p))
    # fields_heatmap(in_, meta, figs)

# ========================= EOF ===================================================================
