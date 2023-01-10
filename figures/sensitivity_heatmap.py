import os
import json
from datetime import date, datetime, timedelta
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (12, 8),
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


def heatmap(data_dir, fig_d, param='r2'):
    met_periods = [1, 6, 12, 18, 24, 30, 36]

    for month in range(5, 11):
        use_periods = list(range(1, month - 2))

        data = os.path.join(data_dir, 'irrigated_indices_{}.json'.format(month))

        with open(data, 'r') as f_obj:
            data = json.load(f_obj)

        combos = [('SPI', 'SCUI'), ('SPI', 'SIMI'), ('SPEI', 'SCUI'), ('SPEI', 'SIMI')]

        keys = data[list(data.keys())[0]].keys()
        param_vals = {k: [] for k in keys}

        for sid, v in data.items():
            for key, val in v.items():
                if val['p'] < 0.05:
                    param_vals[key].append(val[param])

        for met, use in combos:
            grid = np.zeros((len(use_periods), len(met_periods)))
            for i, u in enumerate(use_periods):
                for j, m in enumerate(met_periods):
                    grid[i, j] = np.mean(param_vals['{}_{}_{}_{}'.format(met, m, use, u)])

            grid = pd.DataFrame(index=use_periods, columns=met_periods, data=grid)
            ax = sns.heatmap(grid, annot=True, cmap='magma')
            y, x = np.unravel_index(np.argmax(np.abs(grid), axis=None), grid.shape)
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='green', lw=4, clip_on=False))
            plt.xlabel('{} - Months'.format(use))
            plt.ylabel('{} - Months'.format(met))
            mstr = datetime(1991, month, 1).strftime('%B')
            plt.title('Mean Study Area Basin Correlations\nSensitivity in {}'.format(mstr))
            plt.tight_layout()

            fig_file = os.path.join(fig_d, param, '{}_{}_{}_heatmap.png'.format(met.lower(), use.lower(), month))
            plt.savefig(fig_file)
            plt.close()
            print('{:.3f} {}'.format(grid.values[y, x], os.path.basename(fig_file)))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'
    figs = os.path.join(root, 'figures', 'heatmaps')
    js_ = os.path.join(root, 'analysis', 'basin_sensitivities')
    heatmap(js_, figs, param='r2')

# ========================= EOF ====================================================================
