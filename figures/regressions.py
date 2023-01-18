import os
from scipy.stats import linregress

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def plot_regressions(f, fig_dir):
    df = gpd.read_file(f)
    param = 'season'
    x = df[param]
    for c in df.columns:
        if c in ['geometry']:
            continue
        y = df[c].values
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(16)
        lr = linregress(x, y)
        ax.scatter(x, y, s=15, marker='.', c='b')
        ax.title.set_text(c)
        ax.set(xlabel=param, ylabel=c)
        txt = AnchoredText('n={}\nr={:.3f}\nb={:.3f}'.format(df.shape[0], lr.rvalue ** 2, lr.slope), loc='lower right')
        ax.add_artist(txt)
        plt.tight_layout()
        _file = os.path.join(fig_dir, '{}.png'.format(c))
        plt.savefig(_file)
        print(_file)
        plt.close()
        plt.clf()


if __name__ == '__main__':
    f = '/media/research/IrrigationGIS/expansion/tables/prepped_bands/bands_2020.shp'
    fig_ = '/media/research/IrrigationGIS/expansion/figures/regressions'
    plot_regressions(f, fig_)
# ========================= EOF ====================================================================
