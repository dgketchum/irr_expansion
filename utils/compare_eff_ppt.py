import os
import json
from datetime import date
from itertools import product


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from scipy.stats import linregress

from gage_data import hydrograph


def compare(meta, interp, swb, outfig):
    with open(meta, 'r') as f_obj:
        stations = json.load(f_obj)

    sdct = {m: [] for m in range(4, 11)}
    idct = {m: [] for m in range(4, 11)}
    ssums, isums = [], []

    for e, (sid, data) in enumerate(stations.items()):
        s_file = os.path.join(swb, '{}.csv'.format(sid))
        i_file = os.path.join(interp, '{}.csv'.format(sid))

        try:
            sdf = hydrograph(s_file).loc['1991-01-01':]
            idf = hydrograph(i_file).loc['1991-01-01':]
        except FileNotFoundError:
            print(sid, 'not found')

        for m in range(4, 11):
            dates = [idx for idx in sdf.index if idx.month == m]
            [sdct[m].append(x) for x in sdf.loc[dates, 'cc']]
            [idct[m].append(x) for x in idf.loc[dates, 'cc']]

        sdf_sum = sdf['cc'].sum()
        idf_sum = idf['cc'].sum()

        if np.isnan(np.log10(sdf_sum)) or np.isnan(np.log10(idf_sum)):
            print(sid, 'has nonfinite values')
            continue

        ssums.append(np.log10(sdf_sum))
        isums.append(np.log10(idf_sum))
        print('{} swb: {:.2f}, interp: {:.2f}'.format(sid, np.log10(sdf_sum), np.log10(idf_sum)))

    fig, ax = plt.subplots(2, 4,)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    lr = linregress(ssums, isums)
    ax[0, 0].scatter(ssums, isums, s=15, marker='.', c='b')
    ax[0, 0].title.set_text('Growing Season Apr - Oct')
    ax[0, 0].set(xlabel='Log (SSEBop - TerraClimate)', ylabel='Log (SSEBop - SSEBop Interpolated Natural)')
    txt = AnchoredText('n={}\nr={:.3f}\nb={:.3f}'.format(len(isums), lr.rvalue ** 2, lr.slope), loc=4)
    ax[0, 0].add_artist(txt)

    for e, ax_ in enumerate(ax.ravel()[1:], start=4):
        s, i = sdct[e], idct[e]
        lr = linregress(s, i)
        mstr = date.strftime(date(1990, e, 1), '%B')
        ax_.scatter(s, i, s=15, marker='.', c='b')
        ax_.title.set_text('Crop Consumption {}'.format(mstr))
        ax_.set(xlabel='SSEBop - TerraClimate', ylabel='SSEBop - SSEBop Interpolated Natural')
        txt = AnchoredText('n={}\nr={:.3f}\nb={:.3f}'.format(len(isums), lr.rvalue ** 2, lr.slope), loc=4)
        ax_.add_artist(txt)

    plt.tight_layout()
    plt.savefig('/home/dgketchum/Downloads/iwu_estimates.png')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    meta_ = os.path.join(root, 'expansion', 'gages', 'irrigated_gage_metadata.json')
    swb_ = os.path.join(root, 'impacts', 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
    interp_ = os.path.join(root, 'expansion', 'tables', 'input_flow_climate_tables', 'extracts_NatET_terrain_23DEC2022')
    fig_ = os.path.join(root, 'expansion', '../figures', 'swb_interp_comparison')

    compare(meta_, interp_, swb_, fig_)

# ========================= EOF ====================================================================
