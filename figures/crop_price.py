import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.cdl import ppi_to_cdl_crop, study_area_crops
from gridded_data import BASIN_STATES


def plot_regressions(ppi_data, fig_dir):
    ppi = ppi_to_cdl_crop()
    df = pd.read_csv(ppi_data, index_col='DATE', infer_datetime_format=True, parse_dates=True)
    df = df.loc['2000-01-01': '2022-12-31']
    scale = df.loc['2000-01-01'] / 100
    df = df / scale

    for c, ind in ppi.items():

        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(22)

        if c == 'Farm Products':
            continue

        try:
            plt.plot(df.index, df[ind], label='{} - {}'.format(c, ind))
            plt.plot(df.index, df[ppi['Farm Products']], c='b', lw=3, label='Farm Products')
            print('plotting', c)

        except KeyError as e:
            print(e, c, ind)
            plt.close()
            continue

        plt.legend()
        plt.tight_layout()
        # plt.show()
        name_ = c.replace(' ', '_').replace('/', '_')
        _file = os.path.join(fig_dir, '{}.png'.format(name_))
        plt.savefig(_file)
        print(_file)
        plt.close()
        plt.clf()


if __name__ == '__main__':
    f = '/media/research/IrrigationGIS/expansion/tables/crop_value/ppi_cdl_monthly.csv'
    fig_ = '/media/research/IrrigationGIS/expansion/figures/crop_prices'
    price_ts = '/media/research/IrrigationGIS/expansion/tables/crop_value/time_series.json'
    plot_regressions(f, fig_)
# ========================= EOF ====================================================================
