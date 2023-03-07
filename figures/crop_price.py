import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.cdl import ppi_to_cdl_crop, study_area_crops
from gridded_data import BASIN_STATES


def plot_regressions(ppi_data, nass_price, fig_dir):
    ppi = ppi_to_cdl_crop()
    cdl = study_area_crops()
    inv_cdl = {v[0]: k for k, v in cdl.items() if v[0] != 'Farm Products'}
    with open(nass_price, 'r') as fp:
        nass = json.load(fp)

    nass_ind = [pd.to_datetime('{}-07-01'.format(y)) for y in range(2003, 2023)]
    df = pd.read_csv(ppi_data, index_col='DATE', infer_datetime_format=True, parse_dates=True)
    df = df.loc['2008-07-01': '2022-12-31']
    scale = df.loc['2008-07-01'] / 100
    df = df / scale

    for c, ind in ppi.items():
        if c == 'Farm Products':
            continue
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(16)

        try:
            code = inv_cdl[c]
            annual = nass[str(code)]
            national = annual['national']

            plt.plot(df.index, df[ind], label='{} - {}'.format(c, ind))
            plt.plot(df.index, df[ppi['Farm Products']], c='b', lw=3, label='Farm Products')

            for s in BASIN_STATES:

                data = [x[1] for x in annual[s][1]]
                if not data:
                    continue
                index = [pd.to_datetime('{}-07-01'.format(x[0])) for x in annual[s][1]]
                legend_item = '{} - {}'.format(c, s)
                if national:
                    legend_item = '{} - National'.format(c)
                series = pd.Series(data=data, name=legend_item, index=index)
                scale = series.loc['2008-07-01'] / 100
                series = series / scale
                series = series.loc['2008-07-01':]
                series.plot()
                if national:
                    break

            print('plotting', c)

        except KeyError as e:
            print(e, c, ind)
            plt.close()
            continue

        plt.legend()
        plt.tight_layout()
        # plt.show()
        name_ = c.replace(' ', '_').replace('/', '_')
        _file = os.path.join(fig_dir, 'nass_{}.png'.format(name_))
        plt.savefig(_file)
        print(_file)
        plt.close()
        plt.clf()


if __name__ == '__main__':
    f = '/media/research/IrrigationGIS/expansion/tables/crop_value/ppi_cdl_monthly.csv'
    fig_ = '/media/research/IrrigationGIS/expansion/figures/crop_prices'
    price_ts = '/media/research/IrrigationGIS/expansion/tables/crop_value/time_series.json'
    plot_regressions(f, price_ts, fig_)
# ========================= EOF ====================================================================
