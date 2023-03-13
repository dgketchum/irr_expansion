import os
import json
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

from utils.cdl import cdl_key

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def planting_area_prediction(price_csv, area_js, keys, files_js, timescales, mode='mv'):
    with open(area_js, 'r') as _file:
        areas_ = json.load(_file)

    with open(keys, 'r') as _file:
        keys = json.load(_file)['keys']

    areas_ = {int(k): v for k, v in areas_.items()}
    means = [(k, np.array(v).mean()) for k, v in areas_.items()]
    means = sorted(means, key=lambda x: x[1], reverse=True)

    incl = sorted([k for k, v in means if v > 100 and k not in [59, 61, 121, 122, 152, 176, 190, 195]])
    cdl = cdl_key()
    bnames = [c for c in os.listdir(price_csv)]
    code_files = {k: process.extractOne(v[0], bnames)[0] for k, v in cdl.items() if len(v[0]) > 0}
    code_files = {k: os.path.join(price_csv, v) for k, v in code_files.items() if k in incl}

    opt_flag, opt_tlag = None, None
    fcode, tcode, rmse_min = None, None, None
    lags = {}

    for fcode, tcode in keys:
        area = areas_[tcode]
        tcrop = cdl[tcode][0]
        fcrop = cdl[fcode][0]
        if tcode not in code_files.keys():
            continue
        price = pd.read_csv(code_files[tcode], index_col=0, infer_datetime_format=True, parse_dates=True)
        price = price.mean(axis=1)
        rmse_min, p_min, opt_lag = 1e7, 1, 0

        for t_lag in range(1, 24):
            years = list(range(2008, 2022))
            delta = relativedelta(months=t_lag)
            tdata = [price.loc[pd.to_datetime('{}-05-01'.format(y)) - delta] for y in years]

            for f_lag in range(1, 24):
                delta = relativedelta(months=f_lag)
                fdata = [price.loc[pd.to_datetime('{}-05-01'.format(y)) - delta] for y in years]
                a = area

                if f_lag > 20 or t_lag > 20:
                    tdata, fdata, a = tdata[1:], fdata[1:], area[1:]

                if mode == 'mv':
                    data = np.array([fdata, tdata]).T
                else:
                    data = np.array([tdata]).T

                try:
                    lr = LinearRegression().fit(data, a)
                except ValueError:
                    continue

                pred = lr.predict(data)
                rmse = mean_squared_error(a, pred, squared=False)

                if rmse < rmse_min:
                    rmse_min, opt_flag, opt_tlag = rmse, f_lag, t_lag

                if mode == 'uv':
                    break

        if rmse_min > 1e6:
            print('\n{} to {} failed\n'.format(fcrop, tcrop))
            continue
        else:
            print('{} to {}: {}/{}, lag: f {} t {}, rmse: {:.1f}'.format(fcrop, tcrop, fcode, tcode,
                                                                         opt_flag, opt_tlag, rmse_min,
                                                                         np.array(area).mean()))

        lags['{}_{}'.format(fcode, tcode)] = {'flag': opt_flag, 'tlag': opt_tlag, 'rmse': rmse_min}

    if mode == 'mv':
        with open(files_js, 'w') as fp:
            json.dump(code_files, fp, indent=4)

        with open(timescales, 'w') as fp:
            json.dump(lags, fp, indent=4)


if __name__ == '__main__':
    deflated = '/media/research/IrrigationGIS/expansion/tables/crop_value/deflated'
    files_ = '/media/research/IrrigationGIS/expansion/tables/crop_value/price_files.json'
    cdl_area_ = '/media/research/IrrigationGIS/expansion/tables/cdl/cdl_area_timesereies.json'
    transition_keys = '/media/research/IrrigationGIS/expansion/analysis/transition/keys.json'
    time_scales = '/media/research/IrrigationGIS/expansion/analysis/transition/time_scales.json'
    planting_area_prediction(deflated, cdl_area_, transition_keys, files_, mode='mv', timescales=time_scales)
# ========================= EOF ====================================================================
