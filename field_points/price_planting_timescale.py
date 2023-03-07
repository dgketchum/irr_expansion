import os
import json
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from scipy.stats.stats import linregress
from fuzzywuzzy import process

from utils.cdl import cdl_key


def planting_area_prediction(price_csv, area_js):
    with open(area_js, 'r') as _file:
        areas = json.load(_file)

    areas = {int(k): v for k, v in areas.items()}
    means = [(k, np.array(v).mean()) for k, v in areas.items()]
    means = sorted(means, key=lambda x: x[1], reverse=True)

    incl = sorted([k for k, v in means if v > 100 and k not in [61, 121, 122, 152, 176, 190, 195]])
    cdl = cdl_key()
    bnames = [c for c in os.listdir(price_csv)]
    code_files = {k: process.extractOne(v[0], bnames)[0] for k, v in cdl.items() if len(v[0]) > 0}
    code_files = {k: os.path.join(price_csv, v) for k, v in code_files.items()}

    lags, areas_l = [], []
    for c, area in areas.items():
        if c not in incl:
            continue
        crop = cdl[c][0]
        price = pd.read_csv(code_files[c], index_col=0, infer_datetime_format=True, parse_dates=True)
        price = price.mean(axis=1)
        r_max, p_min, opt_lag = 0, 1, 0
        for lag in range(1, 36):
            data = [price.loc[pd.to_datetime('{}-05-01'.format(y)) - relativedelta(months=lag)]
                    for y in range(2011, 2022)]
            a = area[3:]
            lr = linregress(data, a)
            b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
            if r ** 2 > r_max:
                r_max, opt_lag, p_min = r ** 2, lag, p
        print('{}: {}, lag:{}, p: {:.3f}, r: {:.3f}, area: {:.3f}'.format(c, crop, opt_lag, p_min, r_max,
                                                                          np.array(area).mean()))
        lags.append(opt_lag)
        areas_l.append(np.array(area).mean())

    weights = np.array(areas_l) / sum(areas_l)
    weighted_lags = weights * np.array(lags)
    print(np.sum(weighted_lags))


if __name__ == '__main__':
    deflated = '/media/research/IrrigationGIS/expansion/tables/crop_value/deflated'
    cdl_area_ = '/media/research/IrrigationGIS/expansion/tables/cdl/cdl_area_timesereies.json'
    planting_area_prediction(deflated, cdl_area_)
# ========================= EOF ====================================================================
