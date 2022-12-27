import os
import sys
from copy import deepcopy

import numpy as np
from numpy import ones_like, where, zeros_like
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from geopandas import GeoDataFrame
from shapely.geometry import Point

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']
CLASS_NAMES = ['IRR', 'DRYL', 'WETl', 'UNCULT']
PROPS = ['aspect', 'elevation', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']
DROP = ['MGRS_TILE', '.geo', 'system:index', 'et_2020_10', 'et_2020_4',
        'et_2020_5', 'et_2020_6', 'et_2020_7', 'et_2020_9', 'study_uncu', 'id']


def random_forest(csv, n_estimators=50, out_shape=None, year=2020):
    if not isinstance(csv, DataFrame):
        print('\n', csv)
        c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        c = csv

    split = int(c.shape[0] * 0.7)

    for m in range(4, 11):
        df = deepcopy(c.loc[:split, :])
        target = 'et_{}_{}'.format(year, m)
        y = df[target].values
        df.drop(columns=DROP + [target], inplace=True)
        df.dropna(axis=1, inplace=True)
        x = df.values

        val = deepcopy(c.loc[split:, :])
        y_test = val[target].values
        geo = val.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
        val.drop(columns=DROP + [target], inplace=True)
        val.dropna(axis=1, inplace=True)
        x_test = val.values

        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=-1,
                                   bootstrap=True)

        rf.fit(x, y)
        y_pred = rf.predict(x_test)
        lr = linregress(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('month {}, r: {:.3f}, rmse: {:.3f}'.format(m, lr.rvalue, rmse))

    if out_shape:
        val['label'] = y_test
        val['label'] = val['label'].apply(lambda x: x if x > 0 else np.nan)

        val['pred'] = y_pred
        val['pred'] = np.where(y_test == 0, np.nan, y_pred)

        d = (y_pred - y_test) / (y_test + 0.0001)
        val['diff'] = np.where(y_test == 0, np.nan, d)

        gdf = GeoDataFrame(val, geometry=geo, crs="EPSG:4326")
        gdf.to_file(out_shape)
        gdf = gdf[gdf['corr'] == 0]
        incor = os.path.join(os.path.dirname(out_shape),
                             '{}_{}'.format('incor', os.path.basename(out_shape)))
        gdf.to_file(incor)

    return


def find_rf_variable_importance(csv, target, out_csv=None, drop=None, n_features=25):
    first = True
    master = {}
    df = read_csv(csv, engine='python')
    try:
        df = df[df['irr_2020'] == 2]
    except KeyError:
        pass
    df_copy = deepcopy(df)
    labels = list(df[target].values)
    try:
        df.drop(columns=drop + [target], inplace=True)
    except KeyError:
        df.drop(columns=[target], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values
    names = df.columns

    for x in range(10):
        d, _, l, _ = train_test_split(data, labels, train_size=0.67)
        print('model iteration {}'.format(x + 1))
        rf = RandomForestRegressor(n_estimators=150,
                                   n_jobs=-1,
                                   bootstrap=True)

        rf.fit(d, l)
        _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
        imp = sorted(_list, key=lambda x: x[1], reverse=True)
        print([f[0] for f in imp[:10]])

        if first:
            for (k, v) in imp:
                master[k] = v
            first = False
        else:
            for (k, v) in imp:
                master[k] += v

    master = list(master.items())
    master = sorted(master, key=lambda x: x[1], reverse=True)
    print('\ntop {} features:'.format(n_features))
    carry_features = [x[0] for x in master[:n_features]]
    print(carry_features)
    df[target] = df_copy.loc[df.index, target]
    try:
        df.index = df['id']
    except KeyError:
        pass
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    d = os.path.join(root, 'tables', 'band_extracts', 'bands_29NOV2022')
    c = os.path.join(d, 'domain_2020.csv')
    out = os.path.join(d, 'domain_2020_select.csv')
    s = os.path.join(d, 'domain_2020.shp')
    random_forest(c, year=2020)
# ========================= EOF ====================================================================
