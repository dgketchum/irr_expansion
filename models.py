import os
import sys
from pprint import pprint
from time import time
from subprocess import call
from copy import deepcopy

import numpy as np
from numpy import dot, mean, flatnonzero, ones_like, where, zeros_like
from pandas import read_csv, concat, DataFrame
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold

from geopandas import GeoDataFrame
from shapely.geometry import Point

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']
CLASS_NAMES = ['IRR', 'DRYL', 'WETl', 'UNCULT']


def random_forest(csv, n_estimators=50, out_shape=None):
    if not isinstance(csv, DataFrame):
        print('\n', csv)
        c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        c = csv

    c = c[c['irr_2020'] == 2]
    split = int(c.shape[0] * 0.7)

    df = deepcopy(c.loc[:split, :])
    et_cols = [x for x in df.columns if 'et_2020' in x]
    target = et_cols[-2]
    y = df[target].values
    df.drop(columns=et_cols + ['.geo', 'system:index'], inplace=True)
    df.dropna(axis=1, inplace=True)
    x = df.values

    val = deepcopy(c.loc[split:, :])
    y_test = val[target].values
    geo = val.apply(lambda x: Point(x['Lon_GCS'], x['LAT_GCS']), axis=1)
    val.drop(columns=et_cols + ['.geo', 'system:index'], inplace=True)
    val.dropna(axis=1, inplace=True)
    x_test = val.values

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               n_jobs=-1,
                               bootstrap=True)

    rf.fit(x, y)
    y_pred = rf.predict(x_test)
    if out_shape:
        val['label'] = y_test
        val['label'] = val['label'].apply(lambda x: x if x > 0 else np.nan)

        val['pred'] = y_pred
        val['pred'] = np.where(y_test == 0, np.nan, y_pred)

        d = (y_pred - y_test) / (y_test + 0.0001)
        val['diff'] = np.where(y_test == 0, np.nan, d)

        ones = ones_like(y_test)
        zeros = zeros_like(y_test)
        val['corr'] = where(y_pred == y_test, ones, zeros)

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
    root = '/media/research/IrrigationGIS/expansion/tables/points_extracts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion/tables/points_extracts'

    c = os.path.join(root, 'uinta_2020.csv')
    out = os.path.join(root, 'uinta_2020_select.csv')
    s = os.path.join(root, 'uinta_2020.shp')
    find_rf_variable_importance(out, target='et_2020_9', out_csv=None,
                                drop=['.geo', 'system:index', 'et_2020_10', 'et_2020_4',
                                      'et_2020_5', 'et_2020_6', 'et_2020_7', 'et_2020_8'])
# ========================= EOF ====================================================================
