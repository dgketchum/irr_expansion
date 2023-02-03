import os
import sys
import json
from pprint import pprint
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
from scipy.stats import linregress
from geopandas import GeoDataFrame
from shapely.geometry import Point

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from call_ee import PROPS

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)


class RandomForest:

    def __init__(self):
        self.rf = None
        self.feature_importances = None

    def fit(self, x, y):
        self.rf = RandomForestRegressor(n_estimators=150,
                                        n_jobs=-1,
                                        bootstrap=True,
                                        oob_score=True,
                                        min_samples_leaf=5)
        self.rf.fit(x, y)
        self.feature_importances = self.rf.feature_importances_

    def predict(self, x):
        return self.rf.predict(x)


def model_data(csv, out_shape=None, year=2020, clamp_et=False, importance=None, glob=None,
               n_feats=10):
    if not isinstance(csv, DataFrame):
        print('\n', csv)
        c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        c = csv

    et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]
    for etc in et_cols:
        c[etc] = c[etc].values.astype(float) * 0.00001
    c['season'] = c[et_cols].sum(axis=1)
    print(c.shape)
    if clamp_et:
        c = c[c['season'] < c['ppt_wy_et'] * 0.001]

    c.drop(columns=['season'], inplace=True)

    split = int(c.shape[0] * 0.7)
    val_df = None

    targets, features, first = [], None, True
    for m in range(4, 11):
        df = deepcopy(c.loc[:split, :])
        mstr = str(m)
        target = 'et_{}_{}'.format(year, m)

        df.dropna(axis=0, inplace=True)
        y = df[target].values
        df.drop(columns=et_cols, inplace=True)
        x = df.values
        targets.append(target)
        val = deepcopy(c.loc[split:, :])

        if first:
            features = list(df.columns)
            geo = val.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
            val_df = deepcopy(c.loc[split:, :])
            first = False

        val.dropna(axis=0, inplace=True)
        y_test = val[target].values
        val.drop(columns=et_cols, inplace=True)
        x_test = val.values

        rf = RandomForest()
        rf.fit(x, y)
        y_pred = rf.predict(x_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        val_df['label_{}'.format(mstr)] = y_test
        val_df['label_{}'.format(mstr)] = val_df['label_{}'.format(mstr)].apply(lambda x: x if x > 0 else np.nan)
        val_df['pred_{}'.format(mstr)] = y_pred
        val_df['pred_{}'.format(mstr)] = np.where(y_test == 0, np.nan, y_pred)
        d = y_pred - y_test
        d[np.abs(d) > 1] = np.nan
        val_df['diff_{}'.format(mstr)] = np.where(y_test == 0, np.nan, d)
        print('\n month {}'.format(mstr))
        print('observed ET: {:.3f} m'.format(y.mean()))
        print('rmse ET: {:.3f} mm'.format(rmse * 1000))
        print('rmse {:.3f} %'.format(rmse / y.mean() * 100.))

        if importance:
            vars = os.path.join(importance, 'variables_{}_{}.json'.format(glob, m))
            with open(vars, 'r') as fp:
                d = json.load(fp)
            features = [f[0] for f in d[str(m)][:n_feats]]
            x = df[features].values
            x_test = val[features].values
            rf = RandomForest()
            rf.fit(x, y)
            y_pred = rf.predict(x_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            print('sel observed ET: {:.3f} m'.format(y.mean()))
            print('sel rmse ET: {:.3f} mm'.format(rmse * 1000))
            print('sel rmse {:.3f} %'.format(rmse / y.mean() * 100.))

    et_pred = ['pred_{}'.format(m) for m in range(4, 11)]
    pred_gs = val_df[et_pred].sum(axis=1)
    label_gs = val_df[et_cols].sum(axis=1)
    season_rmse = mean_squared_error(label_gs, pred_gs, squared=False)
    print('\nmean predicted ET: {:.3f} m'.format(pred_gs.mean()))
    print('mean observed ET: {:.3f} m'.format(label_gs.mean()))
    print('mean difference ET: {:.3f} m'.format((pred_gs - label_gs).mean()))
    print('seasonal rmse ET: {:.3f} m'.format(season_rmse))

    print('predicted {} targets: '.format(len(targets)))
    print(targets, '\n')
    print('predicted on {} features: '.format(len(features)))
    print(features, '\n')

    if out_shape:
        gdf = GeoDataFrame(val_df, geometry=geo, crs='EPSG:4326')
        gdf.to_file(out_shape)

    return val_df


def test_covariate_shift(csv_dir, inference_dir):
    mdf = pd.DataFrame()
    for year in range(2010, 2022):
        train = os.path.join(csv_dir, 'bands_29DEC2022_{}.csv'.format(year))
        inference = os.path.join(inference_dir, 'bands_irr_29DEC2022_{}.csv'.format(year))
        df = read_csv(train, engine='python').sample(frac=1.0).reset_index(drop=True)
        idf = read_csv(inference, engine='python', index_col='OPENET_ID').sample(frac=1.0)
        idf_ppt = idf['ppt_wy_et'].values * 0.001
        df_ppt = df['ppt_wy_et'].values * 0.001
        et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]

        labels, _dirs, _indirs = [], [], []
        for etc in et_cols:
            df[etc] = df[etc].values.astype(float) * 0.00001
            idf[etc] = idf[etc].values.astype(float) * 0.00001

        df['season'] = df[et_cols].sum(axis=1)
        print(df.shape)
        df = df[df['season'] < df['ppt_wy_et'] * 0.001]
        df.drop(columns=['season'], inplace=True)
        targets, features, first = [], None, True

        for m in range(4, 11):
            mstr = str(m)

            if first:
                geo = idf.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
                val_df = GeoDataFrame(index=idf.index, geometry=geo)
                if mdf.empty:
                    mdf = GeoDataFrame(index=idf.index, geometry=geo)
                first = False

            target = 'et_{}_{}'.format(year, m)
            df.dropna(axis=0, inplace=True)
            y = df[target].values
            x = df[PROPS].values
            targets.append(target)
            rf = RandomForest()
            rf.fit(x, y)

            # preict ET directly, estimate ppt:ept
            irr_label = idf[target].values
            irr_x = idf[PROPS].values
            irr_test_direct = rf.predict(irr_x)

            # preict ET indirectly, model ppt:ept
            rf = RandomForest()
            y = df[target] / df_ppt
            rf.fit(x, y)
            irr_test_indirect = rf.predict(irr_x)

            labels.append('label_{}'.format(mstr))
            _dirs.append('dir_{}'.format(mstr))
            _indirs.append('indir_{}'.format(mstr))
            val_df[labels[-1]] = irr_label / idf_ppt
            val_df[_dirs[-1]] = irr_test_direct / idf_ppt
            val_df[_indirs[-1]] = irr_test_indirect
            print('{} label: {:.3f}, dir: {:.3f}, indir: {:.3f}'.format(m, val_df[labels[-1]].mean(),
                                                                        val_df[_dirs[-1]].mean(),
                                                                        val_df[_indirs[-1]].mean()))

        mdf.loc[val_df.index, 'geometry'] = geo
        mdf.loc[val_df.index, 'label_{}'.format(year)] = val_df[labels].sum(axis=1)
        mdf.loc[val_df.index, 'dir_{}'.format(year)] = val_df[_dirs].sum(axis=1)
        mdf.loc[val_df.index, 'indir_{}'.format(year)] = val_df[_indirs].sum(axis=1)

    gdf = GeoDataFrame(val_df, geometry=geo, crs='EPSG:4326')
    _file = os.path.join(inference_dir, 'infer_irr_29DEC2022.shp')
    gdf.to_file(_file)


def find_rf_variable_importance(csv_dir, glob=None, importance_json=None):
    years_ = [x for x in range(2000, 2021)]
    years_.reverse()

    for i, m in enumerate(range(5, 11)):
        print('\nmonth {}'.format(m))
        master, first = {}, True
        for year in years_:
            print(year)
            csv = os.path.join(csv_dir, '{}_{}.csv'.format(glob, year))
            df = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
            et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]
            for etc in et_cols:
                df[etc] = df[etc].values.astype(float) * 0.00001
            for iter_ in range(2):
                d, _, l, _ = train_test_split(df[PROPS].values, df[et_cols[i]].values, train_size=0.67)
                print('model iteration {}'.format(iter_))
                rf = RandomForest()
                rf.fit(d, l)
                _list = [(f, v) for f, v in zip(PROPS, rf.feature_importances)]
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
        d = {m: master}
        jsn = os.path.join(importance_json, 'variables_{}_{}.json'.format(glob, m))
        with open(jsn, 'w') as fp:
            fp.write(json.dumps(d, indent=4, sort_keys=True))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    tprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    iprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_irr_29DEC2022')
    # test_covariate_shift(tprepped, iprepped)

    imp_js = os.path.join(root, 'expansion', 'analysis', 'importance')
    find_rf_variable_importance(tprepped, glob='bands_29DEC2022', importance_json=imp_js)

    # model_data(tprepped, importance=imp_js, glob='bands_29DEC2022', n_feats=20)
# ========================= EOF ====================================================================
