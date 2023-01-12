import os
import sys
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
from scipy.stats import linregress
from geopandas import GeoDataFrame
from shapely.geometry import Point

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from tables import NLCD_UNCULT
from call_ee import BASIN_STATES

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

PROPS = ['aspect', 'elevation', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']


def study_wide_accuracy(dir_, glob, year, seed=1234):
    first = True
    for st in BASIN_STATES:
        ff = os.path.join(dir_, '{}_{}_{}.csv'.format(glob, st, year))
        if os.path.exists(ff):
            c = random_forest(ff, year=year, show_importance=False, clamp_et=True, out_shape=None, seed=seed)
            if first:
                df = deepcopy(c)
                first = False
            else:
                df = pd.concat([df, c], axis=0, ignore_index=True)

    et_cols = ['et_{}_{}'.format(2020, mm) for mm in range(4, 11)]
    et_pred = ['pred_{}'.format(m) for m in range(4, 11)]
    pred_gs = df[et_pred].sum(axis=1)
    label_gs = df[et_cols].sum(axis=1)
    season_rmse = mean_squared_error(label_gs, pred_gs, squared=False)
    print('========================   OVERALL   ==========================')
    print('mean ET', label_gs.mean())
    print('rmse ', season_rmse / label_gs.mean() * 100., '%')
    print('rmse', season_rmse)


class RandomForest:

    def __init__(self):
        self.rf = None

    def fit(self, x, y):
        self.rf = RandomForestRegressor(n_estimators=150,
                                        n_jobs=-1,
                                        bootstrap=True,
                                        oob_score=True,
                                        min_samples_leaf=5)
        self.rf.fit(x, y)

    def predict(self, x):
        return self.rf.predict(x)


class DNN:
    def __init__(self, x):
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(x)
        self.dnn_model = self.build_and_compile_model()

    def build_and_compile_model(self):
        model = keras.Sequential([
            self.normalizer,
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def fit(self, x, y):
        self.dnn_model.fit(x, y, verbose=0, epochs=200)

    def predict(self, x_test):
        return self.dnn_model.predict(x_test).flatten()


def model_data(csv, model='random_forest', out_shape=None, year=2020, out_fig=None, clamp_et=False):
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
    for m in range(4, 5):
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

        if model == 'random_forest':
            rf = RandomForest()
            rf.fit(x, y)
            y_pred = rf.predict(x_test)
        elif model == 'dnn':
            nn = DNN(x)
            nn.fit(x, y)
            y_pred = nn.predict(x_test)

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

    # et_pred = ['pred_{}'.format(m) for m in range(4, 11)]
    # pred_gs = val_df[et_pred].sum(axis=1)
    # label_gs = val_df[et_cols].sum(axis=1)
    # season_rmse = mean_squared_error(label_gs, pred_gs, squared=False)
    # print('\nmean predicted ET: {:.3f} m'.format(pred_gs.mean()))
    # print('mean observed ET: {:.3f} m'.format(label_gs.mean()))
    # print('mean difference ET: {:.3f} m'.format((pred_gs - label_gs).mean()))
    # print('seasonal rmse ET: {:.3f} m'.format(season_rmse))
    #
    # print('predicted {} targets: '.format(len(targets)))
    # print(targets, '\n')
    # print('predicted on {} features: '.format(len(features)))
    # print(features, '\n')

    if out_fig:
        plot_regressions(val_df, out_fig)
    if out_shape:
        gdf = GeoDataFrame(val_df, geometry=geo, crs='EPSG:4326')
        gdf.to_file(out_shape)

    return val_df


def plot_regressions(df, outfig):
    fig, ax = plt.subplots(2, 4, )
    fig.set_figheight(10)
    fig.set_figwidth(16)
    lr = linregress(df['pred_gs'], df['label_gs'])
    ax[0, 0].scatter(df['label_gs'], df['pred_gs'], s=15, marker='.', c='b')
    ax[0, 0].title.set_text('Growing Season Apr - Oct')
    ax[0, 0].set(xlabel='Observed EffPpt - Growing Season', ylabel='Predicted EffPpt - Growing Season')
    txt = AnchoredText('n={}\nr={:.3f}\nb={:.3f}'.format(df.shape[0], lr.rvalue ** 2, lr.slope), loc=4)
    ax[0, 0].add_artist(txt)

    months = list(range(4, 11))
    cols = [('label_{}'.format(m), 'pred_{}'.format(m)) for m in range(4, 11)]
    for e, ax_ in enumerate(ax.ravel()[1:]):
        p, l = df[cols[e][1]] * 0.00001, df[cols[e][0]] * 0.00001
        lr = linregress(p, l)
        mstr = date.strftime(date(1990, months[e], 1), '%B')
        ax_.scatter(l, p, s=15, marker='.', c='b')
        ax_.title.set_text('Effective Precipitation {} [m]'.format(mstr))
        ax_.set(xlabel='Observed EffPpt - {}'.format(mstr), ylabel='Predicted EffPpt - {}'.format(mstr))
        txt = AnchoredText('n={}\nr={:.3f}\nb={:.3f}'.format(df.shape[0], lr.rvalue ** 2, lr.slope), loc=4)
        ax_.add_artist(txt)

    plt.tight_layout()
    plt.savefig(outfig)


def write_histograms(csv, fig_dir):
    if not isinstance(csv, DataFrame):
        print('\n', csv)
        df = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        df = csv

    yr = int(os.path.basename(csv).split('.')[0][-4:])
    df['ppt'] = df[['ppt_{}_{}'.format(yr, m) for m in range(1, 10)] +
                   ['ppt_{}_{}'.format(yr - 1, m) for m in range(10, 13)]].sum(axis=1)
    df['etr'] = df[['etr_{}_{}'.format(yr, m) for m in range(1, 10)] +
                   ['etr_{}_{}'.format(yr - 1, m) for m in range(10, 13)]].sum(axis=1)
    et_cols = ['et_{}_{}'.format(yr, mm) for mm in range(4, 11)]
    df['season'] = df[et_cols].sum(axis=1) * 0.00001

    print(df.shape)

    try:
        alt = ['STUSPS', 'uncult']
        df.drop(columns=alt, inplace=True)
    except KeyError:
        alt = ['uncult']
        df.drop(columns=alt, inplace=True)

    df['nlcd_class'] = df['nlcd'].apply(lambda x: True if x in NLCD_UNCULT else False)
    df = df[df['nlcd_class']]
    df.drop(columns=['.geo', 'system:index', 'id'], inplace=True)

    cols = list(df.columns)
    for col in cols:
        arr = df[col].values
        if isinstance(arr[0], float):
            bins = 20
        else:
            continue
        frq, edges = np.histogram(arr, bins)
        plt.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        plt.title(col)
        _figfile = os.path.join(fig_dir, '{}.png'.format(col))
        plt.savefig(_figfile)
        plt.close()
        print(col)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    prepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    study_area = os.path.join(prepped, 'bands_29DEC2022_2020.csv')
    # model_data(study_area, 'random_forest', clamp_et=True)
    model_data(study_area, 'dnn', clamp_et=False)
# ========================= EOF ====================================================================
