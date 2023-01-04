import os
import sys
from copy import deepcopy
from datetime import date

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from geopandas import GeoDataFrame
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from tables import NLCD_UNCULT
from call_ee import BASIN_STATES

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

PROPS = ['aspect', 'elevation', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']


def random_forest(csv, n_estimators=50, out_shape=None, year=2020, out_fig=None,
                  show_importance=False, clamp_et=False):

    if not isinstance(csv, DataFrame):
        print('\n', csv)
        c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        c = csv

    et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]
    for etc in et_cols:
        c[etc] = c[etc] * 0.00001
    c['season'] = c[et_cols].sum(axis=1)
    print(c.shape)
    if clamp_et:
        c = c[c['season'] < c['ppt_wy_et'] * 0.001]
    print(c.shape)

    try:
        drop = ['uncult', 'season', 'STUSPS', '.geo', 'system:index', 'id', 'cdl', 'nlcd']
        c.drop(columns=drop, inplace=True)
    except KeyError:
        drop = ['.geo', 'system:index', 'STUSPS', 'id', 'season']
        c.drop(columns=drop, inplace=True)

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

        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=-1,
                                   bootstrap=True,
                                   random_state=123)

        rf.fit(x, y)

        if show_importance:
            _list = [(f, v) for f, v in zip(list(df.columns), rf.feature_importances_)]
            imp = sorted(_list, key=lambda x: x[1], reverse=True)
            print([f[0] for f in imp[:10]])

        y_pred = rf.predict(x_test)
        lr = linregress(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('{} r: {:.3f}, fractional: {:.3f}, rmse: {:.3f}\n'.format(mstr, lr.rvalue, rmse / y_test.mean(), rmse))

        val_df['label_{}'.format(mstr)] = y_test
        val_df['label_{}'.format(mstr)] = val_df['label_{}'.format(mstr)].apply(lambda x: x if x > 0 else np.nan)

        val_df['pred_{}'.format(mstr)] = y_pred
        val_df['pred_{}'.format(mstr)] = np.where(y_test == 0, np.nan, y_pred)

        d = y_pred - y_test
        d[np.abs(d) > 1] = np.nan
        val_df['diff_{}'.format(mstr)] = np.where(y_test == 0, np.nan, d)

    et_pred = ['pred_{}'.format(m) for m in range(4, 11)]
    pred_gs = val_df[et_pred].sum(axis=1)
    label_gs = val_df[et_cols].sum(axis=1)
    season_rmse = mean_squared_error(label_gs, pred_gs, squared=False)
    print('mean predicted ET: {:.3f} m'.format(pred_gs.mean()))
    print('mean observed ET: {:.3f} m'.format(label_gs.mean()))
    print('mean difference ET: {:.3f} m'.format((pred_gs - label_gs).mean()))
    print('seasonal rmse ET: {:.3f} m'.format(season_rmse))

    print('predicted {} targets: '.format(len(targets)))
    print(targets, '\n')
    print('predicted on {} features: '.format(len(features)))
    print(features, '\n')

    if out_fig:
        plot_regressions(val_df, out_fig)
    if out_shape:
        gdf = GeoDataFrame(val_df, geometry=geo, crs='EPSG:4326')
        gdf.to_file(out_shape)


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


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    extracts = os.path.join(root, 'expansion', 'tables', 'band_extracts')
    r = os.path.join(extracts, 'bands_29DEC2022')

    for st in BASIN_STATES:
        ff = os.path.join(r, 'bands_29DEC2022_{}_2020.csv'.format(st))
        if os.path.exists(ff):
            random_forest(ff, year=2020, show_importance=True, clamp_et=True, out_shape=None)
# ========================= EOF ====================================================================
