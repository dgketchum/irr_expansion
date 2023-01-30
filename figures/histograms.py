import os

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def compare_training_inference(train, infer, fig_dir, year=2020):
    df = pd.read_csv(train, engine='python').sample(frac=1.0).reset_index(drop=True)
    idf = pd.read_csv(infer, engine='python').sample(frac=1.0).reset_index(drop=True)

    et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]
    df['season'] = df[et_cols].sum(axis=1)

    cols = [c for c in list(df.columns) if c not in et_cols]
    for col in cols:
        fig, ax = plt.subplots()
        try:
            sns.kdeplot(df[col], ax=ax, label='train')
            sns.kdeplot(idf[col], ax=ax, label='infer')
        except KeyError:
            continue
        plt.legend()
        plt.title(col)
        _figfile = os.path.join(fig_dir, '{}.png'.format(col))
        plt.savefig(_figfile)
        plt.close()
        plt.clf()
        print(col)


def inference_histogram(csv, raster, raster_pred, figs):

    with rasterio.open('/home/dgketchum/Downloads/ept_30m_6_bands.tif') as src:
        img = src.read()
        img = np.moveaxis(img, 0, 2)
        img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img = np.expand_dims(img, axis=1)
        img = np.expand_dims(img, axis=1)

    ids = tf.data.Dataset.from_tensor_slices(img).batch(256)
    img = np.concatenate([y for y in ids], axis=0).squeeze().T

    with rasterio.open(raster_pred) as src:
        pred = src.read()
    pred = pred.flatten()

    df = pd.read_csv(csv)
    df = df.sample(n=img.shape[1])

    features = ['aspect', 'elevation', 'slope', 'ppt_wy_et', 'etr_gs', 'ppt_gs']

    for i, _name in enumerate(features):
        fig, ax = plt.subplots()
        col = df[_name].values
        ras = img[i, :]
        sns.kdeplot(ras, ax=ax, label='raster {}'.format(_name), c='red')
        sns.kdeplot(col, ax=ax, label='tabular {}'.format(_name), c='blue')
        _file = os.path.join(figs, '{}.png'.format(_name))
        plt.legend()
        plt.suptitle(_name)
        plt.savefig(_file)
        plt.clf()
        print(_file)

    fig, ax = plt.subplots()
    sns.kdeplot(img[3, :], ax=ax, label='raster_ppt_wy_et', c='blue')
    sns.kdeplot(pred, ax=ax, label='raster_pred', c='red')
    plt.suptitle('tabular_ppt_et_pred')
    _file = os.path.join(figs, 'tabular_ppt_et_pred.png')
    plt.legend()
    plt.savefig(_file)
    plt.clf()
    print(_file)

    fig, ax = plt.subplots()
    sns.kdeplot(df['ppt_wy_et'], ax=ax, label='ppt_wy_et', c='blue')
    sns.kdeplot(df['y_pred'], ax=ax, label='table_pred', c='red')
    plt.suptitle('tabular_ppt_et_pred')
    _file = os.path.join(figs, 'tabular_ppt_et_pred.png')
    plt.legend()
    plt.savefig(_file)
    plt.clf()
    print(_file)

    fig, ax = plt.subplots()
    sns.kdeplot(df['y_test'], ax=ax, label='table_label', c='green')
    sns.kdeplot(df['y_pred'], ax=ax, label='table_pred', c='blue')
    plt.suptitle('tabular_et_gs')
    _file = os.path.join(figs, 'tabular_et_gs.png')
    plt.legend()
    plt.savefig(_file)
    plt.clf()
    print(_file)

    fig, ax = plt.subplots()
    sns.kdeplot(pred, ax=ax, label='raster_pred', c='red')
    sns.kdeplot(df['y_pred'], ax=ax, label='table_pred', c='blue')
    plt.suptitle('raster_tabular_et_pred')
    plt.legend()
    _file = os.path.join(figs, 'raster_tabular_et_pred.png')
    plt.savefig(_file)
    plt.clf()
    print(_file)

    pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    prepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    training_area = os.path.join(prepped, 'bands_29DEC2022_2020.csv')

    prepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_irr_29DEC2022')
    inference_area = os.path.join(prepped, 'bands_irr_29DEC2022_2020.csv')

    fdir = '/media/research/IrrigationGIS/expansion/figures/train_infer_comparison'

    # compare_training_inference(training_area, inference_area, fdir)

    c = '/home/dgketchum/Downloads/ept_test_data.csv'
    r = '/home/dgketchum/Downloads/ept_features_6_bands.tif'
    oraster = '/home/dgketchum/Downloads/ept_pred.tif'
    out_d = '/media/research/IrrigationGIS/expansion/figures/debug'
    inference_histogram(c, r, oraster, out_d)

# ========================= EOF ====================================================================
