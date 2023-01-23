import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.tools import saved_model_utils

from call_ee import PROPS

PROJECT = 'ssebop-montana'
DATA_BUCKET = 'wudr'
OUTPUT_BUCKET = 'wudr'
REGION = 'us-central1'

LABEL = 'et_2020_10'

FEATURE_NAMES = list(PROPS)
FEATURE_NAMES.append(LABEL)

columns = [
    tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES
]
FEATURES_DICT = dict(zip(FEATURE_NAMES, columns))

MODEL_DIR = 'gs://wudr/ept_model'
EEIFIED_DIR = 'gs://wudr/ept_model_eeified'
MODEL_NAME = 'ept_dnn_model'
VERSION_NAME = 'v0'


class DNN:
    def __init__(self, x, _dir=None):
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(x)
        self.dnn_model = self.build_and_compile_model()
        self._dir = _dir

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
        self.dnn_model.fit(x, y, verbose=1, epochs=200, batch_size=256)

    def predict(self, x_test):
        return self.dnn_model.predict(x_test).flatten()

    def save(self):
        self.dnn_model.save(self._dir, save_format='tf')

    def eeify(self):
        meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_DIR, 'serve')
        inputs = meta_graph_def.signature_def['serving_default'].inputs
        outputs = meta_graph_def.signature_def['serving_default'].outputs
        input_name = None
        for k, v in inputs.items():
            input_name = v.name
            break

        output_name = None
        for k, v in outputs.items():
            output_name = v.name
            break

        input_dict = "'" + json.dumps({input_name: "array"}) + "'"
        output_dict = "'" + json.dumps({output_name: "output"}) + "'"
        print(input_dict)
        print(output_dict)


def train(csv_dir, glob='bands_29DEC2022', clamp_et=False):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y = int(splt[-1][:-4])
        print(y)
        if y == 2021:
            continue
        if first:
            c = pd.read_csv(csv)
            et_cols = ['et_{}_{}'.format(y, mm) for mm in range(4, 11)]
            t_et_cols = ['et_{}'.format(mm) for mm in range(4, 11)]
            c = c.rename(columns={k: v for k, v in zip(et_cols, t_et_cols)})
            first = False
        else:
            d = pd.read_csv(csv)
            et_cols = ['et_{}_{}'.format(y, mm) for mm in range(4, 11)]
            d = d.rename(columns={k: v for k, v in zip(et_cols, t_et_cols)})
            c = pd.concat([c, d])

    _file = os.path.join(csv_dir, 'all.csv')
    c.to_csv(_file)

    for etc in t_et_cols:
        c[etc] = c[etc].values.astype(float) * 0.00001
    c['season'] = c[t_et_cols].sum(axis=1)
    print(c.shape)
    if clamp_et:
        c = c[c['season'] < c['ppt_wy_et'] * 0.001]

    c.drop(columns=['season'], inplace=True)

    split = int(c.shape[0] * 0.8)

    targets, features, first = [], None, True
    for m in range(4, 11):
        df = deepcopy(c.iloc[:split, :])
        mstr = str(m)
        target = 'et_{}'.format(m)

        df.dropna(axis=0, inplace=True)
        y = tf.convert_to_tensor(df[target].values.astype('float32'))
        df.drop(columns=t_et_cols + ['STUSPS', 'id'], inplace=True)
        x = tf.convert_to_tensor(df.values.astype('float32'))
        targets.append(target)
        print('training on {}'.format(df.shape[0]))

        val = deepcopy(c.iloc[split:, :])
        val.dropna(axis=0, inplace=True)
        y_test = tf.convert_to_tensor(val[target].values.astype('float32'))
        val.drop(columns=t_et_cols + ['STUSPS', 'id'], inplace=True)
        x_test = tf.convert_to_tensor(val.values.astype('float32'))
        print('validating on {}'.format(val.shape[0]))

        nn = DNN(x, MODEL_DIR)
        nn.fit(x, y)
        y_pred = nn.predict(x_test)
        nn.save()
        nn.eeify()

        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print('\n month {}'.format(mstr))
        print('observed ET: {:.3f} m'.format(y.mean()))
        print('rmse ET: {:.3f} mm'.format(rmse * 1000))
        print('rmse {:.3f} %'.format(rmse / y.mean() * 100.))

        return nn


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    tprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    imp_js = os.path.join(root, 'expansion', 'analysis', 'importance')
    train(tprepped)
# ========================= EOF ====================================================================
