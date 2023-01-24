import os
import json
from copy import deepcopy
from subprocess import check_call

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.tools import saved_model_utils

from call_ee import PROPS

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'

PROJECT = 'ssebop-montana'
MODEL_NAME = 'ept_model'
DATA_BUCKET = 'wudr'
OUTPUT_BUCKET = 'wudr'
REGION = 'us-central1'
MODEL_DIR = 'gs://wudr/ept_model'
EEIFIED_DIR = 'gs://wudr/ept_model_eeified'
MODEL_NAME = 'ept_dnn_model'
VERSION_NAME = 'v00'


class DNN:
    def __init__(self, x, label=None, _dir=None):
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(x)
        self.dnn_model = self.build_and_compile_model()
        self.model_dir = _dir
        self.label = label

        if _dir and label:
            self.ee_dir = os.path.join(EEIFIED_DIR, self.label)
            self.feature_names = list(PROPS)
            self.feature_names.append(self.label)
            columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in self.feature_names]
            self.features_dict = dict(zip(self.feature_names, columns))

    def build_and_compile_model(self):
        model = keras.Sequential([
            self.normalizer,
            layers.Dense(90, activation='relu'),
            layers.Dense(150, activation='relu'),
            layers.Dense(300, activation='relu'),
            layers.Dense(150, activation='relu'),
            layers.Dense(90, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def fit(self, x, y):
        self.dnn_model.fit(x, y, verbose=1, epochs=1, batch_size=2560)

    def predict(self, x_test):
        return self.dnn_model.predict(x_test).flatten()

    def save(self):
        self.dnn_model.save(self.model_dir, save_format='tf')

    def deploy(self):
        meta_graph_def = saved_model_utils.get_meta_graph_def(self.model_dir, 'serve')
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

        input_dict = json.dumps({input_name: "array"})
        output_dict = json.dumps({output_name: "output"})

        cmd = ['earthengine',
               'model',
               'prepare',
               '--source_dir',
               self.model_dir,
               '--dest_dir',
               self.ee_dir,
               '--input',
               input_dict,
               '--output',
               output_dict]

        check_call(cmd)

        cmd = ['gcloud',
               'ai-platform',
               'models',
               'create',
               MODEL_NAME,
               '--project',
               PROJECT,
               '--region',
               REGION]

        check_call(cmd)

        cmd = ['gcloud',
               'ai-platform',
               'versions',
               'create',
               VERSION_NAME,
               '--project',
               PROJECT,
               '--region',
               REGION,
               '--model',
               MODEL_DIR,
               '--origin',
               self.ee_dir,
               '--framework',
               '"TENSORFLOW"',
               '--runtime-version=2.3',
               '--python-version=3.7']

        check_call(cmd)


def train(csv, clamp_et=False):
    c = pd.read_csv(csv)
    t_et_cols = ['et_{}'.format(mm) for mm in range(4, 11)]
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

        model_dir = os.path.join(MODEL_DIR, target)
        nn = DNN(x, target, model_dir)
        nn.fit(x, y)
        y_pred = nn.predict(x_test)
        nn.save()
        nn.deploy()

        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print('\n month {}'.format(mstr))
        print('observed ET: {:.3f} m'.format(np.array(y).mean()))
        print('rmse ET: {:.3f} mm'.format(rmse * 1000))
        print('rmse {:.3f} %'.format(rmse / np.array(y).mean() * 100.))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    tprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    training = os.path.join(tprepped, 'all.csv')
    train(training)
# ========================= EOF ====================================================================
