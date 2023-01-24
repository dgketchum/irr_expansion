import os
import json
import subprocess
from pprint import pprint
from copy import deepcopy
from subprocess import check_call
from calendar import monthrange
from datetime import date

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.lib.io import file_io

from call_ee import PROPS, stack_bands

PROPS = ['elevation', 'slope', 'aspect', 'lat', 'lon']

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'
GCLOUD = '/home/dgketchum/google-cloud-sdk/bin/gcloud'

PROJECT = 'ssebop-montana'

MODEL_NAME = 'ept_model_m_{}'

DATA_BUCKET = 'wudr'
OUTPUT_BUCKET = 'wudr'
REGION = 'us-central1'
MODEL_DIR = 'gs://wudr/ept_model'
EEIFIED_DIR = 'gs://wudr/ept_model_eeified'
VERSION_NAME = 'v01'


class DNN:
    def __init__(self, label=None, _dir=None):
        self.normalizer = None
        self.model_dir = _dir
        self.label = label
        self.model = None
        self.model_name = None

        if _dir and label:
            self.ee_dir = os.path.join(EEIFIED_DIR, self.label)
            self.feature_names = list(PROPS)
            self.feature_names.append(self.label)
            columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in self.feature_names]
            self.features_dict = dict(zip(self.feature_names, columns))

    def prepare(self, x):
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(x)
        self.model = self._build_and_compile_model()

    def _build_and_compile_model(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((None, None, len(PROPS),)),
            tf.keras.layers.Conv2D(64, (1, 1), activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.softmax)
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def to_tuple(self, inputs, label):
        return (tf.expand_dims(tf.transpose(list(inputs.values())), 1),
                tf.expand_dims(tf.transpose(list(label.values())), 1))

    def _prep_csv(self, csv, month):
        c = pd.read_csv(csv)
        t_et_cols = ['et_{}'.format(mm) for mm in range(4, 11)]
        for etc in t_et_cols:
            c[etc] = c[etc].values.astype(float) * 0.00001

        return c



    def train(self, csv, month):
        c = self._prep_csv(csv, month)

        split = int(c.shape[0] * 0.8)

        targets, features, first = [], None, True
        df = deepcopy(c.iloc[:split, :])
        target = 'et_{}'.format(month)
        targets.append(target)

        df.dropna(axis=0, inplace=True)
        y = tf.convert_to_tensor(df[target].values.astype('float32'))
        df = df[PROPS]
        x = tf.convert_to_tensor(df.values.astype('float32'))
        print('training on {}'.format(df.shape[0]))

        val = deepcopy(c.iloc[split:, :])
        val.dropna(axis=0, inplace=True)
        y_test = tf.convert_to_tensor(val[target].values.astype('float32'))
        val = val[PROPS]
        x_test = tf.convert_to_tensor(val.values.astype('float32'))
        print('validating on {}'.format(val.shape[0]))

        self.prepare(x)
        self.model.fit(x, y, verbose=1, epochs=20, batch_size=2560)
        y_pred = self.model.predict(x_test).flatten()

        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mstr = str(m)
        print('\n month {}'.format(mstr))
        print('observed ET: {:.3f} m'.format(np.array(y).mean()))
        print('rmse ET: {:.3f} mm'.format(rmse * 1000))
        print('rmse {:.3f} %'.format(rmse / np.array(y).mean() * 100.))
        print('trained on {}'.format(PROPS))

    def save(self):
        self.model.save(self.model_dir, save_format='tf')

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
        output_dict = json.dumps({output_name: "prediction"})

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

        cmd = [GCLOUD,
               'ai-platform',
               'models',
               'create',
               self.model_name,
               '--project',
               PROJECT,
               '--region',
               REGION]
        try:
            check_call(cmd)
        except subprocess.CalledProcessError:
            pass

        cmd = [GCLOUD,
               'ai-platform',
               'versions',
               'create',
               VERSION_NAME,
               '--project',
               PROJECT,
               '--region',
               REGION,
               '--model',
               self.model_name,
               '--origin',
               self.ee_dir,
               '--runtime-version=2.3',
               '--python-version=3.7']

        check_call(cmd)

    def infer(self, asset_root, roi, scale, yr, month):

        ee.Initialize()
        desc = 'eff_ppt_{}_{}'.format(yr, month)
        target_str = 'et_{}'.format(month)

        roi = ee.FeatureCollection(roi)
        image = stack_bands(yr, roi, gridmet_res=True)

        image = image.select(PROPS)
        array_image = image.float().toArray()

        model = ee.Model.fromAiPlatformPredictor(
            projectName=PROJECT,
            modelName=MODEL_NAME,
            version=VERSION_NAME,
            inputTileSize=[8, 8],
            proj=ee.Projection('EPSG:4326').atScale(scale),
            fixInputProj=True,
            region=REGION,
            inputShapes={"array": [len(PROPS)]},
            outputBands={'output': {
                'type': ee.PixelType.float(),
                'dimensions': 1}})

        classified_img = model.predictImage(array_image)
        classified_img = classified_img.set({
            'system:index': ee.Date('{}-{}-01'.format(yr, month)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-{}-01'.format(yr, month)).millis(),
            'system:time_end': ee.Date('{}-{}-{}'.format(yr, month, monthrange(yr, month)[1])).millis(),
            'date_ingested': str(date.today()),
            'image_name': desc,
            'target': target_str})

        classified_img = classified_img.rename('eff_ppt')
        classified_img = classified_img.clip(roi.geometry())

        asset_id = os.path.join(asset_root, desc)

        task = ee.batch.Export.image.toAsset(
            image=classified_img,
            description=desc,
            assetId=asset_id,
            pyramidingPolicy={'.default': 'mean'},
            maxPixels=1e13)

        task.start()
        print(asset_id, target_str)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_dir)

    def predict_csv(self, csv, month):

        if not self.model:
            raise NotImplemented

        df = self._prep_csv(csv, month)
        target = 'et_{}'.format(month)

        df.dropna(axis=0, inplace=True)
        y = tf.convert_to_tensor(df[target].values.astype('float32'))
        df = df[PROPS]
        x = tf.convert_to_tensor(df.values.astype('float32'))
        y_pred = self.model.predict(x).flatten()
        rmse = mean_squared_error(y, y_pred, squared=False)

        print('\n month {}'.format(m))
        print('observed ET: {:.3f} m'.format(np.array(y).mean()))
        print('rmse ET: {:.3f} mm'.format(rmse * 1000))
        print('rmse {:.3f} %'.format(rmse / np.array(y).mean() * 100.))
        print('predicted on {}'.format(PROPS))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    tprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    training = os.path.join(tprepped, 'all.csv')

    clip = 'users/dgketchum/expansion/columbia_basin'
    asset_rt = ic = 'users/dgketchum/expansion/eff_ppt_gcloud'

    for m in range(4, 5):
        t = 'et_{}'.format(m)
        model_dir = os.path.join(MODEL_DIR, t)
        nn = DNN(label=t, _dir=model_dir)
        nn.model_name = MODEL_NAME.format(m)
        nn.train(training, m)
        # nn.save()
        # nn.load(model_dir)
        # nn.predict_csv(training, m)
        # nn.deploy()
        # scale_ = 1000
        # nn.infer(asset_rt, clip, scale_, 2020, m)

# ========================= EOF ====================================================================
