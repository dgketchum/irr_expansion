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

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

from call_ee import PROPS, stack_bands
from assets import list_bucket_files

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
TRAINING_DATA = 'gs://wudr/tfr_29DEC2022'
EEIFIED_DIR = 'gs://wudr/ept_model_eeified'
VERSION_NAME = 'v00'

PROPS = ['elevation', 'slope', 'aspect', 'lat', 'lon', 'etr_gs', 'ppt_wy_et', 'et_gs']


# ET_COLS = ['et_{}'.format(mm) for mm in range(4, 11)]
# PROPS = PROPS + ['et_gs']


class DNN:
    def __init__(self, month=None, label=None, _dir=None):
        self.indices = None
        self.normalizer = None
        self.model_dir = _dir
        self.label = label
        self.model = None
        self.model_name = None
        self.month = month
        self.target = 'et_{}'.format(month)

        if _dir and label:
            self.ee_dir = os.path.join(EEIFIED_DIR, self.label)
            self.feature_names = list(PROPS)
            self.feature_names.append(self.label)
            columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in self.feature_names]
            self.features_dict = dict(zip(self.feature_names, columns))

    def prepare(self, n_inputs):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((None, None, n_inputs,)),
            tf.keras.layers.Conv2D(64, (1, 1), activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Conv2D(1, (1, 1), activation='linear')
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.01))
        self.model = model

    def parse_tfrecord(self, example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, self.features_dict)

        if self.target == 'et_gs':
            labels = parsed_features[self.target]
        else:
            labels = parsed_features.pop(self.target)

        return parsed_features, tf.cast(labels, tf.float32)

    def to_tuple(self, inputs, label):
        lb1 = tf.expand_dims(label, 1)
        lb1 = tf.expand_dims(lb1, 1)
        feats = tf.expand_dims(tf.transpose(list(inputs.values())), 1)
        feats = tf.gather(feats, self.indices, axis=-1)
        return feats, lb1

    @tf.function
    def _filtercond(self, x, _):
        return tf.less(x['et_gs'], x['ppt_wy_et'])

    def train(self, batch):

        # training_data = list_bucket_files(TRAINING_DATA)
        training_data = [os.path.join('/media/nvm/tfr/', x) for x in os.listdir('/media/nvm/tfr/')]
        dataset = tf.data.TFRecordDataset(training_data, compression_type='GZIP')

        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)

        split = 5
        train_dataset = dataset.window(split, split + 1).flat_map(lambda ds: ds)
        parsed_training = train_dataset.map(self.parse_tfrecord, num_parallel_calls=5)
        filtered = parsed_training.filter(lambda x, y: tf.reduce_all(x['et_gs'] < x['ppt_wy_et']))

        # remove 'et_gs' from training data
        filt = iter(filtered).next()
        keys = list(filt[0].keys())
        ind = keys.index('et_gs')
        self.indices = [x for x in range(len(keys)) if x != ind]
        keys.pop(ind)
        input_dataset = filtered.map(self.to_tuple)
        input_dataset = input_dataset.shuffle(128).batch(batch)

        test_dataset = dataset.skip(split).window(1, split + 1).flat_map(lambda ds: ds)
        parsed_test = test_dataset.map(self.parse_tfrecord, num_parallel_calls=5)
        filtered = parsed_test.filter(lambda x, y: tf.reduce_all(x['et_gs'] < x['ppt_wy_et']))
        input_test = filtered.map(self.to_tuple)
        input_test = input_test.batch(batch)

        self.prepare(len(keys))

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3, min_lr=0.001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='auto', restore_best_weights=True)
        self.model.fit(x=input_dataset, verbose=1, epochs=100, validation_split=0.,
                       validation_data=input_test, validation_freq=1, callbacks=[reduce_lr, early_stopping])

        y_pred = self.model.predict(x=input_test)[:, 0, 0, :]
        x_test = np.concatenate([x for x, y in input_test], axis=0)[:, 0, 0, :]
        y_test = np.concatenate([y for x, y in input_test], axis=0)[:, 0, 0, :]
        df = pd.DataFrame(columns=['y_test', 'y_pred'], data=np.array([y_test, y_pred])[:, :, 0].T)
        df[keys] = x_test
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('target {}'.format(self.target))
        print('observed ET: {:.3f} mm'.format(y_test.mean()))
        print('predicted ET: {:.3f} mm'.format(y_pred.mean()))
        print('rmse ET: {:.3f} mm'.format(rmse))
        print('rmse {:.3f} %'.format(rmse / np.array(y_test).mean() * 100.))
        print('trained on {}\n'.format(keys))

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


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    tprepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    training = os.path.join(tprepped, 'all.csv')

    clip = 'users/dgketchum/expansion/columbia_basin'
    asset_rt = ic = 'users/dgketchum/expansion/eff_ppt_gcloud'

    m = 'gs'
    t = 'et_{}'.format(m)
    model_dir = os.path.join(MODEL_DIR, t)
    nn = DNN(month=m, label=t, _dir=model_dir)
    nn.model_name = MODEL_NAME.format(m)
    nn.train(256)
    nn.save()
    nn.deploy()
    # scale_ = 1000
    # nn.infer(asset_rt, clip, scale_, 2020, m)

# ========================= EOF ====================================================================
