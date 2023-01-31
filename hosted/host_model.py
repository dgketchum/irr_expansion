import os
import json
import time
from subprocess import CalledProcessError
from subprocess import Popen, PIPE, check_call
from calendar import monthrange
from datetime import date

import ee
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_squared_error

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

from googleapiclient import discovery
from googleapiclient import errors

from call_ee import stack_bands, BASIN_STATES, BOUNDARIES
from assets import list_bucket_files

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
RINFO = os.path.join(conda, 'irrimp', 'bin', 'rio')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'
GCLOUD = '/home/dgketchum/google-cloud-sdk/bin/gcloud'

PROJECT = 'ssebop-montana'

MODEL_NAME = 'ept_model_{}'

DATA_BUCKET = 'wudr'
OUTPUT_BUCKET = 'wudr'
REGION = 'us-central1'
MODEL_DIR = '/media/nvm/ept/ept_models'
EEIFIED_DIR = 'gs://wudr/ept_model_eeified'
VERSION_SCALE = 1000


class DNN:
    def __init__(self, month=None, label=None, _dir=None):

        '''
        Class to train and deploy to Google AI Platform a deep neural network. The model has to start and end
        with Conv2D.
        :param month: Month to target, 'gs' for growing season, or months 4 - 10.
        :param label: String target
        :param _dir: GCS Bucket where the model is heading
        '''
        self.unordered_index = None
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
            self.feature_names = PROPS
            self.all_cols = self.feature_names + [self.label]
            columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in self.all_cols]
            self.features_dict = dict(zip(self.all_cols, columns))

    def prepare(self, n_inputs):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((None, None, n_inputs,)),
            tf.keras.layers.Conv2D(64, (1, 1), activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Conv2D(1, (1, 1), activation='linear')
        ])
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.01))
        self.model = model

    def parse_tfrecord(self, example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, self.features_dict)
        labels = parsed_features.pop(self.target)
        return parsed_features, tf.cast(labels, tf.float32)

    @staticmethod
    def to_tuple(inputs, label):
        lb1 = tf.expand_dims(label, 1)
        lb1 = tf.expand_dims(lb1, 1)
        feats = tf.expand_dims(tf.transpose(list(inputs.values())), 1)
        return feats, lb1

    def train(self, batch, data_dir, save_test_data=None):

        training_data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        dataset = tf.data.TFRecordDataset(training_data, compression_type='GZIP')

        split = 5
        train_dataset = dataset.window(split, split + 1).flat_map(lambda ds: ds)
        parsed_training = train_dataset.map(self.parse_tfrecord, num_parallel_calls=5)

        input_dataset = parsed_training.map(self.to_tuple)
        input_dataset = input_dataset.shuffle(1000).batch(batch)
        samp = iter(input_dataset).next()

        test_dataset = dataset.skip(split).window(1, split + 1).flat_map(lambda ds: ds)
        parsed_test = test_dataset.map(self.parse_tfrecord, num_parallel_calls=5)

        input_test = parsed_test.map(self.to_tuple)
        input_test = input_test.batch(batch)

        self.prepare(len(self.feature_names))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.001, cooldown=3)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
        self.model.fit(x=input_dataset, verbose=1, epochs=1000, validation_data=input_test,
                       validation_freq=1, callbacks=[reduce_lr, early_stopping])

        y_pred = self.model.predict(x=input_test)[:, 0, 0, :]
        x_test = np.concatenate([x for x, y in input_test], axis=0)[:, 0, 0, :]
        y_test = np.concatenate([y for x, y in input_test], axis=0)[:, 0, 0, :]

        if save_test_data:
            df = pd.DataFrame(columns=['y_test', 'y_pred'], data=np.array([y_test, y_pred])[:, :, 0].T)
            df[self.feature_names] = x_test
            df.to_csv(save_test_data, float_format='%.4f')

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('target {}'.format(self.target))
        print('observed ET: {:.3f} mm'.format(y_test.mean()))
        print('predicted ET mean: {:.3f} mm'.format(y_pred.mean()))
        print('predicted ET min: {:.3f} mm'.format(y_pred.min()))
        print('predicted ET max: {:.3f} mm'.format(y_pred.max()))
        print('predicted ET std: {:.3f} mm'.format(y_pred.std()))

        print('rmse ET: {:.3f} mm'.format(rmse))
        print('rmse {:.3f} %'.format(rmse / np.array(y_test).mean() * 100.))
        print('trained on {}\n'.format(self.feature_names))

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

        input_dict = json.dumps({input_name: 'array'})
        output_dict = json.dumps({output_name: 'prediction'})

        print('DEPLOYING {}, {}'.format(self.model_name, VERSION_NAME))

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
        except CalledProcessError:
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

    def infer_ee(self, asset_root, roi, yr, month, props):
        ee.Initialize()
        desc = 'eff_ppt_{}_{}'.format(yr, month)
        target_str = 'et_{}'.format(month)

        roi = ee.FeatureCollection(roi)
        image = stack_bands(yr, roi, VERSION_SCALE)

        if 'et_gs' in props:
            ind = props.index('et_gs')
            props.pop(ind)

        image = image.select(props)
        array_image = image.float().toArray()

        model = ee.Model.fromAiPlatformPredictor(
            projectName=PROJECT,
            modelName=self.model_name,
            version=VERSION_NAME,
            inputTileSize=[8, 8],
            proj=ee.Projection('EPSG:4326').atScale(VERSION_SCALE),
            fixInputProj=True,
            region=REGION,
            inputShapes={'array': [len(props)]},
            outputBands={'output': {
                'type': ee.PixelType.float(),
                'dimensions': 1}})

        classified_img = model.predictImage(array_image)
        if month == 'gs':
            indx = ee.Date('{}-04-01'.format(yr)).format('YYYYMMdd')
            ts = ee.Date('{}-04-01'.format(yr)).millis()
            te = ee.Date('{}-10-31'.format(yr)).millis()
        else:
            indx = ee.Date('{}-{}-01'.format(yr, month)).format('YYYYMMdd')
            ts = ee.Date('{}-{}-01'.format(yr, month)).millis()
            te = ee.Date('{}-{}-{}'.format(yr, month, monthrange(yr, month)[1])).millis()

        classified_img = classified_img.set({
            'system:index': indx,
            'system:time_start': ts,
            'system:time_end': te,
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

    def infer_local(self, in_rasters, out_rasters):

        rasters = [os.path.join(in_rasters, x) for x in os.listdir(in_rasters)]

        p = Popen([RINFO, 'info', rasters[0]], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = p.communicate()
        s = out[0].decode('ascii').splitlines()
        raster_info = json.loads(s[0])

        self.unordered_index = [raster_info['descriptions'].index(p) for p in self.feature_names]

        for r in rasters:
            base = os.path.basename(r)
            out_raster_file = os.path.join(out_rasters, base.replace('.', '_{}.'.format(self.label)))
            self._write_inference(r, out_raster_file)

    def _parse_image(self, img):
        img = img[self.unordered_index, :, :]
        img = np.moveaxis(img, 0, 2)
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[-1]))
        img = img[:, np.newaxis, np.newaxis, :]
        return img

    def _write_inference(self, inras, outras):

        with rasterio.open(inras, 'r') as src:
            meta = src.meta
            meta['count'] = 1
            with rasterio.open(outras, 'w', **meta) as dst:
                for block_index, window in src.block_windows(1):
                    block_array = src.read(window=window)
                    shp = (1, block_array.shape[1], block_array.shape[2])
                    if not np.all(np.isnan(block_array)):
                        ds = self._parse_image(block_array)
                        ds = tf.data.Dataset.from_tensor_slices([ds])
                        result = self.model.predict(ds, verbose=0).reshape(shp)
                        result = np.where(np.isnan(result), np.ones_like(result) - 2, result)
                    else:
                        result = np.zeros_like(block_array)[:1, :, :] - 1
                    dst.write(result, window=window)
        print(outras)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_dir)


def remove_models(model_name, gc_project_id):
    project_id = 'projects/{}'.format(gc_project_id)
    model_id = '{}/models/{}'.format(project_id, model_name)

    ml = discovery.build('ml', 'v1')
    request = ml.projects().models().versions().list(parent=model_id)
    response = request.execute()

    if 'versions' in response.keys():
        versions = response['versions']
        while len(versions) >= 1:
            for version in response['versions']:
                request = ml.projects().models().versions().delete(name=version['name'])
                try:
                    request.execute()
                except errors.HttpError as err:
                    reason = err._get_reason()
                    if 'Cannot delete the default version' in reason:
                        next

            request = ml.projects().models().versions().list(parent=model_id)
            response = request.execute()
            time.sleep(1)
            try:
                versions = response['versions']
            except:
                break

    request = ml.projects().models().delete(name=model_id)

    while True:
        try:
            response = request.execute()
        except errors.HttpError as err:
            print('There was an error deleting the model.' +
                  ' Check the details:')
            reason = err._get_reason()
            print(reason)
            if 'A model with versions cannot be deleted' in reason:
                time.sleep(1)
                continue
        break


def request_band_extract(file_prefix, points_layer, region, years, clamp_et=False):
    ee.Initialize()
    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)

    for st in BASIN_STATES:
        if st not in ['CA', 'OR']:
            continue
        for yr in years:

            scale_feats = {'climate': 0.001, 'soil': 0.01, 'terrain': 0.001}
            stack = stack_bands(yr, roi, VERSION_SCALE, **scale_feats)

            state_bound = ee.FeatureCollection(os.path.join(BOUNDARIES, st))
            stack = stack.clip(state_bound)
            st_points = points.filterMetadata('STUSPS', 'equals', st)

            # st_points = ee.FeatureCollection(ee.Geometry.Point(-120.260594, 46.743666))

            plot_sample_regions = stack.sampleRegions(
                collection=st_points,
                scale=30,
                tileScale=16,
                projection=ee.Projection('EPSG:4326'))

            if clamp_et:
                plot_sample_regions = plot_sample_regions.filter(ee.Filter.lt('ratio', ee.Number(1.0)))

            # point = plot_sample_regions.first().getInfo()

            desc = '{}_{}_{}'.format(file_prefix, st, yr)
            task = ee.batch.Export.table.toCloudStorage(
                plot_sample_regions,
                description=desc,
                bucket='wudr',
                fileNamePrefix=desc,
                fileFormat='TFRecord')

            task.start()
            print(desc)


def export_inference_rasters():
    ee.Initialize()
    roi = ee.FeatureCollection('users/dgketchum/study_area_klamath')
    geo = roi.geometry()
    props = sorted(['elevation', 'slope', 'aspect', 'etr_gs', 'ppt_gs', 'ppt_wy_et'])

    for yr in range(2021, 2022):
        scale_feats = {'climate': 0.001, 'soil': 0.01, 'terrain': 0.001}
        stack = stack_bands(yr, roi, VERSION_SCALE, **scale_feats)
        stack = stack.select(props)

        desc = 'ept_image_full_stack_{}'.format(yr)
        task = ee.batch.Export.image.toCloudStorage(
            image=stack,
            description=desc,
            bucket='wudr',
            fileNamePrefix=desc,
            scale=1000,
            region=geo)

        task.start()
        print(desc)


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    clip = 'users/dgketchum/expansion/study_area_dissolve'
    asset_rt = 'users/dgketchum/expansion/eff_ppt_gcloud'

    tf.keras.utils.set_random_seed(1234)

    VERSION_NAME = 'v00'
    PROPS = sorted(['elevation', 'slope', 'aspect', 'etr_gs', 'ppt_gs', 'ppt_wy_et'])

    for m in range(4, 11):
        t = 'et_{}'.format(m)
        model_dir = os.path.join(MODEL_DIR, t)
        nn = DNN(month=m, label=t, _dir=model_dir)
        nn.model_name = MODEL_NAME.format(m)
        nn.train(2560, '/media/nvm/ept/tfr'.format(t))
        nn.save()
        # nn.load()
        nn.infer_local('/media/nvm/ept/small_stack',
                       '/media/nvm/ept/small_stack_pred')

    points_ = 'users/dgketchum/expansion/points/pts_30JAN2023'
    bucket = 'wudr'
    years_ = [x for x in range(1987, 2022)]
    clip = 'users/dgketchum/expansion/study_area_dissolve'
    # request_band_extract('bands_30JAN2023', points_, clip, years_, clamp_et=True)

    # export_inference_rasters()

# ========================= EOF ====================================================================
