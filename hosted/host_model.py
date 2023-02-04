import os
import json
from subprocess import CalledProcessError
from subprocess import Popen, PIPE, check_call
from calendar import monthrange
from datetime import date
import pickle as pkl

import ee
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_squared_error

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

from call_ee import stack_bands, PROPS

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

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
        self.mean_ = None
        self.std_ = None
        self.unordered_index = None
        self.indices = None
        self.normalizer = None
        self.model_dir = _dir
        self.label = label
        self.model = None
        self.model_name = None
        self.month = month
        self.target = 'et_{}'.format(month)
        self.norm_weight_file = os.path.join(self.model_dir, 'norm_weights.npy')

        if _dir and label:
            self.ee_dir = os.path.join(EEIFIED_DIR, self.label)
            self.feature_names = PROPS
            self.all_cols = self.feature_names + [self.label]
            columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in self.all_cols]
            self.features_dict = dict(zip(self.all_cols, columns))

    def prepare(self, n_inputs):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((None, None, n_inputs,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Conv2D(64, (1, 1), activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
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
            tf.keras.layers.Dense(1, activation='linear'),
            # tf.keras.layers.Conv2D(1, (1, 1), activation='linear')
        ])
        model.compile(loss=self.custom_loss,
                      optimizer=tf.keras.optimizers.Adam(0.001))
        self.model = model

    def custom_loss(self, y, pred):
        error = 100 * (pred - y)
        condition = tf.greater(error, 0)
        overestimation_loss = 3 * tf.square(error)
        underestimation_loss = 1 * tf.square(error)
        cost = tf.reduce_mean(tf.where(condition, overestimation_loss, underestimation_loss))
        return cost

    def normalize_fixed(self, x):
        x_normed = (x - self.mean_) / self.std_
        return x_normed

    def parse_tfrecord(self, example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, self.features_dict)
        labels = parsed_features.pop(self.target)
        return parsed_features, tf.cast(labels, tf.float32)

    def to_tuple(self, inputs, label):
        lb1 = tf.expand_dims(label, 1)
        lb1 = tf.expand_dims(lb1, 1)
        feats = tf.expand_dims(tf.transpose(list(inputs.values())), 1)
        # feats = self.normalize_fixed(feats)
        return feats, lb1

    def calc_mean_std(self, data_dir):

        training_data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        dataset = tf.data.TFRecordDataset(training_data, compression_type='GZIP')
        parsed_training = dataset.map(self.parse_tfrecord, num_parallel_calls=5)

        input_dataset = parsed_training.map(self.to_tuple)
        input_dataset = input_dataset.shuffle(1000)
        mean_, std_, M2 = 0, 0, 0

        for e, (f, l) in enumerate(input_dataset):
            f = np.array(f).squeeze()
            delta = f - mean_
            mean_ = mean_ + delta / (e + 1)
            delta2 = f - mean_
            M2 = M2 + delta * delta2
            std_ = np.sqrt(M2 / (e + 1))

        pkl_name = os.path.join(self.model_dir, 'meanstd.pkl')
        with open(pkl_name, 'wb') as handle:
            pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)

    def get_norm(self):

        norm = pkl.load(open(os.path.join(self.model_dir, 'meanstd.pkl'), 'rb'))
        return norm

    @staticmethod
    def get_available_features(self, data_dir):
        training_data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        dataset = tf.data.TFRecordDataset(training_data, compression_type='GZIP')
        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)

    def train(self, batch, data_dir, save_test_data=None):

        if not os.path.exists(os.path.join(self.model_dir, 'meanstd.pkl')):
            self.calc_mean_std(data_dir)

        norm = self.get_norm()
        self.mean_ = norm[0]
        self.std_ = norm[1]

        training_data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        dataset = tf.data.TFRecordDataset(training_data, compression_type='GZIP')

        split = 5
        train_dataset = dataset.window(split, split + 1).flat_map(lambda ds: ds)
        parsed_training = train_dataset.map(self.parse_tfrecord, num_parallel_calls=5)

        samp = iter(parsed_training).next()
        input_dataset = parsed_training.map(self.to_tuple)
        input_dataset = input_dataset.shuffle(1000).batch(batch)

        test_dataset = dataset.skip(split).window(1, split + 1).flat_map(lambda ds: ds)
        parsed_test = test_dataset.map(self.parse_tfrecord, num_parallel_calls=5)

        input_test = parsed_test.map(self.to_tuple)
        input_test = input_test.batch(batch)

        self.prepare(len(self.feature_names))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.00001, cooldown=3)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                                       restore_best_weights=True, min_delta=0.00001)
        self.model.fit(x=input_dataset, verbose=1, epochs=10, validation_data=input_test,
                       validation_freq=1, callbacks=[reduce_lr, early_stopping])
        self.model.get_config()

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

    def infer_local(self, in_rasters, out_rasters, calc_diff=False, overwrite=False):

        rasters = [os.path.join(in_rasters, x) for x in os.listdir(in_rasters) if x.endswith('.tif')]
        rasters = sorted(rasters)
        rasters.reverse()

        p = Popen([RINFO, 'info', rasters[0]], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = p.communicate()
        raster_info = out[0].decode('ascii').splitlines()
        raster_info = json.loads(raster_info[0])

        self.unordered_index = [raster_info['descriptions'].index(p) for p in self.feature_names]

        for r in rasters[:1]:
            base = os.path.basename(r)
            out_raster_file = os.path.join(out_rasters, base.replace('.', '_{}.'.format(self.label)))
            if os.path.exists(out_raster_file) and not overwrite:
                print('{} exists, skipping'.format(os.path.basename(out_raster_file)))
                continue
            self._write_inference(r, out_raster_file)
            if calc_diff:
                ind = raster_info['descriptions'].index(self.target)
                diff_file = out_raster_file.replace('.tif', '_diff.tif')
                with rasterio.open(out_raster_file, 'r') as src:
                    pred = src.read()
                    meta = src.meta
                with rasterio.open(r, 'r') as src:
                    obs = src.read(ind)
                diff = pred - obs
                with rasterio.open(diff_file, 'w', **meta) as dst:
                    dst.write(diff)
                print(diff_file)

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
                    ds = self._parse_image(block_array)
                    ds = tf.data.Dataset.from_tensor_slices([ds])
                    result = self.model.predict(ds, verbose=0).reshape(shp)
                    dst.write(result, window=window)
        print(outras)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_dir, compile=False)


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

    root = '/media/nvm/ept'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/ept'

    tf.keras.utils.set_random_seed(1234)
    models = '/media/nvm/ept/ept_model_zoran_3FEB2023'

    data = os.path.join(root, 'tfr')
    VERSION_NAME = 'v00'

    for m in range(4, 11):
        t = 'et_{}'.format(m)
        model_dir = os.path.join(models, t)
        nn = DNN(month=m, label=t, _dir=model_dir)
        nn.model_name = MODEL_NAME.format(m)
        # nn.train(2560, data)
        # nn.save()
        nn.load()
        nn.infer_local(os.path.join(root, 'full_stack'),
                       os.path.join(root, 'full_stack_pred'),
                       calc_diff=False,
                       overwrite=True)

    # m = 'gs'
    # t = 'et_{}'.format(m)
    # model_dir = os.path.join(model_dir, t)
    # nn = DNN(month=m, label=t, _dir=model_dir)
    # nn.model_name = MODEL_NAME.format(m)
    # nn.train(2560, data)
    # nn.save()
# ========================= EOF ====================================================================
