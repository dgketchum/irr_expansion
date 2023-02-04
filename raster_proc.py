import os
import json
from subprocess import check_call, Popen, PIPE

import numpy as np
import rasterio

WARP = '/home/dgketchum/miniconda3/envs/irrimp/bin/gdalwarp'
RINFO = '/home/dgketchum/miniconda3/envs/irrimp/bin/rio'


def get_raster_desc(tif_):
    p = Popen([RINFO, 'info', tif_], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out = p.communicate()
    raster_info = out[0].decode('ascii').splitlines()
    raster_info = json.loads(raster_info[0])
    return raster_info


def clip_raster_stack(in_dir, out_dir, clip_shp):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    bnames = [x for x in os.listdir(in_dir)]
    for f, b in zip(l, bnames):
        o = os.path.join(out_dir, b)

        cmd = [WARP, '-r', 'near', '-s_srs', 'EPSG:4326', '-t_srs', 'EPSG:4326', '-overwrite',
               '-cutline', clip_shp, '-crop_to_cutline', f, o]
        check_call(cmd)


def get_summaries(stack, prediction, year, out_dir):
    bands = ['et_{}'.format(m) for m in range(4, 11)]
    features = os.path.join(stack, 'ept_image_full_stack_{}.tif'.format(year))
    pred_images = [os.path.join(prediction, 'ept_image_full_stack_{}_{}.tif'.format(year, b)) for b in bands]
    info_ = get_raster_desc(features)
    zeros = np.zeros((len(bands), info_['height'], info_['width']))
    with rasterio.open(features, 'r') as src:
        img = zeros.copy()
        for i, b in enumerate(bands):
            ind = info_['descriptions'].index(b)
            print('read feature {}'.format(b))
            img[i, :, :] = src.read(ind)
    obs = img.sum(axis=0)
    img = zeros.copy()
    for i, r in enumerate(pred_images):
        with rasterio.open(r, 'r') as src:
            meta = src.meta
            print('read prediction {}'.format(os.path.basename(r)))
            img[i, :, :] = src.read()
    pred = img.sum(axis=0)

    out_img = os.path.join(out_dir, 'pred_obs_{}.tif'.format(year))
    meta['count'] = 3
    with rasterio.open(out_img, 'w', **meta) as dst:
        dst.write(obs, 1)
        dst.write(pred, 2)
        dst.write(pred - obs, 3)
    print(out_img)


if __name__ == '__main__':
    root = '/media/nvm/ept'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/ept'

    ind = os.path.join(root, 'full_stack')
    od = os.path.join(root, 'full_stack_or')
    cshp = '/media/research/IrrigationGIS/boundaries/states/OR_WGS.shp'
    # clip_raster_stack(ind, od, cshp)

    feature_stack = os.path.join(root, 'full_stack')
    prediction_ = os.path.join(root, 'full_stack_pred')
    out_images = os.path.join(root, 'pred_summaries')

    get_summaries(feature_stack, prediction_, 2021, out_images)
# ========================= EOF ====================================================================
