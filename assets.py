import os
import csv
from pprint import pprint
from datetime import datetime
from calendar import monthrange
from subprocess import check_call, Popen, PIPE

import ee

from call_ee import is_authorized

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'


def list_bucket_files(location):
    command = 'ls'
    cmd = ['{}'.format(GS), '{}'.format(command), '{}'.format(location)]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['name'])
    assets = [x['name'] for x in reader]
    return assets


def list_assets(location):
    command = 'ls'
    cmd = ['{}'.format(EE), '{}'.format(command), '{}'.format(location)]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['name'])
    assets = [x['name'] for x in reader]
    assets = [x for x in assets if 'Running' not in x]
    return assets


def delete_asset(ee_asset_path):
    command = 'rm'
    cmd = ['{}'.format(EE), '{}'.format(command), '{}'.format(ee_asset_path)]
    print(cmd)
    check_call(cmd)


def push_bands_to_asset(_dir, asset_folder, glob, bucket):
    asset_pref = 'projects/earthengine-legacy/assets'
    transferred, asset_ids = [], []
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    existing = list_assets(asset_folder)

    for local_f in l:
        base_ = os.path.basename(local_f)
        gcs_file = os.path.join(bucket, base_)
        cmd = [GS, 'cp', local_f, gcs_file]
        transferred.append(gcs_file)
        asset_id = os.path.join(asset_pref, asset_folder, base_.split('.')[0])
        if asset_id not in existing:
            check_call(cmd)
            asset_ids.append(asset_id)

    for gcs_file, asset_id in zip(transferred, asset_ids):
        cmd = [EE, 'upload', 'table', '-f', '--asset_id={}'.format(asset_id), gcs_file]
        check_call(cmd)


def push_points_to_asset(_dir, state, bucket):
    local_files = [os.path.join(_dir, '{}.{}'.format(state, ext)) for ext in
                   ['shp', 'prj', 'shx', 'dbf']]
    bucket = os.path.join(bucket, 'openet_field_centroids')
    bucket_files = [os.path.join(bucket, '{}.{}'.format(state, ext)) for ext in
                    ['shp', 'prj', 'shx', 'dbf']]
    for lf, bf in zip(local_files, bucket_files):
        cmd = [GS, 'cp', lf, bf]
        check_call(cmd)

    asset_id = os.path.basename(bucket_files[0]).split('.')[0]
    ee_dst = 'users/dgketchum/openet/field_centroids/{}'.format(asset_id)
    cmd = [EE, 'upload', 'table', '-f', '--asset_id={}'.format(ee_dst), bucket_files[0]]
    check_call(cmd)
    print(asset_id, bucket_files[0])


def push_images_to_asset(_dir, asset_folder, glob, bucket):
    asset_pref = 'projects/earthengine-legacy/assets'
    transferred, asset_ids = [], []
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    existing = list_assets(asset_folder)

    for local_f in l:
        base_ = os.path.basename(local_f)
        gcs_file = os.path.join(bucket, base_)
        cmd = [GS, 'cp', local_f, gcs_file]
        transferred.append(gcs_file)
        asset_id = os.path.join(asset_pref, asset_folder, base_.split('.')[0])
        if asset_id not in existing:
            check_call(cmd)
            asset_ids.append(asset_id)

    for gcs_file, asset_id in zip(transferred, asset_ids):
        cmd = [EE, 'upload', 'image', '-f', '--asset_id={}'.format(asset_id), gcs_file]
        check_call(cmd)


def copy_asset(ee_asset, dst):
    cmd = ['{}'.format(EE), 'cp', ee_asset, dst]
    check_call(cmd)


def move_assets(in_dir, out_dir):
    l = list_assets(in_dir)
    ol = list_assets(out_dir)
    ol = [os.path.basename(i) for i in ol]
    for i in l:
        bname = os.path.basename(i)
        o = os.path.join(out_dir, bname)
        if bname not in ol:
            copy_asset(i, o)
            print('moved', bname)
        else:
            print(bname, 'exists, skipping')


def check_assets(_dir, yr_start=1987, yr_end=2021, months=(4, 5, 6, 7, 8, 9, 10)):
    l = list_assets(_dir)
    prec_dir = 'projects/earthengine-legacy/assets'
    done, missing = [], []
    for yr in range(yr_start, yr_end + 1):
        for m in months:
            bname = 'et_{}_{}'.format(yr, m)
            i = os.path.join(prec_dir, _dir, bname)
            if i in l:
                done.append(bname)
            else:
                missing.append(bname)
    pprint(missing)


def set_metadata(ee_asset, string):
    cmd = ['{}'.format(EE), 'asset', 'set',
           '-p', 'image_name={}'.format(string), ee_asset]
    print(' '.join(cmd))
    check_call(cmd)


def set_time_metadata(asset):
    splt = os.path.basename(asset).split('_')
    yr, month = int(splt[4]), int(splt[6])
    end_day = monthrange(yr, month)[1]
    cmd = ['{}'.format(EE), 'asset', 'set',
           '{}'.format('--time_start'), '{}-{}-01T00:00:00'.format(yr, str(month).rjust(2, '0')), asset]
    print(' '.join(cmd))
    check_call(cmd)

    cmd = ['{}'.format(EE), 'asset', 'set',
           '{}'.format('--time_end'), '{}-{}-{}T00:00:00'.format(yr, str(month).rjust(2, '0'),
                                                                 str(end_day)), asset]
    print(' '.join(cmd))
    check_call(cmd)


if __name__ == '__main__':
    is_authorized()
    _bucket = 'gs://wudr'
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    d = '/media/nvm/ept/full_stack_pred'
    af = 'users/dgketchum/expansion/eff_ppt'
    glob = 'ept_image_full_stack'
    bucket_ = 'gs://wudr'
    # push_images_to_asset(d, af, glob, bucket_)

    l = list_assets(af)
    for f in l:
        set_time_metadata(f)

# ========================= EOF ====================================================================
