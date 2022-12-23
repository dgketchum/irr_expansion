import os
import csv
from pprint import pprint
from calendar import monthrange
from subprocess import check_call, Popen, PIPE

from call_ee import is_authorized

import ee

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'


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


def push_bands_to_asset(_dir, glob, bucket):
    shapes = []
    local_f = os.path.join(_dir, 'domain_{}.csv'.format(glob))
    bucket = os.path.join(bucket, 'expansion_bands')
    _file = os.path.join(bucket, 'domain_{}.csv'.format(glob))
    cmd = [GS, 'cp', local_f, _file]
    check_call(cmd)
    shapes.append(_file)
    asset_ids = [os.path.basename(shp).split('.')[0] for shp in shapes]
    ee_root = 'users/dgketchum/expansion/tables/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '-f', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


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


if __name__ == '__main__':
    is_authorized()
    _bucket = 'gs://wudr'
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    dp = 'users/dpendergraph/expansion/naturalized_et'
    kj = 'users/kelseyjencso/naturalized_et'
    dk = 'users/dgketchum/expansion/naturalized_et'
    check_assets(dk)
    # move_assets(kj, dk)

    ee_dir = 'users/dgketchum/expansion/tables'
    rt = '/home/dgketchum/Downloads/bands_'
    # for yr in range(1987, 2022):
    # asset_ = os.path.join(ee_dir, 'domain_{}'.format(yr))
    # delete_asset(asset_)
    # push_bands_to_asset(rt, glob=yr, bucket=_bucket)

# ========================= EOF ====================================================================
