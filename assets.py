import os
from subprocess import check_call

from call_ee import is_authorized

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'irrimp', 'bin', 'earthengine')
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'


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


if __name__ == '__main__':
    is_authorized()
    _bucket = 'gs://wudr'
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    ee_dir = 'users/dgketchum/expansion/tables'
    rt = '/home/dgketchum/Downloads/bands_'
    for yr in range(1987, 2022):
        # asset_ = os.path.join(ee_dir, 'domain_{}'.format(yr))
        # delete_asset(asset_)
        push_bands_to_asset(rt, glob=yr, bucket=_bucket)

# ========================= EOF ====================================================================
