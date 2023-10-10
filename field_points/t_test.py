import os
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from field_points.crop_codes import cdl_key
from field_points.itype_mapping import itype_integer_mapping


def t_test_crop_partion(part_dir):
    files = [os.path.join(part_dir, x) for x in os.listdir(part_dir) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()

    classes = ['Grain', 'Vegetable', 'Forage', 'Orchard', 'Uncultivated', 'Fallow']
    classes = {i + 1: classes[i] for i in range(len(classes))}
    cdl = cdl_key()

    for ts, f in zip(timescales[3:4], files[3:4]):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        # rec = rec[:, np.random.randint(0, rec.shape[1], 40000)]
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'cdl'])
        df = df[df['cdl'] > 0]
        df['cdl'] = df['cdl'].values.astype(int)
        df['cdl'] = df['cdl'].apply(lambda x: cdl[x][1])
        df = df[df['cdl'] < 5]

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'
        df.dropna(axis=0, inplace=True)

    dct = {}
    for crop in df['cdl'].unique():
        s1 = df.loc[(df['Class'] == 'Normal') & (df['cdl'] == crop), 'SIMI']
        s2 = df.loc[(df['Class'] == 'Wet') & (df['cdl'] == crop), 'SIMI']
        print(classes[crop], 'Normal', s1.mean(), 'Wet', s2.mean(), '{:.3f}'.format(s2.mean() - s1.mean()))
        t, p = ttest_ind(s1, s2)
        dct[crop] = {'t': t, 'p': p, 'crop': classes[crop]}

    pprint(dct)


def t_test_itype_partion(part_dir):
    files = [os.path.join(part_dir, x) for x in os.listdir(part_dir) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()
    itype = itype_integer_mapping()
    itype = {v: k for k, v in itype.items()}

    for ts, f in zip(timescales[3:4], files[3:4]):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        # rec = rec[:, np.random.randint(0, rec.shape[1], 40000)]
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'itype'])
        df = df[df['itype'] > 0]
        df['itype'] = df['itype'].values.astype(int)

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'
        df.dropna(axis=0, inplace=True)

    dct = {}
    for _type in df['itype'].unique():
        s1 = df.loc[(df['Class'] == 'Normal') & (df['itype'] == _type), 'SIMI']
        s2 = df.loc[(df['Class'] == 'Dry') & (df['itype'] == _type), 'SIMI']
        print(itype[_type], 'Normal', s1.mean(), 'Dry', s2.mean(), '{:.3f}'.format(s2.mean() - s1.mean()))
        t, p = ttest_ind(s1, s2)
        dct[_type] = {'t': t, 'p': p, 'itype': itype[_type]}

    pprint(dct)


def t_test_usbr_partion(part_dir):
    files = [os.path.join(part_dir, x) for x in os.listdir(part_dir) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()

    for ts, f in zip(timescales[3:4], files[3:4]):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'Management'])
        df['Management'] = df['Management'].apply(lambda x: 'Reclamation' if x > 0 else 'Non-Federal')
        print(np.unique(df['Management'], return_counts=True))

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'
        df.dropna(axis=0, inplace=True)

    dct = {}
    for cls_ in df['Class'].unique():

        s1 = df.loc[(df['Class'] == cls_) & (df['Management'] == 'Reclamation'), 'SIMI']
        s2 = df.loc[(df['Class'] == cls_) & (df['Management'] == 'Non-Federal'), 'SIMI']

        print(cls_, 'Reclamation', s1.mean(), 'Non-Federal',
              s2.mean(), '{:.3f}'.format(s1.mean() - s2.mean()))
        t, p = ttest_ind(s1, s2)
        dct[cls_] = {'t': t, 'p': p}

    pprint(dct)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    param = 'cdl'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    # t_test_crop_partion(part_)

    param = 'itype'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    # t_test_itype_partion(part_)

    param = 'usbr'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    # t_test_usbr_partion(part_)
# ========================= EOF ====================================================================
