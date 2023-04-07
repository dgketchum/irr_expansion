import os
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from field_points.crop_codes import cdl_key


def t_test_partion(part_dir):
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
        s2 = df.loc[(df['Class'] == 'Dry') & (df['cdl'] == crop), 'SIMI']
        t, p = ttest_ind(s1, s2)
        dct[crop] = {'t': t, 'p': p, 'crop': classes[crop]}

    pprint(dct)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'
    param = 'cdl'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    t_test_partion(part_)
# ========================= EOF ====================================================================
