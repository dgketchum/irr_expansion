import os
import datetime
from calendar import monthrange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from field_points.itype_mapping import itype_integer_mapping
from field_points.crop_codes import cdl_key

np.random.seed(1234)
palette = sns.color_palette("rocket", as_cmap=True)

two_color = [palette.colors[i] for i in [50, 175]]
four_color = [palette.colors[i] for i in [30, 80, 130, 200]]


def partition_usbr_response(npy, out_fig):
    files = [os.path.join(npy, x) for x in os.listdir(npy) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()
    first = True
    for ts, f in zip(timescales[3:4], files[3:4]):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(ts, rec.shape[1])
        # rec = rec[:, np.random.randint(0, rec.shape[1], int(1e6))]
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'Management'])
        df['Management'] = df['Management'].apply(lambda x: 'Reclamation' if x > 0 else 'Non-Federal')
        if first:
            all_cts = np.unique(df['Management'].values, return_counts=True)
            print(all_cts)
            first = False

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'

        for clss in ['Normal', 'Dry', 'Wet']:
            for mng in ['Reclamation', 'Non-Federal']:
                v = df.loc[(df['Class'] == clss) & (df['Management'] == mng), 'SIMI'].values
                v = v[~np.isnan(v)]
                q75, q25 = np.percentile(v, [75, 25])
                iqr = q75 - q25
                print('{} {}: {:.3f}, {:.3f}, {} values'.format(clss, mng, np.median(v), iqr, len(v)))

        df.drop(columns=['month'], inplace=True)
        sns.violinplot(data=df, x='Class', y='SIMI', hue='Management', order=['Dry', 'Normal', 'Wet'],
                       palette=two_color)
        plt.axhline(y=0, linestyle='--', color='black', linewidth=1)
        plt.xlabel('Climate Classification')
        plt.ylabel('Standardized Irrigation Management Index')

        # plt.suptitle('Western Irrigation Management {}'.format(ts))
        ofig = os.path.join(out_fig, 'management_{}.png'.format(ts))
        plt.legend(ncol=2)
        plt.savefig(ofig)
        plt.close()
        print(ofig, '\n')


def partition_itype_response(npy, out_fig):
    files = [os.path.join(npy, x) for x in os.listdir(npy) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()

    itype = itype_integer_mapping()
    itype = {v: k for k, v in itype.items()}
    first = True

    for ts, f in zip(timescales[3:4], files[3:4]):

        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'itype'])
        df = df[df['itype'] > 0]
        df['itype'] = df['itype'].values.astype(int)
        df['Infrastructure'] = df['itype'].apply(lambda x: itype[x])

        if first:
            all_cts = np.unique(df['Infrastructure'].values, return_counts=True)
            print(all_cts)
            first = False

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'

        for clss in ['Normal', 'Dry', 'Wet']:
            for itp in range(1, 5):
                v = df.loc[(df['Class'] == clss) & (df['itype'] == itp), 'SIMI'].values
                v = v[~np.isnan(v)]
                q75, q25 = np.percentile(v, [75, 25])
                iqr = q75 - q25
                print('{} {}: {:.3f}, {:.3f}, {} values'.format(clss, itype[itp], np.median(v), iqr, len(v)))

        df.drop(columns=['itype', 'month'], inplace=True)
        sns.violinplot(data=df, x='Class', y='SIMI', hue='Infrastructure', order=['Dry', 'Normal', 'Wet'],
                       palette=four_color)
        plt.axhline(y=0, linestyle='--', color='black', linewidth=1)
        plt.xlabel('Climate Classification')
        plt.ylabel('Standardized Irrigation Management Index')

        ofig = os.path.join(out_fig, 'itype_{}.png'.format(ts))
        legend = plt.legend(ncol=2)
        plt.savefig(ofig)
        plt.close()
        print(ofig, '\n')


def partition_cdl_response(npy, out_fig):
    files = [os.path.join(npy, x) for x in os.listdir(npy) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()

    classes = ['Grain', 'Vegetable', 'Forage', 'Orchard', 'Uncultivated', 'Fallow']
    classes = {i + 1: classes[i] for i in range(len(classes))}
    cdl = cdl_key()
    first = True

    for ts, f in zip(timescales[3:4], files[3:4]):

        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'cdl'])
        df = df[df['cdl'] > 0]
        df['cdl'] = df['cdl'].values.astype(int)
        df['cdl'] = df['cdl'].apply(lambda x: cdl[x][1])
        df = df[df['cdl'] < 5]

        if first:
            all_cts = np.unique(df['cdl'].values, return_counts=True)
            print(all_cts)
            first = False

        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'

        for clss in ['Normal', 'Dry', 'Wet']:
            for itp in range(1, 5):
                v = df.loc[(df['Class'] == clss) & (df['cdl'] == itp), 'SIMI'].values
                v = v[~np.isnan(v)]
                q75, q25 = np.percentile(v, [75, 25])
                iqr = q75 - q25
                print('{} {}: {:.3f}, {:.3f}, {} values'.format(clss, classes[itp], np.median(v), iqr, len(v)))

        df['Crop'] = df['cdl'].apply(lambda x: classes[x])
        df.drop(columns=['cdl', 'month'], inplace=True)
        sns.violinplot(data=df, x='Class', y='SIMI', hue='Crop',
                       order=['Dry', 'Normal', 'Wet'], palette=four_color)

        plt.axhline(y=0, linestyle='--', color='black', linewidth=1)
        plt.xlabel('Climate Classification')
        plt.ylabel('Standardized Irrigation Management Index')

        ofig = os.path.join(out_fig, 'cdl_{}.png'.format(ts))
        legend = plt.legend(ncol=2)

        plt.savefig(ofig)
        plt.close()
        print(ofig, '\n')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    param = 'usbr'
    part_ = os.path.join(root, 'field_pts/partitioned_npy/{}'.format(param))
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    partition_usbr_response(part_, out_)

    param = 'cdl'
    part_ = os.path.join(root, 'field_pts/partitioned_npy/{}'.format(param))
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    partition_cdl_response(part_, out_)

    param = 'itype'
    part_ = os.path.join(root, 'field_pts/partitioned_npy/{}'.format(param))
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    # partition_itype_response(part_, out_)
# ========================= EOF ====================================================================
