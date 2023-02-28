import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from field_points.itype_mapping import itype_integer_mapping

np.random.seed(1234)


def partition_usbr_response(npy, out_fig):
    files = [os.path.join(npy, x) for x in os.listdir(npy) if x.endswith('.npy')]
    timescales = list(set([os.path.basename(n).strip('.npy') for n in files]))
    timescales.sort()

    for ts, f in zip(timescales, files):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(ts, rec.shape[1])
        rec = rec[:, np.random.randint(0, rec.shape[1], int(1e6))]
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'Management'])
        df['Management'] = df['Management'].apply(lambda x: 'Reclamation' if x > 0 else 'Non-Federal')

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
        sns.violinplot(data=df, x='Class', y='SIMI', hue='Management', order=['Dry', 'Normal', 'Wet'])
        plt.suptitle('Western Irrigation Management {}'.format(ts))
        ofig = os.path.join(out_fig, ts)
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

    for ts, f in zip(timescales, files):
        rec = np.fromfile(f, dtype=float).reshape((4, -1))
        print(rec.shape[1])
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'itype'])
        df = df[df['itype'] > 0]
        df['itype'] = df['itype'].values.astype(int)
        df['Infrastructure'] = df['itype'].apply(lambda x: itype[x])

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
        sns.violinplot(data=df, x='Class', y='SIMI', hue='Infrastructure',
                       order=['Dry', 'Normal', 'Wet'])

        plt.suptitle('Irrigation Management and Irrigation Type {}'.format(ts))
        ofig = os.path.join(out_fig, '{}.png'.format(ts))
        plt.legend(ncol=2)
        plt.savefig(ofig)
        plt.close()
        print(ofig)


def partition_time_response(npy, out_fig):
    timescales = list(set(['_'.join(n.split('_')[:-1]) for n in os.listdir(npy)]))
    timescales.sort()
    for ts in timescales:
        recfile = [os.path.join(npy, n) for n in os.listdir(npy) if 'early' in n and ts in n][0]
        nonrecfile = [os.path.join(npy, n) for n in os.listdir(npy) if 'late' not in n and ts in n][0]

        rec = np.fromfile(recfile, dtype=float).reshape((4, -1))
        print(rec.shape[1])
        rec = rec[:, np.random.randint(0, rec.shape[1], int(1e6))]
        df = pd.DataFrame(data=rec.T, columns=['SIMI', 'SPEI', 'month', 'Time'])
        df['Time'] = ['1987-1996' for _ in range(df.shape[0])]

        nrec = np.fromfile(nonrecfile, dtype=float).reshape((4, -1))
        nrec = nrec[:, np.random.randint(0, nrec.shape[1], int(1e6))]
        ndf = pd.DataFrame(data=nrec.T, columns=['SIMI', 'SPEI', 'month', 'Time'])
        ndf['Time'] = ['2011-2021' for _ in range(ndf.shape[0])]

        df = pd.concat([df, ndf], ignore_index=True)
        df.loc[df['SPEI'] < 0, 'Class'] = 'Normal'
        df.loc[df['SPEI'] < -1.3, 'Class'] = 'Dry'
        df.loc[df['SPEI'] >= 0, 'Class'] = 'Wet'

        for clss in ['Normal', 'Dry', 'Wet']:
            for itp in ['1987-1996', '2011-2021']:
                v = df.loc[(df['Class'] == clss) & (df['Time'] == itp), 'SIMI'].values
                v = v[~np.isnan(v)]
                q75, q25 = np.percentile(v, [75, 25])
                iqr = q75 - q25
                print('{} {}: {:.3f}, {:.3f}, {} values'.format(clss, itp, np.median(v), iqr, len(v)))

        sns.violinplot(data=df, x='Class', y='SIMI', hue='Time',
                       order=['Dry', 'Normal', 'Wet'])
        plt.suptitle('Western Irrigation Management')
        ofig = os.path.join(out_fig, ts)
        plt.legend(ncol=2)
        plt.savefig(ofig)
        plt.close()
        print(ofig, '\n')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    param = 'usbr'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    # partition_usbr_response(part_, out_)

    param = 'itype'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    # partition_itype_response(part_, out_)

    param = 'time'
    part_ = '/media/nvm/field_pts/fields_data/partitioned_npy/{}'.format(param)
    out_ = os.path.join(root, 'figures', 'partitions', '{}'.format(param))
    partition_time_response(part_, out_)
# ========================= EOF ====================================================================
