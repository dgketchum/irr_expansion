import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TARGET_STATES = ['CA', 'WY']


def partition_response(csv, attrs, out_fig, month=10):

    first = True
    bnames = [x for x in os.listdir(csv) if x.strip('.csv') in TARGET_STATES]
    csv = [os.path.join(csv, x) for x in bnames]
    attrs = [os.path.join(attrs, x) for x in os.listdir(attrs) if x in bnames]
    for m, f in zip(attrs, csv):
        c = pd.read_csv(f, index_col=0)
        meta = pd.read_csv(m, index_col='OPENET_ID')
        c.loc[meta.index, 'usbrid'] = meta['usbrid']
        area_ = os.path.basename(f).split('.')[0].split('_')[-1].upper()
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c])

    spei_w, spei_d, spei_n = [], [], []
    simi_w, simi_d, simi_n = [], [], []
    area = []

    for k, v in data.items():
        for kk, vv in v.items():
            m = pd.to_datetime(kk).month
            sm, sp = vv['simi'], vv['spei']
            if m == month and np.isfinite(sm) and np.isfinite(sp):
                if sp > 0:
                    spei_w.append('Wet')
                    simi_w.append(sm)
                elif -1. < sp <= 0:
                    spei_n.append('Normal')
                    simi_n.append(sm)
                else:
                    spei_d.append('Dry')
                    simi_d.append(sm)
                area.append(area_)
                if len(simi_n + simi_d + simi_w) % 10000 == 0:
                    print(len(simi_n + simi_d + simi_w))
        else:
            continue
        break

    simi = simi_n + simi_d + simi_w
    spei = spei_n + spei_d + spei_w
    if first:
        c = pd.DataFrame(data=np.array([simi, spei, area]).T,
                          columns=['SIMI', 'SPEI', 'Management'])
        first = False
    else:
        f = pd.DataFrame(data=np.array([simi, spei, area]).T,
                         columns=['SIMI', 'SPEI', 'Management'])
        c = pd.concat([c, f])

    c['SIMI'] = c['SIMI'].values.astype(float)
    sns.violinplot(data=c, x='SPEI', y='SIMI', hue='Management', order=['Dry', 'Normal', 'Wet'])
    plt.suptitle('Colorado Irrigation Management')
    plt.savefig(out_fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = '/media/nvm/field_pts/indices'
    meta = '/media/nvm/field_pts/usbr_attr'
    out_ = os.path.join(root, 'figures', 'partitions', 'fields.png')
    partition_response(in_, meta, out_)
# ========================= EOF ====================================================================
