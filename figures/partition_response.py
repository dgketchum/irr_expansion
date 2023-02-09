import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def partition_response(in_json, out_fig, month=10):
    if not isinstance(in_json, list):
        in_json = [in_json]

    first = True
    for js in in_json:
        with open(js, 'r') as f_obj:
            data = json.load(f_obj)

        area_ = os.path.basename(js).split('.')[0].split('_')[-1].upper()

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
                    if len(simi_n + simi_d + simi_w) % 1000 == 0:
                        print(len(simi_n + simi_d + simi_w))
            else:
                continue
            break

        simi = simi_n + simi_d + simi_w
        spei = spei_n + spei_d + spei_w
        if first:
            df = pd.DataFrame(data=np.array([simi, spei, area]).T,
                              columns=['SIMI', 'SPEI', 'Management'])
            first = False
        else:
            c = pd.DataFrame(data=np.array([simi, spei, area]).T,
                             columns=['SIMI', 'SPEI', 'Management'])
            df = pd.concat([df, c])
            df['SIMI'] = df['SIMI'].values.astype(float)

    sns.violinplot(data=df, x='SPEI', y='SIMI', hue='Management', order=['Dry', 'Normal', 'Wet'])
    plt.suptitle('Klamath Basin, CA Irrigation Management')
    plt.savefig(out_fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = ['/media/nvm/field_pts/indices/CA_usbr.json',
           '/media/nvm/field_pts/indices/CA_nonusbr.json']
    out_ = os.path.join(root, 'figures', 'partitions', 'Klamath.png')
    partition_response(in_, out_, month=10)
# ========================= EOF ====================================================================
