import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def partition_response(in_json, out_fig, month=10):
    with open(in_json, 'r') as f_obj:
        data = json.load(f_obj)

    spei_w, spei_d, spei_n = [], [], []
    simi_w, simi_d, simi_n = [], [], []

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

                if len(simi_n + simi_d + simi_w) > 1000:
                    break
        else:
            continue
        break

    simi = simi_n + simi_d + simi_w
    spei = spei_n + spei_d + spei_w
    sns.violinplot(x=spei, y=simi, order=['Dry', 'Normal', 'Wet'])
    plt.show()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    in_ = '/media/nvm/field_pts/indices/CA.json'
    out_ = os.path.join(root, 'figures', 'partitions')
    partition_response(in_, out_, month=10)
# ========================= EOF ====================================================================
