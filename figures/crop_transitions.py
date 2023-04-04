import os
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from field_points.crop_codes import cdl_key
from transition_modeling.transition_data import KEYS


def transition_probability(cdl_npy, out_matrix):
    counts, set_ = None, None
    cdl = cdl_key()

    time_series_length = 17
    df = None

    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    set_, classes = KEYS, [cdl[c][0] for c in KEYS]

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = [(int(a), int(b)) for a, b in keys]
    keys = sorted(keys, key=lambda x: x[0])
    rec = np.moveaxis(rec, 0, 1)

    dct = {clime: {k: 0 for k in keys} for clime in ['Wet', 'Normal', 'Dry', 'All']}
    from_ct = {clime: {k: 0 for k in KEYS} for clime in dct.keys()}

    for i, v in enumerate(rec):

        if i % 10000 == 0 and i > 1:
            print('{} of {}'.format(i, rec.shape[0]))


        for e, c in enumerate(v.T):
            if np.any(np.isnan(c)):
                continue
            if e == 16:
                continue
            if c[1] >= 0:
                clime = 'Wet'
            elif 0 > c[1] > -1.3:
                clime = 'Normal'
            elif c[1] < -1.3:
                clime = 'Dry'

            trans = (int(c[3]), int(v.T[e + 1, 3]))

            if trans[0] not in KEYS or trans[1] not in KEYS:
                continue

            dct[clime][trans] += 1
            from_ct[clime][trans[0]] += 1
            dct['All'][trans] += 1
            from_ct['All'][trans[0]] += 1

    for k, d in dct.items():
        map = np.zeros((len(KEYS), len(KEYS)))
        for r, c in d.keys():
            map[KEYS.index(r), KEYS.index(c)] = d[r, c]

        prob = np.divide(map, np.sum(map, axis=0))
        fig = plt.figure(figsize=(16, 10))
        ax = sns.heatmap(prob, square=False, annot=True, cmap='rocket_r', cbar=False,
                         xticklabels=classes, yticklabels=classes, fmt='.2f')
        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(['n = {:,}'.format(v) for k, v in from_ct[k].items()])
        plt.yticks(rotation=0)

        if k == 'All':
            plt.suptitle('{} Year Crop Transition Probabilities'.format(k))
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        else:
            plt.suptitle('{} Year Crop Transition Probabilities'.format(k))
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        plt.tight_layout()
        plt.show()
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    npy = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met5_ag4_fr9.npy'
    out_ = os.path.join(root, 'figures', 'crop_transitions')
    transition_probability(npy, out_)
# ========================= EOF ====================================================================
