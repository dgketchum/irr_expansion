import os
import json
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from field_points.crop_codes import cdl_key
from transition_modeling.transition_data import KEYS


def transition_probability(cdl_npy, out_matrix):
    cdl = cdl_key()
    time_series_length = 17
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

        if np.any(np.isnan(v)):
            continue

        if i % 10000 == 0 and i > 1:
            print('{} of {}'.format(i, rec.shape[0]))

        for e, c in enumerate(v.T):

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
        map_ = np.zeros((len(KEYS), len(KEYS)))
        for r, c in d.keys():
            map_[KEYS.index(r), KEYS.index(c)] = d[r, c]

        prob = map_ / np.sum(map_, axis=1)[:, np.newaxis]
        prob[np.isnan(prob)] = 0.0

        fig = plt.figure(figsize=(16, 10))
        ax = sns.heatmap(prob, square=False, annot=True, cmap='rocket_r', cbar=False,
                         xticklabels=classes, yticklabels=classes, fmt='.2f')
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ytick_labels = ['n = {:,}'.format(from_ct[k][key]) for key in KEYS]
        ax2.set_yticklabels(ytick_labels)
        ax2.set_yticks(ax.get_yticks())
        ax2.tick_params(axis='y', which='both', pad=10)

        if k == 'All':
            plt.suptitle('Crop Transition Probabilities')
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        else:
            plt.suptitle('{} Year Crop Transition Probabilities'.format(k))
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        plt.tight_layout()
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()


def transition_model(summaries, traces, figures, n_include=5):
    cdl = cdl_key()
    predominant_crops = [36, 1, 21, 24, 23]

    for code in KEYS:
        sfile = os.path.join(summaries, 'model_sft_{}.csv'.format(code))
        try:
            df = pd.read_csv(sfile, index_col=0)
        except FileNotFoundError:
            print('{} not found'.format(sfile))
            continue
        counts = [(r['label'], int(r['counts'])) for i, r in df.iterrows() if r['coeff'] == 'a']
        counts = sorted(counts, key=lambda x: x[1], reverse=True)
        counts = counts[:n_include]
        out_keys = [x[0] for x in counts]
        names = [cdl[int(c)][0] for c in out_keys]
        print(names)
        df = df.loc[df['label'].apply(lambda x: x in [x[0] for x in counts])]

        tfile = os.path.join(traces, 'model_sft_{}.nc'.format(code))
        trace = az.from_netcdf(tfile)

        labels = [int(x) for x in list(trace.posterior['a'].labels.values)]
        label_idx = [labels.index(c) for c in out_keys if c in labels]

        alpha = trace.posterior['a'].values
        alpha = alpha.reshape(alpha.shape[0] * alpha.shape[1], -1)
        dct = {'a[{}]'.format(k): alpha[:, i] for i, k in zip(label_idx, out_keys)}

        coeffs = trace.posterior['b'].features.values
        arr = trace.posterior['b'].values
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], -1)
        for i, c in enumerate(coeffs):
            for j, k in zip(label_idx, out_keys):
                dct['b[{}, {}]'.format(c, k)] = arr[:, i, j]
        data_df = pd.DataFrame(dct)
        min_max_cols = [i for i in df.index if df.loc[i, 'coeff'] != 'a']
        min_, max_ = data_df[min_max_cols].values.flatten().min(), data_df[min_max_cols].values.flatten().max()
        fig, ax = plt.subplots(1, 5, figsize=(16, 8))
        for i, (name, key) in enumerate(zip(names, out_keys)):
            cols = [i for i in df.index if df.loc[i, 'label'] == key and df.loc[i, 'coeff'] != 'a']
            data = data_df[cols]
            sns.boxplot(data, ax=ax[i])
            ax[i].set_xticklabels(coeffs)
            ax[i].title.set_text(name)
            ax[i].set_ylim([min_, max_])

        plt.suptitle('{} Transition Model'.format(cdl[int(code)][0]))
        out_name = cdl[int(code)][0].replace('/', '_')
        sfig = os.path.join(figures, '{}_{}.png'.format(code, out_name))
        plt.savefig(sfig)
        print(sfig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    npy = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/met5_ag4_fr9.npy'
    out_ = os.path.join(root, 'figures', 'crop_transitions', 'transition_probability')
    # transition_probability(npy, out_)

    out_f = os.path.join(root, 'figures', 'crop_transitions', 'transition_models')
    summaries_ = os.path.join(root, 'analysis', 'transition', 'summaries')
    traces = os.path.join(root, 'analysis', 'transition', 'models')
    transition_model(summaries_, traces, out_f)
# ========================= EOF ====================================================================
