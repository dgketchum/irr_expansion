import os
import json
from itertools import combinations
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import arviz as az

from field_points.crop_codes import cdl_key
from transition_modeling.transition_data import KEYS

# count sort
SORTED_KEYS = [36, 1, 21, 37, 24, 23, 43, 41, 42, 49, 53, 12, 28]


def partition_cdl(cdl_npy, out_js):
    cdl = cdl_key()
    time_series_length = 17
    rec = np.fromfile(cdl_npy, dtype=float).reshape((4, -1, time_series_length))
    set_, classes = KEYS, [cdl[c][0] for c in SORTED_KEYS]

    set_ = [s for s in set_ if s > 0]
    keys = list((combinations(set_, 2)))
    opposites = [(s, f) for f, s in keys]
    [keys.append(o) for o in opposites]
    [keys.append((i, i)) for i in set_]
    keys = [(int(a), int(b)) for a, b in keys]
    keys = sorted(keys, key=lambda x: x[0])
    keys = ['{}_{}'.format(*trans) for trans in keys]
    rec = np.moveaxis(rec, 0, 1)

    dct = OrderedDict({clime: {k: 0 for k in keys} for clime in ['Wet', 'Normal', 'Dry', 'All']})
    from_ct = {clime: {k: 0 for k in KEYS} for clime in dct.keys()}

    for i, v in enumerate(rec):

        if np.any(np.isnan(v)):
            continue

        if i % 10000 == 0 and i > 1:
            print('{} of {}'.format(i, rec.shape[0]))
            # break

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

            dct[clime]['{}_{}'.format(*trans)] += 1
            from_ct[clime][trans[0]] += 1
            dct['All']['{}_{}'.format(*trans)] += 1
            from_ct['All'][trans[0]] += 1

    dct['Difference'] = 'None'
    dct['from_ct'] = from_ct['All'].copy()

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


def transition_probability(cdl_data, cdl_acc, out_matrix):
    with open(cdl_acc, 'r') as fp:
        acc = json.load(fp)

    with open(cdl_data, 'r') as fp:
        dct = json.load(fp)

    cdl = cdl_key()
    set_, classes = KEYS, [cdl[c][0] for c in SORTED_KEYS]

    dct['from_ct'] = {int(k): v for k, v in dct['from_ct'].items()}

    for k, d in dct.items():
        if k == 'from_ct':
            continue
        elif k != 'Difference':
            tupd = {(int(k.split('_')[0]), int(k.split('_')[1])): v for k, v in d.items()}
            map_ = np.zeros((len(KEYS), len(KEYS)))
            for r, c in tupd.keys():
                map_[SORTED_KEYS.index(r), SORTED_KEYS.index(c)] = tupd[r, c]

            prob = map_ / np.sum(map_, axis=1)[:, np.newaxis]
            prob[np.isnan(prob)] = 0.0
            cmap = 'rocket_r'
        else:
            prob = dry_p - normal_p
            cmap = 'vlag'

        if k == 'Wet':
            normal_p = prob.copy()
        if k == 'Dry':
            dry_p = prob.copy()

        fig = plt.figure(figsize=(16, 10))
        ax = sns.heatmap(prob, square=False, annot=True, cmap=cmap, cbar=False,
                         xticklabels=classes, yticklabels=classes, fmt='.2f', annot_kws={'fontsize': 16})
        if k == 'All':
            ax.add_patch(Rectangle((2, 8), 1, 1, fill=False, edgecolor='green', lw=4, clip_on=False))
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ncounts = [dct['from_ct'][key] for key in SORTED_KEYS]
        acc_ct = [int(np.rint((1 - acc[str(key)]) * c)) for key, c in zip(SORTED_KEYS, ncounts)]
        decs = [len(str(c)) for c in ncounts]
        decs_acc = [len(str(c)) for c in acc_ct]
        ncounts = [np.around(c, -1 * (d - 3)) for c, d in zip(ncounts, decs)]
        acc_ct = [np.around(c, -1 * (d - 3)) for c, d in zip(acc_ct, decs_acc)]
        ytick_labels = ['n = {:,} \u00b1 {:,}'.format(a, b) for a, b in zip(ncounts, acc_ct)]
        ax2.set_yticklabels(ytick_labels)
        ax2.set_yticks(ax.get_yticks())
        ax2.tick_params(axis='y', which='both', pad=10)

        for tick in ax.get_xticklabels():
            tick.set_fontsize(16)

        for tick in ax2.get_yticklabels():
            tick.set_fontsize(16)

        for tick in ax.get_yticklabels():
            tick.set_fontsize(16)

        if k == 'All':
            plt.suptitle('Crop Transition Probabilities')
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        elif k == 'Difference':
            plt.suptitle('Crop Transition Probabilities Difference\nDry Minus Wet'.format(k))
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        else:
            plt.suptitle('{} Year Crop Transition Probabilities'.format(k))
            fig_file = os.path.join(out_matrix, 'transition_heatmap_{}.png'.format(k))

        plt.tight_layout()
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()


def transition_et(cdl_data, cdl_acc, et_data, out_matrix):
    with open(cdl_acc, 'r') as fp:
        acc = json.load(fp)

    with open(cdl_data, 'r') as fp:
        dct = json.load(fp)

    with open(et_data, 'r') as fp:
        et = json.load(fp)
        et = pd.Series(et)
        et = pd.DataFrame(et)
        et = et.rename(columns={0: 'IWU'})
        et['Crop Code'] = et.index
        et = et.loc[[i for i in et.index if int(i) in SORTED_KEYS]]
        et = et.reindex([str(k) for k in SORTED_KEYS])
        et.index = [i for i in range(len(et))]

    cdl = cdl_key()
    set_, classes = KEYS, [cdl[c][0] for c in SORTED_KEYS]

    et['Crop'] = classes
    et = et[['Crop', 'IWU']]
    et['sd'] = et['IWU'] * 0.33

    dct['from_ct'] = {int(k): v for k, v in dct['from_ct'].items()}

    for k, d in dct.items():
        if k == 'from_ct':
            continue
        elif k != 'Difference':
            tupd = {(int(k.split('_')[0]), int(k.split('_')[1])): v for k, v in d.items()}
            map_ = np.zeros((len(KEYS), len(KEYS)))
            for r, c in tupd.keys():
                map_[SORTED_KEYS.index(r), SORTED_KEYS.index(c)] = tupd[r, c]

            prob = map_ / np.sum(map_, axis=1)[:, np.newaxis]
            prob[np.isnan(prob)] = 0.0
            cmap = 'rocket_r'
        else:
            prob = dry_p - normal_p
            cmap = 'vlag'

        if k == 'Wet':
            normal_p = prob.copy()
        if k == 'Dry':
            dry_p = prob.copy()

        if k != 'Difference':
            continue

        fig, [ax2, ax1] = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        palette = sns.color_palette('vlag')

        sns.barplot(x='Crop', y='IWU', data=et, yerr=et['sd'], ec='black', color='white', errorbar=('ci', 95),
                    orient='v', width=0.9, ax=ax1)
        ax1.set_xlabel('Mean Crop Irrigation Water Use', fontsize=16)
        ax1.set_ylabel('[m yr$^{-1}$]', fontsize=16)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.legend([], [], frameon=False)
        for tick in ax1.get_yticklabels():
            tick.set_fontsize(16)

        classes[3] = 'Other Hay'
        sns.heatmap(prob, square=False, annot=True, cmap=cmap, cbar=False,
                    xticklabels=classes, yticklabels=classes, fmt='.2f', ax=ax2, annot_kws={'fontsize': 16})
        ax3 = ax2.twinx()
        ax3.set_ylim(ax2.get_ylim())
        ncounts = [dct['from_ct'][key] for key in SORTED_KEYS]
        acc_ct = [int(np.rint((1 - acc[str(key)]) * c)) for key, c in zip(SORTED_KEYS, ncounts)]
        decs = [len(str(c)) for c in ncounts]
        decs_acc = [len(str(c)) for c in acc_ct]
        ncounts = [np.around(c, -1 * (d - 3)) for c, d in zip(ncounts, decs)]
        acc_ct = [np.around(c, -1 * (d - 3)) for c, d in zip(acc_ct, decs_acc)]
        ytick_labels = ['n = {:,} \u00b1 {:,}'.format(a, b) for a, b in zip(ncounts, acc_ct)]
        ax3.set_yticklabels(ytick_labels)
        ax3.set_yticks(ax2.get_yticks())
        ax3.tick_params(axis='y', which='both', pad=10)

        for tick in ax2.get_xticklabels():
            tick.set_fontsize(16)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

        for tick in ax2.get_yticklabels():
            tick.set_fontsize(16)

        for tick in ax3.get_yticklabels():
            tick.set_fontsize(16)

        # plt.suptitle('Crop Transition Probabilities Difference\nDry Minus Wet'.format(k))
        fig_file = os.path.join(out_matrix, 'transition_cc_{}.png'.format(k))
        plt.tight_layout()
        # plt.show()
        plt.savefig(fig_file, bbox_inches='tight')
        plt.close()


def transition_model(summaries, traces, sample, figures, n_include=5):
    cdl = cdl_key()
    # predominant_crops = [1, 21, 24, 23, 36]
    palette = sns.color_palette('rocket', as_cmap=True)
    three_colors = [palette.colors[i] for i in [100, 150, 200]]

    with open(sample, 'r') as f_obj:
        sample = json.load(f_obj)

    for code in KEYS:

        print('\ncode {}'.format(code))
        sample_d = sample[str(code)]
        label_map = {v: k for k, v in sample_d['label_map'].items()}
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
        crop_codes = [label_map[c] for c in out_keys]
        names = [cdl[int(c)][0] for c in crop_codes]
        print(names)
        df = df.loc[df['label'].apply(lambda x: x in [x[0] for x in counts])]

        tfile = os.path.join(traces, 'model_sft_{}.nc'.format(code))
        if not os.path.exists(tfile):
            continue
        trace = az.from_netcdf(tfile)
        assert trace.observed_data.indexes.dims['obs_dim_0'] == len(sample_d['y'])

        labels = [int(x) for x in list(trace.posterior['a'].labels.values)]
        label_idx = [labels.index(c) for c in out_keys if c in labels]

        alpha = trace.posterior['a'].values
        alpha = alpha.reshape(alpha.shape[0] * alpha.shape[1], -1)
        dct = {'a[{}]'.format(k): alpha[:, i] for i, k in zip(label_idx, out_keys)}

        coeffs = trace.posterior['b'].features.values
        coeff_names = ['Climate', 'From Price', 'To Price']
        arr = trace.posterior['b'].values
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], -1)

        for i, c in enumerate(coeffs):
            for j, k in zip(label_idx, out_keys):
                dct['b[{}, {}]'.format(c, k)] = arr[:, i, j]

        data_df = pd.DataFrame(dct)
        min_max_cols = [i for i in df.index if df.loc[i, 'coeff'] != 'a']
        min_, max_ = data_df[min_max_cols].values.flatten().min(), data_df[min_max_cols].values.flatten().max()
        fig, ax = plt.subplots(1, 5, figsize=(16, 6))
        ct_strings = {x[0]: x[1] for x in counts}
        alphas = {k: data_df['a[{}]'.format(k)].mean(axis=0) for k in out_keys}
        alphas = {k: '{:.2f}'.format(v) for k, v in alphas.items()}
        for i, (name, key) in enumerate(zip(names, out_keys)):
            cols = [i for i in df.index if df.loc[i, 'label'] == key and df.loc[i, 'coeff'] != 'a']
            data = data_df[cols]
            sns.boxplot(data, ax=ax[i], palette=three_colors)
            ax[i].set_xticklabels(coeff_names)
            ax[i].title.set_text(name)
            ax[i].set_ylim([min_, max_])
            ct_str = ct_strings[key]
            ax[i].set(xlabel='n = {:,}\n \u03B1: {}'.format(ct_str, alphas[key]))

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
    out_js_ = '/media/nvm/field_pts/fields_data/partitioned_npy/cdl/cdl_prob.json'
    # partition_cdl(npy, out_js_)

    out_ = os.path.join(root, 'figures', 'crop_transitions', 'transition_matrix')
    cdl_accuracy = os.path.join(root, 'analysis/cdl_acc.json')
    # transition_probability(out_js_, cdl_accuracy, out_)

    out_ = os.path.join(root, 'figures', 'crop_transitions', 'transition_cc')
    ccons = '/media/nvm/field_pts/fields_data/cdl_cc.json'
    transition_et(out_js_, cdl_accuracy, ccons, out_)

    out_f = os.path.join(root, 'figures', 'crop_transitions', 'transition_models')
    summaries_ = os.path.join(root, 'analysis', 'transition', 'summaries')
    traces = os.path.join(root, 'analysis', 'transition', 'models')
    data = os.path.join(root, 'analysis', 'transition', 'sample_data', 'sample_50000.json')
    # transition_model(summaries_, traces, data, out_f)

# ========================= EOF ====================================================================
