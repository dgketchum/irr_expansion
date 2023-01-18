import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_training_inference(train, infer, fig_dir, year=2020):
    df = pd.read_csv(train, engine='python').sample(frac=1.0).reset_index(drop=True)
    idf = pd.read_csv(infer, engine='python').sample(frac=1.0).reset_index(drop=True)

    et_cols = ['et_{}_{}'.format(year, mm) for mm in range(4, 11)]
    df['season'] = df[et_cols].sum(axis=1)

    cols = [c for c in list(df.columns) if c not in et_cols]
    for col in cols:
        fig, ax = plt.subplots()
        try:
            sns.kdeplot(df[col], ax=ax, label='train')
            sns.kdeplot(idf[col], ax=ax, label='infer')
        except KeyError:
            continue
        plt.legend()
        plt.title(col)
        _figfile = os.path.join(fig_dir, '{}.png'.format(col))
        plt.savefig(_figfile)
        plt.close()
        plt.clf()
        print(col)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    prepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_29DEC2022')
    training_area = os.path.join(prepped, 'bands_29DEC2022_2020.csv')

    prepped = os.path.join(root, 'expansion', 'tables', 'prepped_bands', 'bands_irr_29DEC2022')
    inference_area = os.path.join(prepped, 'bands_irr_29DEC2022_2020.csv')

    fdir = '/media/research/IrrigationGIS/expansion/figures/train_infer_comparison'

    compare_training_inference(training_area, inference_area, fdir)

# ========================= EOF ====================================================================
