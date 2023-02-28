import os
import json
import timeit
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from call_ee import BASIN_STATES
from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
META_COLS = ['STUSPS', 'x', 'y', 'name', 'usbrid']

IDX_KWARGS = dict(distribution=indices.Distribution.gamma,
                  data_start_year=1984,
                  calibration_year_initial=1984,
                  calibration_year_final=2021,
                  periodicity=compute.Periodicity.monthly)


def split(a, n):
    k, m = divmod(len(a), n)
    inds = [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]
    return inds


class ArrayDisAssembly(object):

    def __init__(self, arr):
        self.arrays = None
        self.n_sections = None
        self.assembled = None
        self.axis = None

        if isinstance(arr, list):
            self.arrays = arr
            self.assembled = self.assemble(arr)

        self.original = arr
        self.shape = arr.shape

    def disassemble(self, n_sections, axis=1):
        self.arrays = np.array_split(self.original, n_sections, axis=axis)
        self.n_sections = n_sections
        return self.arrays

    def assemble(self, results, axis=1):
        d = {r.idx: r.arr for r in results}
        l = [d[k] for k in sorted(d.keys())]
        self.assembled = np.concatenate(l, axis=axis)
        return self.assembled


def f_(d_):
    coref = [np.ma.corrcoef(d_[0, i, :], d_[1, i, :])[0][1].item() ** 2 for i in range(d_.shape[1])]
    return coref


def correlations(state, npy_dir, out_dir, procs, calc):
    met_periods = list(range(1, 13)) + [18, 24, 30, 36]
    ag_periods = list(range(1, 8))
    periods = list(product(met_periods, ag_periods))

    npy = os.path.join(npy_dir, '{}.npy'.format(state))
    print('\n', npy)
    js = npy.replace('.npy', '_index.json')
    data = np.fromfile(npy, dtype=float)

    with open(js, 'r') as fp:
        index = json.load(fp)['index']

    print(len(index), 'fields')
    data = data.reshape((len(index), -1, len(COLS)))
    df = pd.DataFrame(index=index)
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(1987, 2022) for m in range(1, 13)]
    months = np.multiply(np.ones((len(index), len(dt_range))), np.array([dt.month for dt in dt_range]))

    for met_p, ag_p in periods:

        start_time = timeit.default_timer()

        if calc == 'simi':
            kc = data[:, :, COLS.index('et')] / data[:, :, COLS.index('ietr')]
        else:
            kc = data[:, :, COLS.index('cc')]
            kc[kc < 0.] = 0.

        simi = np.apply_along_axis(lambda x: indices.spi(x, scale=ag_p, **IDX_KWARGS), arr=kc, axis=1)

        # uses locally modified climate_indices package that takes cwb = ppt - pet as input
        cwb = data[:, :, COLS.index('ppt')] - data[:, :, COLS.index('etr')]
        spei = np.apply_along_axis(lambda x: indices.spei(x, scale=met_p, **IDX_KWARGS), arr=cwb, axis=1)

        stack = np.stack([simi[:, -len(dt_range):], spei[:, -len(dt_range):], months])

        for m in range(4, 11):

            if m - ag_p < 3:
                continue

            d = stack[:, stack[2] == float(m)].reshape((3, len(index), -1))
            mx = np.ma.masked_array(np.repeat(np.isnan(d[:1, :, :]), 3, axis=0))
            d = np.ma.MaskedArray(d, mx)

            a = ArrayDisAssembly(d)
            arrays = a.disassemble(n_sections=procs)
            pool = Pool(processes=procs)

            with pool as p:
                pool_results = [p.apply_async(f_, args=(a_,)) for a_ in arrays]
                corefs = [res.get() for res in pool_results]

            corefs = np.array([item for sublist in corefs for item in sublist])
            col = 'met{}_ag{}_fr{}'.format(met_p, ag_p, m)
            df[col] = corefs

        print(state, met_p, ag_p, '{:.3f}'.format(timeit.default_timer() - start_time))

    ofile = os.path.join(out_dir, calc, '{}.csv'.format(state))
    df.to_csv(ofile)
    return ofile


def partition_data(npy, out_dir, calc='simi', classification='usbr'):
    periods = [(5, 1, 1),
               (6, 2, 1),
               (7, 2, 1),
               (8, 4, 3),
               (9, 5, 4),
               (10, 7, 5)]

    if classification == 'cdl':
        COLS.append('cdl')
        dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(2005, 2022) for m in range(1, 13)]
    else:
        dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(1987, 2022) for m in range(1, 13)]

    join_column, states = None, None

    for p in periods:
        rec, nrec = None, None
        month_end, met_time, ag_time = p
        desc = 'met{}_ag{}_fr{}'.format(met_time, ag_time, month_end)
        first = True

        if classification == 'usbr':
            states = ['CA', 'ID', 'MT', 'OR', 'WA', 'WY']
            join_column = 'usbrid'
        elif classification == 'itype':
            states = ['CO', 'MT', 'NM', 'UT', 'WA', 'WY']
            join_column = 'itype'
        elif classification == 'time':
            states = BASIN_STATES
            join_column = 'index'
        elif classification == 'cdl':
            states = BASIN_STATES
            join_column = 'index'

        for state in states:

            npy_file = os.path.join(npy, '{}.npy'.format(state, state))
            print(npy_file)
            js = npy_file.replace('.npy', '_index.json')
            with open(js, 'r') as fp:
                dct = json.load(fp)
                index = dct['index']
                if classification == 'time':
                    class_ = [int(i.split('_')[1]) for i in index]
                else:
                    class_ = dct[join_column]

            n = int(np.ceil(len(index) / 5e3))
            indx = split(index, n)

            data_mem = np.fromfile(npy_file)
            data_mem = data_mem.reshape((len(index), -1, len(COLS)))

            for i, (s_ind, e_ind) in enumerate(indx):

                data = data_mem[s_ind:e_ind, :, :]

                months = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                     np.array([dt.month for dt in dt_range]))

                if classification == 'cdl':
                    classific = data[:, -len(dt_range):, -1]
                    data = data[:, -len(dt_range):, :]
                else:
                    classific = np.repeat(np.array(class_[s_ind:e_ind]).reshape((len(class_[s_ind:e_ind]), 1)),
                                          len(dt_range), axis=1)

                if calc == 'simi':
                    kc = data[:, :, COLS.index('et')] / data[:, :, COLS.index('ietr')]
                else:
                    kc = data[:, :, COLS.index('cc')]
                    kc[kc < 0.] = 0.

                simi = np.apply_along_axis(lambda x: indices.spi(x, scale=ag_time, **IDX_KWARGS), arr=kc, axis=1)

                # depends on locally modified climate_indices package that takes cwb = ppt - pet as input to spei
                cwb = data[:, :, COLS.index('ppt')] - data[:, :, COLS.index('etr')]
                spei = np.apply_along_axis(lambda x: indices.spei(x, scale=met_time, **IDX_KWARGS), arr=cwb, axis=1)
                stack = np.stack([simi[:, -len(dt_range):], spei[:, -len(dt_range):], months, classific])
                d = stack[:, stack[2] == float(month_end)].reshape((4, len(index[s_ind:e_ind]), -1))

                if classification == 'itype':
                    d = d[:, :, -7:]

                if classification == 'time':
                    early = d[:, :, :10]
                    late = d[:, :, -10:]
                    if first:
                        e = early.copy()
                        l = late.copy()
                        first = False
                    else:
                        e = np.append(e, early, axis=1)
                        l = np.append(l, late, axis=1)
                    print('{}, {} of {}'.format(state, i + 1, n))
                    continue

                if first:
                    rec = d.copy()
                    first = False
                else:
                    rec = np.append(rec, d, axis=1)
                print('{}, {} of {}'.format(state, i + 1, n))

        if classification == 'time':
            out_path = os.path.join(out_dir, '{}_early.npy'.format(desc))
            e.tofile(out_path)
            out_path = os.path.join(out_dir, '{}_late.npy'.format(desc))
            l.tofile(out_path)
            print('writing ', out_path)

        else:
            out_path = os.path.join(out_dir, '{}.npy'.format(desc))
            rec.tofile(out_path)
            print('writing ', out_path)


if __name__ == '__main__':
    root = '/media/nvm'
    if not os.path.exists(root):
        root = '/home/dgketchum/data'

    indir = os.path.join(root, 'field_pts/fields_data/fields_npy')
    odir = os.path.join(root, 'field_pts/indices')

    part = 'itype'
    part_ = os.path.join(root, 'field_pts/fields_data/partitioned_npy/{}'.format(part))
    # partition_data(indir, part_, classification=part)

    part = 'cdl'
    indir = os.path.join(root, 'field_pts/fields_data/fields_cdl_npy')
    part_ = os.path.join(root, 'field_pts/fields_data/partitioned_npy/{}'.format(part))
    partition_data(indir, part_, classification=part)
# ========================= EOF ====================================================================
