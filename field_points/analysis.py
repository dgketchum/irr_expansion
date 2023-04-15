import os
import json
import timeit
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from field_points.crop_codes import BASIN_STATES
from climate_indices import compute, indices
from transition_modeling.transition_data import KEYS, OLD_KEYS
from field_points.crops import cdl_key

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


def cdl_spei(npy, cdl_timescale, out_dir):
    with open(cdl_timescale, 'r') as fp:
        cdl_ts = json.load(fp)

    COLS.append('cdl')
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(2005, 2022) for m in range(1, 13)]

    rec, nrec = None, None
    first = True
    params, ts_len = 5, 17

    for state in BASIN_STATES:

        npy_file = os.path.join(npy, '{}.npy'.format(state))
        print(npy_file)
        js = npy_file.replace('.npy', '_index.json')
        with open(js, 'r') as fp:
            dct = json.load(fp)
            index = dct['index']

        n = int(np.ceil(len(index) / 5e3))
        indx = split(index, n)

        data_mem = np.fromfile(npy_file)
        data_mem = data_mem.reshape((len(index), -1, len(COLS)))

        for i, (s_ind, e_ind) in enumerate(indx):

            data = data_mem[s_ind:e_ind, :, :]
            shape = (params, data.shape[0], ts_len)
            target = np.zeros(shape)

            for crop in KEYS:

                ts, lb = cdl_ts[str(crop)]['ts'], cdl_ts[str(crop)]['lb']

                if float(crop) not in data[:, :, 6]:
                    continue

                months = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                     np.array([dt.month for dt in dt_range]))

                lb_ = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                  np.array([lb for _ in dt_range]))

                ts_ = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                  np.array([ts for _ in dt_range]))

                classific = data[:, -len(dt_range):, -1]
                data = data[:, -len(dt_range):, :]

                # depends on locally modified climate_indices package that takes cwb = ppt - pet as input to spei
                cwb = data[:, :, COLS.index('ppt')] - data[:, :, COLS.index('etr')]
                spei_ = np.apply_along_axis(lambda x: indices.spei(cwb_mm=x, scale=ts, **IDX_KWARGS),
                                            arr=cwb, axis=1)
                if lb > 5:
                    spei_ = np.roll(spei_, 12, axis=1)

                stack = np.stack([spei_[:, -len(dt_range):], months, classific, ts_, lb_])
                d = stack[:, stack[1] == float(lb)].reshape((params, len(index[s_ind:e_ind]), ts_len))
                mask = d[2] == float(crop)
                target = np.where(mask, d, target)

            if first:
                rec = d.copy()
                first = False
            else:
                rec = np.append(rec, d, axis=1)
            print('{}, {} of {}'.format(state, i + 1, n))

    out_path = os.path.join(out_dir, 'cdl_spei.npy')
    rec.tofile(out_path)
    print('writing ', out_path)


def cdl_et(npy, out_js, parameter):
    cdl_ = cdl_key()
    COLS.append('cdl')
    dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(2008, 2022) for m in range(1, 13)]
    ct = 0
    params, ts_len, gs_len = 3, 14, 7
    dct = {}

    for crop in OLD_KEYS:

        cols = ['State', 'crop_code', 'count', 'mean_cc', 'min_cc', 'max_cc', 'std_cc']
        df = pd.DataFrame(columns=cols)

        for state in BASIN_STATES:

            npy_file = os.path.join(npy, '{}.npy'.format(state))
            js = npy_file.replace('.npy', '_index.json')
            with open(js, 'r') as fp:
                ind_dct = json.load(fp)
                index = ind_dct['index']

            n = int(np.ceil(len(index) / 5e3))
            indx = split(index, n)

            data_mem = np.fromfile(npy_file)
            data_mem = data_mem.reshape((len(index), -1, len(COLS)))

            for i, (s_ind, e_ind) in enumerate(indx):

                data = data_mem[s_ind:e_ind, :, :]

                if float(crop) not in data[:, :, 6]:
                    continue

                months = np.multiply(np.ones((len(index[s_ind:e_ind]), len(dt_range))),
                                     np.array([dt.month in list(range(4, 11)) for dt in dt_range]))

                classific = data[:, -len(dt_range):, -1]
                data = data[:, -len(dt_range):, :]
                cc = data[:, :, COLS.index(parameter)]

                stack = np.stack([cc, months, classific])
                d = stack[:, stack[1] == 1]
                d = d.reshape((params, len(index[s_ind:e_ind]), ts_len * gs_len))
                mask = d[2] == float(crop)
                d = np.where(mask, d, np.nan)
                d = np.nansum(d, axis=2) / (np.count_nonzero(~np.isnan(d), axis=2) / 7)
                d = d[0, :]
                d = d[~np.isnan(d)]

                try:
                    df.loc[ct, cols] = [state, crop, d.shape[0], d.mean(), d.min(), d.max(), d.std()]
                except ValueError:
                    continue

                ct += 1

        mean_cc = ((df['count'] * df['mean_cc']) / df['count'].sum()).sum()
        dct[crop] = mean_cc
        print('{} {}: {:.3f}'.format(crop,  cdl_[crop][0], mean_cc))

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/nvm'
    if not os.path.exists(root):
        root = '/home/dgketchum/data'

    indir = os.path.join(root, 'field_pts/fields_data/fields_npy')
    odir = os.path.join(root, 'field_pts/indices')

    part = 'cdl'
    t_scales = '/media/research/IrrigationGIS/expansion/analysis/cdl_spei_timescales.json'
    indir = os.path.join(root, 'field_pts/fields_data/fields_cdl_npy')
    spei = os.path.join(root, 'field_pts/fields_data/cdl_spei')
    # cdl_spei(indir, t_scales, spei)

    param = 'cc'
    ccons = os.path.join(root, 'field_pts/fields_data/cdl_{}.json'.format(param))
    cdl_et(indir, ccons, param)
# ========================= EOF ====================================================================
