import json

import numpy as np
import requests

from field_points.analysis import OLD_KEYS
from field_points.crop_codes import cdl_key

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def cdl_accuracy(out_js):
    cdl = cdl_key()
    dct = {c: [] for c in OLD_KEYS}
    for y in range(2015, 2022):
        for s in BASIN_STATES:
            url = 'https://www.nass.usda.gov/Research_and_Science/' \
                  'Cropland/metadata/metadata_{}{}.htm'.format(s.lower(), str(y)[-2:])
            resp = requests.get(url).content.decode('utf-8')
            for key in OLD_KEYS:
                crop = cdl[key][0]
                line = [l for l in resp.split('\n') if l.startswith(crop)]
                if len(line) > 1 or len(line) == 0:
                    print(crop, 'not found')
                    continue
                line = [x.replace('%', '') for x in line[0].split()]
                try:
                    c = float(line[-7].replace(',', ''))
                    if c < 1000:
                        continue
                    p, r = float(line[-6]), float(line[-3])
                    if p == 0. or r == 0.:
                        continue
                except ValueError as e:
                    print(e, crop)
                    continue
                f1 = 0.01 * 2 * (p * r) / (p + r)
                dct[key].append((f1, c))
                print(s, y, key, crop, '{:.3f}'.format(f1))

    dct = {k: np.array(v) for k, v in dct.items()}
    wdct = {}
    ct = 0
    for k, v in dct.items():
        wgt = (v[:, 0] * (v[:, 1] / v[:, 1].sum(axis=0))).sum()
        ct += v[:, 1].sum()
        wdct[k] = wgt

    print(ct)
    with open(out_js, 'w') as fp:
        json.dump(wdct, fp, indent=4)


if __name__ == '__main__':
    cdl_accuracy('cdl_acc.json')
# ========================= EOF ====================================================================
