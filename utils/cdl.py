import json

import requests

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def cdl_accuracy(out_js):
    dct = {s: [] for s in BASIN_STATES}
    for y in range(2008, 2022):
        for s in BASIN_STATES:
            url = 'https://www.nass.usda.gov/Research_and_Science/' \
                  'Cropland/metadata/metadata_{}{}.htm'.format(s.lower(), str(y)[-2:])
            resp = requests.get(url).content.decode('utf-8')
            for i in resp.split('\n'):
                txt = i.strip('\r')
                if txt.startswith('OVERALL'):
                    l = txt.split(' ')
                    k = float(l[-1])
            dct[s].append(k)
            print(s, y, '{:.3f}'.format(k))

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
