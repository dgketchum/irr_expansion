import os

import numpy as np
import pytz
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import dataretrieval.nwis as nwis

from utils.state_county_names_codes import state_county_code

PARAMS = {'station_nm': 'STANAME',
          'dec_lat_va': 'LAT',
          'dec_long_va': 'LON',
          'county_cd': 'CO_CODE',
          'alt_va': 'ELEV',
          'huc_cd': 'HUC',
          'construction_dt': 'CONSTRU',
          'drain_area_va': 'BASINAREA'}

ORDER = ['STAID', 'STANAME', 'start', 'end', 'last_rec', 'rec_len', 'CO_CODE', 'LAT', 'LON',
         'ELEV', 'HUC', 'CONSTRU', 'BASINAREA', 'COUNTY', 'geometry']


def get_usgs_station_metadata(in_shp, out_shp):
    df = gpd.read_file(in_shp)
    df.index = df['SOURCE_FEA']
    df = df[['geometry']]
    df['CO_CODE'] = [0 for _ in df.iterrows()]

    co_codes = state_county_code()['MT']

    today = datetime.now(pytz.utc)

    for sid, feat in df.iterrows():
        data = nwis.get_info(sites=[sid])

        try:
            for param, name in PARAMS.items():
                df.loc[sid, name] = data[0][param][0]

            df.loc[sid, 'COUNTY'] = co_codes[str(df.loc[sid, 'CO_CODE']).rjust(3, '0')]['NAME']

        except KeyError as e:
            print('\n Error on {}: {}'.format(sid, e))

        recs = nwis.get_record(sites=[sid], start='1800-01-01', end='2023-08-31', service='dv')
        if not isinstance(recs.index, pd.DatetimeIndex):
            df.loc[sid, 'start'] = 'None'
            df.loc[sid, 'end'] = 'None'
        else:
            s, e = recs.index[0], recs.index[-1]
            rec_start = '{}-{}-{}'.format(s.year, str(s.month).rjust(2, '0'), str(s.day).rjust(2, '0'))
            df.loc[sid, 'start'] = rec_start
            rec_end = '{}-{}-{}'.format(e.year, str(e.month).rjust(2, '0'), str(e.day).rjust(2, '0'))
            df.loc[sid, 'end'] = rec_end
            df.loc[sid, 'last_rec'] = np.round((today - e).days / 365.25, 2)
            df.loc[sid, 'rec_len'] = df.shape[0]
            print(sid, df.loc[sid, 'STANAME'], rec_start, rec_end)

    df['STAID'] = df.index
    df = df[ORDER]
    df.to_file(out_shp)


if __name__ == '__main__':
    gages_in = '/media/research/IrrigationGIS/usgs_gages/mt_usgs_gages.shp'
    gages_out = '/media/research/IrrigationGIS/usgs_gages/mt_usgs_gages_por.shp'
    get_usgs_station_metadata(gages_in, gages_out)
# ========================= EOF ====================================================================
