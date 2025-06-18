import os
import json
import pytz
import time
from datetime import timedelta, date, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import geopandas as gpd

import pyproj
import pynldas2 as nld
from rasterstats import zonal_stats

from dri.thredds import GridMet

CLIMATE_COLS = {
    'pet': {
        'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_grass',
        'col': 'eto_mm'},
    'pr': {
        'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
        'var': 'precipitation_amount',
        'col': 'prcp_mm'},
}

GRIDMET_GET = ['elev_m',

               'eto_mm',
               'prcp_mm',
               ]

BASIC_REQ = ['date', 'year', 'month', 'day', 'centroid_lat', 'centroid_lon']

COLUMN_ORDER = BASIC_REQ + GRIDMET_GET


def run_zonal_stats_for_fields(fields_with_gfid, gridmet_points, gridmet_ras,
                               factors_js, field_select=None, feature_id='FID'):
    """"""

    convert_to_wgs84 = pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326', always_xy=True).transform

    fields = gpd.read_file(fields_with_gfid)
    gridmet_pts = gpd.read_file(gridmet_points)
    gridmet_pts = gridmet_pts.set_index('GFID')

    rasters = []
    for var in ['eto', 'etr']:
        for month in range(1, 13):
            raster_path = os.path.join(gridmet_ras, f'gridmet_corrected_{var}_{month}.tif')
            rasters.append(raster_path)

    gridmet_targets = {}

    for i, field in tqdm(fields.iterrows(), desc='Extraction GriMET correction factors', total=fields.shape[0]):

        if field_select and str(field[feature_id]) not in field_select:
            continue

        centroid = field['geometry'].centroid
        lon, lat = convert_to_wgs84(centroid.x, centroid.y)
        gfid = field['GFID']

        fields.at[i, 'STATION_ID'] = gfid

        if gfid not in gridmet_targets:
            gridmet_targets[gfid] = {str(m): {} for m in range(1, 13)}
            gridmet_point_geom = gridmet_pts.loc[gfid]['geometry']
            gridmet_targets[gfid]['lat'] = gridmet_pts.loc[gfid]['lat']
            gridmet_targets[gfid]['lon'] = gridmet_pts.loc[gfid]['lon']
            gdf_point = gpd.GeoDataFrame({'geometry': [gridmet_point_geom]}, crs='EPSG:5071')
            for r in rasters:
                try:
                    splt = os.path.basename(r).split('_')
                    _var, month = splt[-2], splt[-1].replace('.tif', '')
                    stats = zonal_stats(gdf_point, r, stats=['mean'])[0]['mean']
                    gridmet_targets[gfid][month].update({_var: stats})
                except Exception as e:
                    print(f"failed on {r} for GFID {gfid}. Error: {e}")

        print(f'{i} of {fields.shape[0]} gridmet points processed', flush=True)

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4, sort_keys=True)


def get_gridmet_corrections(fields, gridmet_ras, fields_join,
                            factors_js, field_select=None, feature_id='FID'):
    print('Find field-gridmet joins')

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    fields = gpd.read_file(fields)

    oshape = fields.shape[0]

    rasters = []
    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

    gridmet_targets = {}

    for j, (i, field) in enumerate(tqdm(fields.iterrows(), desc='Assigning GridMET IDs', total=fields.shape[0])):

        if field_select:
            if str(field[feature_id]) not in field_select:
                continue

        xx, yy = field['geometry'].centroid.x, field['geometry'].centroid.y
        lat, lon = convert_to_wgs84(xx, yy)
        fields.at[i, 'LAT'] = lat
        fields.at[i, 'LON'] = lon

        closest_fid = j

        fields.at[i, 'GFID'] = closest_fid

        if closest_fid not in gridmet_targets.keys():
            gridmet_targets[closest_fid] = {str(m): {} for m in range(1, 13)}
            gdf = gpd.GeoDataFrame({'geometry': [field['geometry'].centroid]})
            gridmet_targets[closest_fid]['lat'] = lat
            gridmet_targets[closest_fid]['lon'] = lon
            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'], nodata=np.nan)[0]['mean']
                # TODO: raise so tif/shp mismatch doesn't pass silent
                gridmet_targets[closest_fid][month].update({_var: stats})

        g = GridMet('elev', lat=fields.at[i, 'LAT'], lon=fields.at[i, 'LON'])
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV'] = elev

    fields = fields[~np.isnan(fields['GFID'])]
    print(f'Writing {fields.shape[0]} of {oshape} input features')
    fields['GFID'] = fields['GFID'].fillna(-1).astype(int)

    fields.to_file(fields_join, crs='EPSG:5071', engine='fiona')

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)
    print(f'wrote {factors_js}')


def download_gridmet(fields, gridmet_factors, gridmet_csv_dir, start=None, end=None, overwrite=False,
                     append=False, target_fields=None, feature_id='FID', return_df=False):
    if not start:
        start = '1987-01-01'
    if not end:
        end = '2021-12-31'

    fields = gpd.read_file(fields)
    fields.index = fields[feature_id]

    with open(gridmet_factors, 'r') as f:
        gridmet_factors = json.load(f)

    downloaded = {}

    for k, v in tqdm(fields.iterrows(), desc='Downloading GridMET', total=len(fields)):

        elev, existing = None, None
        out_cols = COLUMN_ORDER.copy()
        df, first = pd.DataFrame(), True

        if target_fields and str(k) not in target_fields:
            continue

        g_fid = str(int(v['GFID']))

        if g_fid in downloaded.keys():
            downloaded[g_fid].append(k)

        _file = os.path.join(gridmet_csv_dir, 'gridmet_{}.csv'.format(g_fid))
        if os.path.exists(_file) and not overwrite and not append:
            continue

        if os.path.exists(_file) and append:
            existing = pd.read_csv(_file, index_col='date', parse_dates=True)
            existing['date'] = existing.index
            target_dates = pd.date_range(start, end, freq='D')
            missing_dates = [i for i in target_dates if i not in existing.index]

            if len(missing_dates) == 0:
                return df

            else:
                start, end = missing_dates[0].strftime('%Y-%m-%d'), missing_dates[-1].strftime('%Y-%m-%d')

        r = gridmet_factors[g_fid]
        lat, lon = r['lat'], r['lon']

        for thredds_var, cols in CLIMATE_COLS.items():
            variable = cols['col']

            if not thredds_var:
                continue

            try:
                g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                s = g.get_point_timeseries()
            except OSError as e:
                print('Error on {}, {}'.format(k, e))

            df[variable] = s[thredds_var]

            if first:
                df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
                df['year'] = [i.year for i in df.index]
                df['month'] = [i.month for i in df.index]
                df['day'] = [i.day for i in df.index]
                df['centroid_lat'] = [lat for _ in range(df.shape[0])]
                df['centroid_lon'] = [lon for _ in range(df.shape[0])]
                g = GridMet('elev', lat=lat, lon=lon)
                elev = g.get_point_elevation()
                df['elev_m'] = [elev for _ in range(df.shape[0])]
                first = False

        for _var in ['eto']:  # etr
            variable = '{}_mm'.format(_var)
            out_cols.append('{}_uncorr'.format(variable))
            for month in range(1, 13):
                corr_factor = gridmet_factors[g_fid][str(month)][_var]
                idx = [i for i in df.index if i.month == month]
                df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

        df = df[out_cols]
        if existing is not None and not overwrite and append:
            df = pd.concat([df, existing], axis=0, ignore_index=False)
            df = df.sort_index()

        df.to_csv(_file, index=False)
        downloaded[g_fid] = [k]

        if return_df:
            return df


if __name__ == '__main__':
    ''''''

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')
    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')

    data = os.path.join(nv_data, 'fields_data')

    FEATURE_ID = 'OPENET_ID'

    share_data = os.path.join(root, 'swim', 'gridmet', 'gridmet_corrected')
    gridmet_centroids = os.path.join(nv_fields_boundaries, 'Joined_Points.shp')

    shapefile_path = os.path.join(nv_fields_boundaries, 'Nevada_Agricultural_Field_Boundaries_20250214_5071.shp')
    correction_tifs = os.path.join(share_data, 'correction_surfaces_aea')

    fields_gridmet = os.path.join(nv_fields_boundaries, 'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.shp')
    gridmet_factors = os.path.join(nv_fields_boundaries, 'Nevada_Fields_with_Nearest_GFID.json')

    met = os.path.join(data, 'gridmet')

    run_zonal_stats_for_fields(fields_gridmet, gridmet_centroids, correction_tifs,
                               gridmet_factors, field_select=None, feature_id='FID')

    download_gridmet(fields_gridmet, gridmet_factors, met, start='1980-01-01', end='2024-12-31',
                     overwrite=False, feature_id=FEATURE_ID, target_fields=None)
# ========================= EOF ====================================================================
