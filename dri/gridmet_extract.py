import concurrent.futures
import os

import numpy as np
import pandas as pd
import xarray as xr
import requests

VARIABLES = {
    'tmmn': 'air_temperature',
    'vpd': 'mean_vapor_pressure_deficit',
    'th': 'wind_from_direction',
    'sph': 'specific_humidity',
    'vs': 'wind_speed',
    'srad': 'surface_downwelling_shortwave_flux_in_air',
    'tmmx': 'air_temperature',
    'rmin': 'relative_humidity',
    'rmax': 'relative_humidity',
    'pr': 'precipitation_amount'
}


def extract_gridmet(stations, out_data, nc_dir=None, workers=8, overwrite=False, bounds=None, debug=False,
                    start_yr=1990, end_yr=2024):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 49.1)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside DAYMET NA extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    target_file_fmt = '{}_{}.nc'
    target_files = [[os.path.join(nc_dir, target_file_fmt.format(var_, year)) for var_ in VARIABLES.keys()]
                    for year in range(start_yr, end_yr)]
    years = list(range(start_yr, end_yr))

    base_url = 'https://www.northwestknowledge.net/metdata/data/'
    for year_files in target_files:
        for file_path in year_files:
            if not os.path.exists(file_path):
                filename = os.path.basename(file_path)
                url = os.path.join(base_url, filename)
                print(f"Downloading {filename} from {url}")
                download_file(url, file_path)

    if not all([os.path.exists(tp) for l in target_files for tp in l]):
        raise NotImplementedError('Missing files')

    if debug:
        for fileset, dts in zip(target_files, years):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite,
                                   ) for fileset, dts in zip(target_files, years)]
        concurrent.futures.wait(futures)


def proc_time_slice(fileset_, indexer_, date_string_, fids_, out_, overwrite_, bounds_=None):
    processed_files = []

    for file, (k, v) in zip(fileset_, VARIABLES.items()):
        ds = xr.open_dataset(file)
        ds = ds.set_coords(v)
        ds = ds.rename({v: k})
        processed_files.append(ds)

    ds = xr.merge(processed_files)
    ds = ds.rename({'day': 'time'})
    ds = ds.chunk({'time': len(ds.time.values)})
    if bounds_ is not None:
        ds = ds.sel(y=slice(bounds_[3], bounds_[1]), x=slice(bounds_[0], bounds_[2]))
    ds = ds.sel(lat=indexer_.lat, lon=indexer_.lon, method='nearest', tolerance=4000)
    df_all = ds.to_dataframe()

    ct, skip = 0, 0
    for fid in fids_:
        dst_dir = os.path.join(out_, fid)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_))

        if not os.path.exists(_file) or overwrite_:

            df_station = df_all.loc[(fid, slice(None), 3)].copy()

            if np.isnan(df_station[VARIABLES.keys()].values.sum()) > 0:
                skip += 1
                continue

            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)

            df_station['dt'] = [i.strftime('%Y%m%d') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            ct += 1
        if ct % 1000 == 0.:
            print(f'{ct} of {len(fids_)} for {date_string_}')
    print(f'wrote {ct} for {date_string_}, skipped {skip}')


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


if __name__ == '__main__':

    d = os.path.join('/home', 'ubuntu', 'data', 'IrrigationGIS')
    gridmet = os.path.join('/home', 'ubuntu', 'data', 'gridmet')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')
        gridmet = os.path.join('/data', 'gridmet')

    if not os.path.isdir(d):
        h = os.path.expanduser('~')
        d = os.path.join(h, 'data', 'IrrigationGIS')
        gridmet = os.path.join(h, 'data', 'gridmet')

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    out_files = os.path.join(gridmet, 'station_data')
    nc_files_ = os.path.join(gridmet, 'netcdf')

    print(f'{out_files} exists: {os.path.exists(out_files)}')
    print(f'{nc_files_} exists: {os.path.exists(nc_files_)}')

    bounds = (-125.0, 25.0, -67.0, 49.1)
    extract_gridmet(sites, out_files, nc_dir=nc_files_, workers=14, overwrite=False,
                    bounds=bounds, start_yr=1990, end_yr=1992)

# ========================= EOF ====================================================================
