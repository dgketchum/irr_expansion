import os
import sys
import time
from calendar import monthrange

import ee

from utils.ee_utils import get_world_climate, get_cdl

sys.path.insert(0, os.path.abspath('../..'))

sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
KLAMATH_CLIP = 'users/dgketchum/boundaries/klamath_rogue_buff'
BOUNDARIES = 'users/dgketchum/boundaries'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

PROPS = sorted(
    ['anm_ppt_gs', 'anm_ppt_lspr', 'anm_ppt_spr', 'anm_ppt_sum', 'anm_ppt_win', 'anm_ppt_wy', 'anm_ppt_wy_et',
     'anm_ppt_wy_spr', 'anm_temp_gs', 'anm_temp_lspr', 'anm_temp_spr', 'anm_temp_sum', 'anm_temp_win',
     'anm_temp_wy', 'anm_temp_wy_et', 'anm_temp_wy_spr', 'aspect', 'awc', 'clay', 'cwd_gs', 'cwd_lspr', 'cwd_spr',
     'cwd_sum', 'cwd_win', 'cwd_wy', 'cwd_wy_et', 'cwd_wy_spr', 'elevation', 'etr_gs', 'etr_lspr', 'etr_spr',
     'etr_sum', 'etr_win', 'etr_wy', 'etr_wy_et', 'etr_wy_spr', 'ksat',
     'ppt_gs', 'ppt_lspr', 'ppt_spr', 'ppt_sum', 'ppt_win', 'ppt_wy', 'ppt_wy_et', 'ppt_wy_spr', 'sand', 'slope',
     'tmp_gs', 'tmp_lspr', 'tmp_spr', 'tmp_sum', 'tmp_win', 'tmp_wy', 'tmp_wy_et', 'tmp_wy_spr', 'tpi_1250',
     'tpi_150', 'tpi_250'])


def get_points_cultivation_data(state, tables, file_prefix, target='uncult', join_col='id',
                                extra=True, filter_state=False):
    fc = ee.FeatureCollection(tables)
    fc = fc.filterBounds(ee.FeatureCollection('users/dgketchum/expansion/study_area_klamath').geometry())
    if filter_state:
        fc = fc.filterMetadata('STUSPS', 'equals', state)
    irr_coll = ee.ImageCollection(RF_ASSET)

    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')

    if target == 'uncult':
        remap = coll.map(lambda img: img.eq(2))
        bands = remap.sum().rename('uncult')
    elif target == 'irr':
        remap = coll.map(lambda img: img.eq(0))
        bands = remap.sum().rename('irr')
    else:
        raise NotImplementedError

    selectors_ = [join_col, target]

    if extra:
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD').select('landcover').first().rename('nlcd')
        cdl_cult, cdl_crop, cdl_simple = get_cdl(2020)
        cdl_crop = cdl_crop.rename('cdl')
        ned = ee.Image('USGS/NED')
        coords = ee.Image.pixelLonLat().rename(['lon', 'lat'])
        slope = ee.Terrain.products(ned).select('slope')
        bands = bands.addBands([cdl_crop, nlcd, slope, coords])
        selectors_ = [join_col, target, 'cdl', 'nlcd', 'slope']

    plot_sample_regions = bands.sampleRegions(
        collection=fc,
        scale=30,
        tileScale=16)

    desc = '{}_{}'.format(state, file_prefix)
    task = ee.batch.Export.table.toCloudStorage(
        plot_sample_regions,
        description=desc,
        selectors=selectors_,
        bucket='wudr',
        fileNamePrefix=desc,
        fileFormat='CSV')

    print(desc)
    task.start()


def stack_bands(yr, roi, resolution, **scale_factors):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    non_scale = {'climate': 1, 'terrain': 1, 'soil': 1}
    if not scale_factors:
        scale_factors = non_scale

    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect')
    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    terrain = terrain.addBands([tpi_1250, tpi_250, tpi_150])

    input_bands = terrain.multiply(scale_factors['terrain'])

    water_year_start = '{}-10-01'.format(yr - 1)
    et_water_year_start, et_water_year_end = '{}-11-01'.format(yr - 1), '{}-11-01'.format(yr)
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    for s, e, n, m in [(water_year_start, spring_e, 'wy_spr', (10, 5)),
                       (water_year_start, summer_e, 'wy', (10, 9)),
                       (et_water_year_start, et_water_year_end, 'wy_et', (11, 11)),
                       (spring_e, fall_s, 'gs', (5, 10)),
                       (spring_s, spring_e, 'spr', (3, 5)),
                       (late_spring_s, late_spring_e, 'lspr', (5, 7)),
                       (summer_s, summer_e, 'sum', (7, 9)),
                       (winter_s, winter_e, 'win', (1, 3))]:

        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            roi).filterDate(s, e).select('pr', 'etr', 'tmmn', 'tmmx')

        tmx = gridmet.select('tmmx').reduce(ee.Reducer.mean())
        tmn = gridmet.select('tmmn').reduce(ee.Reducer.mean())
        temp = ee.Image([tmx, tmn]).reduce(ee.Reducer.mean())
        temp = temp.subtract(273.15).multiply(0.1).rename('tmp_{}'.format(n))

        ppt = gridmet.select('pr').reduce(ee.Reducer.sum())
        ppt = ppt.multiply(scale_factors['climate']).rename('ppt_{}'.format(n))
        etr = gridmet.select('etr').reduce(ee.Reducer.sum())
        etr = etr.multiply(scale_factors['climate']).rename('etr_{}'.format(n))

        wd_estimate = ppt.subtract(etr).rename('cwd_{}'.format(n))

        worldclim_prec = get_world_climate(months=m, param='prec').rename('wrldclmppt_{}'.format(n))
        anom_prec = ppt.subtract(worldclim_prec).rename('anm_ppt_{}'.format(n)).multiply(scale_factors['climate'])

        worldclim_temp = get_world_climate(months=m, param='tavg').rename('wrldclmtmp_{}'.format(n))
        anom_temp = temp.subtract(worldclim_temp)
        anom_temp = anom_temp.multiply(scale_factors['climate']).rename('anm_temp_{}'.format(n))

        input_bands = input_bands.addBands([temp, wd_estimate, ppt, etr, anom_prec, anom_temp])

    awc = ee.Image('users/dgketchum/soils/ssurgo_AWC_WTA_0to152cm_composite')
    awc = awc.multiply(scale_factors['soil']).rename('awc')
    clay = ee.Image('users/dgketchum/soils/ssurgo_Clay_WTA_0to152cm_composite')
    clay = clay.multiply(scale_factors['soil']).rename('clay')
    ksat = ee.Image('users/dgketchum/soils/ssurgo_Ksat_WTA_0to152cm_composite')
    ksat = ksat.multiply(scale_factors['soil']).rename('ksat')
    sand = ee.Image('users/dgketchum/soils/ssurgo_Sand_WTA_0to152cm_composite')
    sand = sand.multiply(scale_factors['soil']).rename('sand')

    et = irr_et_data(yr)

    season = et.reduce(ee.Reducer.sum()).rename('et_gs')
    ratio = season.divide(input_bands.select('ppt_wy_et')).rename('ratio')
    input_bands = input_bands.addBands([et, ratio, season, awc, clay, ksat, sand])
    input_bands = input_bands.clip(roi)

    proj = ee.Projection('EPSG:4326')
    input_bands = input_bands.reproject(crs=proj.crs(), scale=resolution)

    return input_bands


def irr_et_data(yr):
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)
    klamath_clip = ee.FeatureCollection(KLAMATH_CLIP)
    bands = None
    names = []
    for month in range(4, 11):
        s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
        end_day = monthrange(yr, month)[1]
        e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

        annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
            ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
        et_coll = annual_coll.filter(ee.Filter.date(s, e))
        et_cmb = et_coll.sum().clip(cmb_clip.geometry())

        annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
            ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
            ee.ImageCollection('users/dpendergraph/ssebop/corb'))
        et_coll = annual_coll.filter(ee.Filter.date(s, e))
        et_corb = et_coll.sum().clip(corb_clip.geometry())

        annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/klamath').merge(
            ee.ImageCollection('users/dpendergraph/ssebop/klamath'))
        et_coll = annual_coll.filter(ee.Filter.date(s, e))
        et_klam = et_coll.sum().clip(klamath_clip.geometry())

        annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
        et_coll = annual_coll_.filter(ee.Filter.date(s, e))
        et_umrb = et_coll.sum().clip(umrb_clip.geometry())

        names.append('et_{}'.format(month))
        et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb, et_klam]).mosaic()
        et = et_sum.multiply(0.00001)

        if month == 4:
            bands = et
        else:
            bands = bands.addBands([et])

    bands = bands.rename(names)
    return bands


def extract_points_data(file_prefix, points_layer, region, years, scale, clamp_et=False):
    ee.Initialize()
    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)

    for st in BASIN_STATES:
        for yr in years:

            scale_feats = {'climate': 0.001, 'soil': 0.01, 'terrain': 0.001}
            stack = stack_bands(yr, roi, scale, **scale_feats)

            state_bound = ee.FeatureCollection(os.path.join(BOUNDARIES, st))
            stack = stack.clip(state_bound)
            st_points = points.filterMetadata('STUSPS', 'equals', st)

            plot_sample_regions = stack.sampleRegions(
                collection=st_points,
                scale=scale,
                tileScale=16,
                projection=ee.Projection('EPSG:4326'))

            if clamp_et:
                plot_sample_regions = plot_sample_regions.filter(ee.Filter.lt('ratio', ee.Number(1.0)))

            desc = '{}_{}_{}'.format(file_prefix, st, yr)
            task = ee.batch.Export.table.toCloudStorage(
                plot_sample_regions,
                description=desc,
                bucket='wudr',
                fileNamePrefix=desc,
                fileFormat='TFRecord')

            ee_task_start(task)
            print(desc)

            if yr == 2021:
                desc = '{}_{}_aux_{}'.format(file_prefix, st, yr)
                task = ee.batch.Export.table.toCloudStorage(
                    plot_sample_regions,
                    description=desc,
                    bucket='wudr',
                    fileNamePrefix=desc,
                    fileFormat='CSV')
                task.start()


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


def ee_task_start(task, n=6):
    """Make an exponential backoff Earth Engine request, credit cgmorton"""
    for i in range(1, n):
        try:
            task.start()
            break
        except Exception as e:
            print('    Resending query ({}/{})'.format(i, n))
            time.sleep(i ** 2)

    return task


if __name__ == '__main__':
    is_authorized()

    pts = 'users/dgketchum/points/validation_intial_28FEB2023'
    for s in BASIN_STATES:
        get_points_cultivation_data(s, pts, 'validation_intial_28FEB2023', target='uncult',
                                    join_col='id', extra=True, filter_state=True)

    points_ = 'users/dgketchum/expansion/points/points_nonforest_2FEB2023'
    bucket = 'wudr'
    years_ = [x for x in range(1987, 2022)]
    clip = 'users/dgketchum/expansion/study_area_klamath'
    # request_band_extract('bands_2FEB2023', points_, clip, years_, 30, clamp_et=True)

# ========================= EOF ====================================================================
