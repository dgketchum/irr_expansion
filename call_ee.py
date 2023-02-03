import os
import sys
from datetime import date
from calendar import monthrange

import ee
import pandas as pd
import geopandas as gpd

from cdl import get_cdl
from ee_utils import get_world_climate

sys.path.insert(0, os.path.abspath('..'))

sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
KLAMATH_CLIP = 'users/dgketchum/boundaries/klamath_rogue_buff'
BOUNDARIES = 'users/dgketchum/boundaries'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'

BASIN_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

PROPS = sorted(
    ['anm_ppt_gs', 'anm_ppt_lspr', 'anm_ppt_spr', 'anm_ppt_sum', 'anm_ppt_win', 'anm_ppt_wy', 'anm_ppt_wy_et',
     'anm_ppt_wy_spr', 'anm_temp_gs', 'anm_temp_lspr', 'anm_temp_spr', 'anm_temp_sum', 'anm_temp_win',
     'anm_temp_wy', 'anm_temp_wy_et', 'anm_temp_wy_spr', 'aspect', 'awc', 'clay', 'cwd_gs', 'cwd_lspr', 'cwd_spr',
     'cwd_sum', 'cwd_win', 'cwd_wy', 'cwd_wy_et', 'cwd_wy_spr', 'elevation', 'etr_gs', 'etr_lspr', 'etr_spr',
     'etr_sum', 'etr_win', 'etr_wy', 'etr_wy_et', 'etr_wy_spr', 'ksat',
     'ppt_gs', 'ppt_lspr', 'ppt_spr', 'ppt_sum', 'ppt_win', 'ppt_wy', 'ppt_wy_et', 'ppt_wy_spr', 'sand', 'slope',
     'tmp_gs', 'tmp_lspr', 'tmp_spr', 'tmp_sum', 'tmp_win', 'tmp_wy', 'tmp_wy_et', 'tmp_wy_spr', 'tpi_1250',
     'tpi_150', 'tpi_250'])

SUBPROPS = ['aspect', 'elevation', 'slope']


def get_uncultivated_points(tables, file_prefix):
    fc = ee.FeatureCollection(tables)
    irr_coll = ee.ImageCollection(RF_ASSET)

    for st in BASIN_STATES:
        coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
        remap = coll.map(lambda img: img.eq(2))
        remap = remap.sum().rename('uncult')
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD').select('landcover').first().rename('nlcd')
        cdl_cult, cdl_crop, cdl_simple = get_cdl(2020)
        cdl_crop = cdl_crop.rename('cdl')
        ned = ee.Image('USGS/NED')
        coords = ee.Image.pixelLonLat().rename(['lon', 'lat'])
        slope = ee.Terrain.products(ned).select('slope')
        bands = remap.addBands([cdl_crop, nlcd, slope, coords])

        st_points = fc.filterMetadata('STUSPS', 'equals', st)

        plot_sample_regions = bands.sampleRegions(
            collection=st_points,
            scale=30,
            tileScale=16)

        desc = '{}_{}'.format(st, file_prefix)
        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description=desc,
            selectors=['id', 'uncult', 'cdl', 'nlcd', 'slope'],
            bucket='wudr',
            fileNamePrefix=desc,
            fileFormat='CSV')

        print(desc)
        task.start()


def get_geomteries():
    uinta = ee.Geometry.Polygon([[-109.45865297030385, 40.23092798019809],
                                 [-109.2142071695226, 40.23092798019809],
                                 [-109.2142071695226, 40.463280555650584],
                                 [-109.45865297030385, 40.463280555650584],
                                 [-109.45865297030385, 40.23092798019809]])

    test_point = ee.Geometry.Point(-111.19417486580859, 45.73357569538925)

    return uinta, test_point


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


def extract_point_data(tables, bucket, years, description, debug=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param features:
    :param bucket:
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    fc = ee.FeatureCollection(tables)
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    irr_coll = ee.ImageCollection(RF_ASSET)
    select_ = ['id']

    for yr in years:
        for month in range(4, 11):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr_str = 'irr_{}'.format(yr)
            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr = irr.rename(irr_str)

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_str = 'et_{}_{}'.format(yr, month)
            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
            et = et_sum.rename(et_str)

            select_.append(et_str)
            if month == 4:
                select_.append(irr_str)
                bands = irr.addBands([et])
            else:
                bands = bands.addBands([et])

        if debug:
            pt = bands.sample(region=get_geomteries()[2],
                              numPixels=1,
                              scale=30)
            p = pt.first().getInfo()['properties']
            print('propeteries {}'.format(p))

        data = bands.reduceRegions(collection=fc,
                                   reducer=ee.Reducer.sum(),
                                   scale=30)

        out_desc = 'pts_{}_{}'.format(description, yr)
        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=out_desc,
            bucket=bucket,
            fileNamePrefix=out_desc,
            fileFormat='CSV',
            selectors=select_)
        task.start()
        print(out_desc)


def request_band_extract(file_prefix, points_layer, region, years, scale, clamp_et=False):
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

            # st_points = ee.FeatureCollection(ee.Geometry.Point(-120.260594, 46.743666))

            plot_sample_regions = stack.sampleRegions(
                collection=st_points,
                scale=scale,
                tileScale=16,
                projection=ee.Projection('EPSG:4326'))

            if clamp_et:
                plot_sample_regions = plot_sample_regions.filter(ee.Filter.lt('ratio', ee.Number(1.0)))

            # point = plot_sample_regions.first().getInfo()

            desc = '{}_{}_{}'.format(file_prefix, st, yr)
            task = ee.batch.Export.table.toCloudStorage(
                plot_sample_regions,
                description=desc,
                bucket='wudr',
                fileNamePrefix=desc,
                fileFormat='TFRecord')

            task.start()
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


def landsat_c2_sr(input_img):
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start'])

    return image


def landsat_masked(start_year, end_year, doy_start, doy_end, roi):
    start = '{}-01-01'.format(start_year)
    end_date = '{}-01-01'.format(end_year)

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
    return lsSR_masked


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()

    pts = 'users/dgketchum/expansion/points/uncult_add_2FEB2023'
    # get_uncultivated_points(pts, 'uncult_add_2FEB2023')

    points_ = 'users/dgketchum/expansion/points/points_nonforest_2FEB2023'
    bucket = 'wudr'
    years_ = [x for x in range(1987, 2022)]
    clip = 'users/dgketchum/expansion/study_area_klamath'
    request_band_extract('bands_2FEB2023', points_, clip, years_, 30, clamp_et=True)

# ========================= EOF ====================================================================
