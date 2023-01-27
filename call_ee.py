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
BOUNDARIES = 'users/dgketchum/boundaries'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'

BASIN_STATES = ['AZ', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

PROPS = sorted(
    ['anm_ppt_gs', 'anm_ppt_lspr', 'anm_ppt_spr', 'anm_ppt_sum', 'anm_ppt_win', 'anm_ppt_wy', 'anm_ppt_wy_et',
     'anm_ppt_wy_spr', 'anm_temp_gs', 'anm_temp_lspr', 'anm_temp_spr', 'anm_temp_sum', 'anm_temp_win',
     'anm_temp_wy', 'anm_temp_wy_et', 'anm_temp_wy_spr', 'aspect', 'awc', 'clay', 'cwd_gs', 'cwd_lspr', 'cwd_spr',
     'cwd_sum', 'cwd_win', 'cwd_wy', 'cwd_wy_et', 'cwd_wy_spr', 'elevation', 'etr_gs', 'etr_lspr', 'etr_spr',
     'etr_sum', 'etr_win', 'etr_wy', 'etr_wy_et', 'etr_wy_spr', 'ksat',
     'lat', 'lon',
     'ppt_gs', 'ppt_lspr', 'ppt_spr', 'ppt_sum', 'ppt_win', 'ppt_wy', 'ppt_wy_et', 'ppt_wy_spr', 'sand', 'slope',
     'tmp_gs', 'tmp_lspr', 'tmp_spr', 'tmp_sum', 'tmp_win', 'tmp_wy', 'tmp_wy_et', 'tmp_wy_spr', 'tpi_1250',
     'tpi_150', 'tpi_250'])

SUBPROPS = ['aspect', 'elevation', 'slope']


def get_uncultivated_points(tables, file_prefix):
    fc = ee.FeatureCollection(tables)
    irr_coll = ee.ImageCollection(RF_ASSET)
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

    plot_sample_regions = bands.sampleRegions(
        collection=fc,
        scale=30,
        tileScale=16)

    desc = '{}'.format(file_prefix)
    task = ee.batch.Export.table.toCloudStorage(
        plot_sample_regions,
        description=desc,
        selectors=['id', 'uncult', 'cdl', 'nlcd', 'slope'],
        bucket='wudr',
        fileNamePrefix='{}'.format(file_prefix),
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


def export_pixels(roi, clip, years, shardfile):
    gdf = gpd.read_file(shardfile)
    point_fids = list(gdf['id'].values)

    points_fc = ee.FeatureCollection('users/dgketchum/grids/shard_wa_18JAN2023')

    roi = ee.FeatureCollection(roi)
    clip = ee.FeatureCollection(clip).geometry()

    for yr in years:
        et_cols = ['et_{}_{}'.format(yr, mm) for mm in range(4, 11)]
        stack = stack_bands(yr, roi).select(SUBPROPS)
        stack = stack.clip(clip)

        mask = ee.Image().byte()
        mask = mask.paint(roi, 1)

        stack = stack.addBands(mask.rename('roi'))

        ct = 0
        geometry_sample = ee.ImageCollection([])
        first = True
        for i in range(100, len(point_fids), 5):
            for j in range(i, i + 5):
                try:
                    loc = int(point_fids[j])
                    if first:
                        print(loc)
                        first = False
                    out_filename = '{}_{}'.format(str(yr), loc)
                    point = points_fc.filter(ee.Filter.eq('id', int(loc)))
                    sample = stack.sample(
                        region=point.geometry(),
                        scale=1000,
                        tileScale=16,
                        dropNulls=True)
                    geometry_sample = geometry_sample.merge(sample)
                    ct += 1
                except IndexError:
                    break

        task = ee.batch.Export.table.toCloudStorage(
            collection=geometry_sample,
            bucket='wudr',
            description=out_filename,
            fileNamePrefix=out_filename,
            fileFormat='TFRecord',
            selectors=PROPS + et_cols)
        task.start()
        exit()


def export_classification(extracts, asset_root, region, years, bag_fraction=0.5, min_irr_years=5,
                          irr_mask=True, gridmet_res=False, indirect=False, glob='bands'):
    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_irr_years)
    proj = ee.Projection('EPSG:5071').getInfo()

    for yr in years:
        targets, input_props = ['et_{}_{}'.format(yr, mm) for mm in range(4, 11)], PROPS
        roi = ee.FeatureCollection(region)
        input_bands = stack_bands(yr, roi, gridmet_res=gridmet_res)

        if not gridmet_res:
            input_bands = input_bands.reproject(crs=proj['crs'], scale=30)

        for target, month in zip(targets, range(4, 11)):
            cols = input_props + [target]
            table_ = os.path.join(extracts, 'bands_29DEC2022_{}'.format(yr))
            fc = ee.FeatureCollection(table_)
            fc = fc.filter(ee.Filter.eq('STUSPS', 'WA'))
            elem = fc.first().getInfo()
            fc = fc.select(cols)

            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=150,
                bagFraction=bag_fraction).setOutputMode('REGRESSION')

            trained_model = classifier.train(fc, target, input_props)

            image_stack = input_bands.select(input_props)

            desc = 'eff_ppt_{}_{}'.format(yr, month)

            classified_img = image_stack.unmask().classify(trained_model).float().set({
                'system:index': ee.Date('{}-{}-01'.format(yr, month)).format('YYYYMMdd'),
                'system:time_start': ee.Date('{}-{}-01'.format(yr, month)).millis(),
                'system:time_end': ee.Date('{}-{}-{}'.format(yr, month, monthrange(yr, month)[1])).millis(),
                'date_ingested': str(date.today()),
                'image_name': desc,
                'training_data': extracts,
                'bag_fraction': bag_fraction,
                'target': target})

            if irr_mask:
                irr = irr_coll.filterDate('{}-01-01'.format(yr),
                                          '{}-12-31'.format(yr)).select('classification').mosaic()
                irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
                classified_img = classified_img.mask(irr_mask)

            if indirect:
                classified_img = classified_img.multiply(input_bands.select('ppt_wy_et').multiply(0.001))

            classified_img = classified_img.rename('eff_ppt')
            classified_img = classified_img.clip(roi.geometry())

            asset_id = os.path.join(asset_root, desc)

            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description=desc,
                assetId=asset_id,
                pyramidingPolicy={'.default': 'mean'},
                maxPixels=1e13)

            task.start()
            print(asset_id, target)


def request_band_extract(file_prefix, points_layer, region, years, scale=1000):
    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)
    for yr in years:
        for st in BASIN_STATES:
            if st != 'WA':
                continue
            stack = stack_bands(yr, roi, scale)

            state_bound = ee.FeatureCollection(os.path.join(BOUNDARIES, st))
            stack = stack.clip(state_bound)

            st_points = points.filterMetadata('STUSPS', 'equals', st)

            # geo = ee.FeatureCollection(ee.Geometry.Point(-120.4404, 47.0072))

            plot_sample_regions = stack.sampleRegions(
                collection=st_points,
                scale=scale,
                tileScale=16)

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

    et = irr_et_data(yr, resolution)

    season = et.reduce(ee.Reducer.sum()).rename('et_gs')
    ratio = season.divide(input_bands.select('ppt_wy_et')).rename('ratio')
    input_bands = input_bands.addBands([et, ratio, season, awc, clay, ksat, sand])
    input_bands = input_bands.clip(roi)

    proj = ee.Projection('EPSG:4326')
    input_bands = input_bands.reproject(crs=proj.crs(), scale=resolution)

    return input_bands


def irr_et_data(yr, scale):
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)
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

        annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
        et_coll = annual_coll_.filter(ee.Filter.date(s, e))
        et_umrb = et_coll.sum().clip(umrb_clip.geometry())

        names.append('et_{}'.format(month))
        et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
        et = et_sum.multiply(0.00001)

        # TODO make this an optional argument
        # proj = ee.Projection('EPSG:4326').getInfo()
        # crs = proj['crs']
        # et = et.setDefaultProjection(proj)
        # et = et.reproject(crs=crs, scale=scale)
        # et = et.reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True)

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


def extract_ndvi_change(tables, bucket, features=None):
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))

    roi = get_geomteries()[-1]
    early = landsat_masked(1987, 1991, 182, 243, roi)
    late = landsat_masked(2017, 2021, 182, 243, roi)
    early_mean = ee.Image(early.map(lambda x: x.normalizedDifference(['B5', 'B4'])).median())
    late_mean = ee.Image(late.map(lambda x: x.normalizedDifference(['B5', 'B4'])).median())
    ndvi_diff = late_mean.subtract(early_mean).rename('nd_diff')

    dataset = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.date('2013-01-01', '2017-12-31'))
    cultivated = dataset.select('cultivated').mode()

    ndvi_cult = ndvi_diff.mask(cultivated.eq(2))
    increase = ndvi_cult.gt(0.2).rename('gain')
    decrease = ndvi_cult.lt(-0.2).rename('loss')
    change = increase.addBands([decrease])
    change = change.mask(cultivated)
    change = change.multiply(ee.Image.pixelArea())

    fc = fc.filterMetadata('STAID', 'equals', '13269000')

    change = change.reduceRegions(collection=fc,
                                  reducer=ee.Reducer.sum(),
                                  scale=30)
    p = change.first().getInfo()
    out_desc = 'ndvi_change'
    selectors = ['STAID', 'loss', 'gain']
    task = ee.batch.Export.table.toCloudStorage(
        change,
        description=out_desc,
        bucket=bucket,
        fileNamePrefix=out_desc,
        fileFormat='CSV',
        selectors=selectors)
    task.start()
    print(out_desc)


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

    pts = 'users/dgketchum/expansion/pts_29DEC2022'
    # get_uncultivated_points(pts, 'sample_pts_29DEC2022')

    extract_loc = 'users/dgketchum/expansion/tables_29DEC2022/'
    ic = 'users/dgketchum/expansion/eff_ppt_indir'
    # export_classification(extract_loc, ic, clip, years_, irr_mask=False, gridmet_res=True)
    clip = 'users/dgketchum/expansion/columbia_basin'
    mask = 'users/dgketchum/expansion/test_buff_env'
    shard = '/media/research/IrrigationGIS/expansion/shapefiles/grids/shard_wa.shp'
    # export_pixels(mask, clip, years=[2020], shardfile=shard)

# ========================= EOF ====================================================================
