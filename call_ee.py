import os
import sys
from datetime import date
from calendar import monthrange

import ee
import pandas as pd

from cdl import get_cdl
from ee_utils import get_world_climate

sys.path.insert(0, os.path.abspath('..'))

sys.setrecursionlimit(2000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
BOUNDARIES = 'users/dgketchum/boundaries'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'

# generalize this
TARGETS = ['et_{yr}_4', 'et_{yr}_5', 'et_{yr}_6', 'et_{yr}_7', 'et_{yr}_8', 'et_{yr}_9', 'et_{yr}_10']

FEATURES = ['aspect', 'awc', 'clay', 'cwd_wy_smr', 'etr_{pyr}_10', 'etr_{pyr}_11', 'etr_{pyr}_12',
            'etr_{yr}_1', 'etr_{yr}_2', 'etr_{yr}_3', 'etr_{yr}_4', 'etr_{yr}_5', 'etr_{yr}_6', 'etr_{yr}_7',
            'etr_{yr}_8', 'etr_{yr}_9', 'ksat', 'pet_tot_wy_smr', 'ppt_{pyr}_10', 'ppt_{pyr}_11',
            'ppt_{pyr}_12', 'ppt_{yr}_1', 'ppt_{yr}_2', 'ppt_{yr}_3', 'ppt_{yr}_4', 'ppt_{yr}_5', 'ppt_{yr}_6',
            'ppt_{yr}_7', 'ppt_{yr}_8', 'ppt_{yr}_9', 'prec_tot_wy_smr', 'sand', 'slope', 'tmp_wy_smr', 'tpi_1250',
            'tpi_150', 'tpi_250', 'ppt']


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


def export_classification(extract, asset_root, region, years, bag_fraction=0.5, min_irr_years=5, irr_mask=True):
    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_irr_years)

    for yr in years:

        targets = [t.format(yr=yr) for t in TARGETS]
        input_props = []
        for f in FEATURES:
            try:
                feat = f.format(yr=yr)
            except KeyError:
                feat = f.format(pyr=yr - 1)
            except KeyError:
                feat = f
            input_props.append(feat)

        extract_path = extract.format(yr)
        roi = ee.FeatureCollection(region)
        input_bands = stack_bands(yr, roi)

        for target, month in zip(targets, range(4, 11)):

            cols = input_props + [target]
            fc = ee.FeatureCollection(extract_path).select(cols)

            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=150,
                bagFraction=bag_fraction).setOutputMode('REGRESSION')

            trained_model = classifier.train(fc, target, input_props)

            if irr_mask:
                irr = irr_coll.filterDate('{}-01-01'.format(yr),
                                          '{}-12-31'.format(yr)).select('classification').mosaic()
                irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            image_stack = input_bands.select(input_props + [target])

            desc = 'eff_ppt_{}_{}'.format(yr, month)

            classified_img = image_stack.unmask().classify(trained_model).float().set({
                'system:index': ee.Date('{}-{}-01'.format(yr, month)).format('YYYYMMdd'),
                'system:time_start': ee.Date('{}-{}-01'.format(yr, month)).millis(),
                'system:time_end': ee.Date('{}-{}-{}'.format(yr, month, monthrange(yr, month)[1])).millis(),
                'date_ingested': str(date.today()),
                'image_name': desc,
                'training_data': extract_path,
                'bag_fraction': bag_fraction,
                'target': target})

            classified_img = classified_img.clip(roi.geometry())

            if irr_mask:
                classified_img = classified_img.mask(irr_mask)

            classified_img = classified_img.rename('eff_ppt')
            asset_id = os.path.join(asset_root, target)
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description=desc,
                assetId=asset_id,
                scale=30,
                pyramidingPolicy={'.default': 'mean'},
                maxPixels=1e13)

            task.start()
            print(asset_id, target)


def request_band_extract(file_prefix, points_layer, region, years, filter_bounds=False):
    """
    Extract raster values from a points kml file in Fusion Tables. Send annual extracts .csv to GCS wudr bucket.
    Concatenate them using map.tables.concatenate_band_extract().
    :param region:
    :param points_layer:
    :param file_prefix:
    :param filter_bounds: Restrict extract to within a geographic extent.
    :return:
    """
    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)
    for yr in years:
        stack = stack_bands(yr, roi)
        if filter_bounds:
            points = points.filterBounds(roi)

        plot_sample_regions = stack.sampleRegions(
            collection=points,
            scale=30,
            tileScale=16)

        plot_sample_regions = plot_sample_regions.randomColumn('rand')
        one = plot_sample_regions.filter(ee.Filter.lt('rand', 0.5))
        two = plot_sample_regions.filter(ee.Filter.gte('rand', 0.5))

        desc = '{}_{}'.format(file_prefix, yr)
        task = ee.batch.Export.table.toCloudStorage(
            one,
            description=desc,
            bucket='wudr',
            fileNamePrefix='{}_a_{}'.format(file_prefix, yr),
            fileFormat='CSV')
        task.start()

        task = ee.batch.Export.table.toCloudStorage(
            two,
            description=desc,
            bucket='wudr',
            fileNamePrefix='{}_b_{}'.format(file_prefix, yr),
            fileFormat='CSV')
        task.start()
        print(desc)


def stack_bands(yr, roi):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    proj = ee.Projection('EPSG:5071').getInfo()

    input_bands = ee.Image.pixelLonLat().rename(['lon', 'lat']).resample('bilinear').reproject(crs=proj['crs'],
                                                                                               scale=30)
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect').reduceResolution(
        ee.Reducer.mean()).reproject(crs=proj['crs'], scale=30)

    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    input_bands = input_bands.addBands([terrain, tpi_1250, tpi_250, tpi_150])

    water_year_start = '{}-10-01'.format(yr - 1)
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    for s, e, n, m in [(spring_s, late_spring_e, 'spr', (3, 8)),
                       (water_year_start, spring_e, 'wy_spr', (10, 5)),
                       (water_year_start, summer_e, 'wy_smr', (10, 9)),
                       (spring_e, fall_s, 'gs', (5, 10)),
                       (spring_s, spring_e, 'spr', (3, 5)),
                       (late_spring_s, late_spring_e, 'lspr', (5, 7)),
                       (summer_s, summer_e, 'sum', (7, 9)),
                       (winter_s, winter_e, 'win', (1, 3))]:

        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            roi).filterDate(s, e).select('pr', 'eto', 'tmmn', 'tmmx')

        temp = ee.Image(gridmet.select('tmmn').mean().add(gridmet.select('tmmx').mean()
                                                          .divide(ee.Number(2))).rename('tmp_{}'.format(n)))
        temp = temp.resample('bilinear').reproject(crs=proj['crs'], scale=30)

        ai_sum = gridmet.select('pr', 'eto').reduce(ee.Reducer.sum()).rename(
            'prec_tot_{}'.format(n), 'pet_tot_{}'.format(n)).resample('bilinear').reproject(crs=proj['crs'],
                                                                                            scale=30)
        wd_estimate = ai_sum.select('prec_tot_{}'.format(n)).subtract(ai_sum.select(
            'pet_tot_{}'.format(n))).rename('cwd_{}'.format(n)).resample('bilinear').reproject(crs=proj['crs'],
                                                                                               scale=30)
        worldclim_prec = get_world_climate(proj=proj, months=m, param='prec')
        anom_prec = ai_sum.select('prec_tot_{}'.format(n)).subtract(worldclim_prec)
        worldclim_temp = get_world_climate(proj=proj, months=m, param='tavg')
        anom_temp = temp.subtract(worldclim_temp).rename('an_temp_{}'.format(n))

        prec = ai_sum.select('prec_tot_{}'.format(n))
        etr = ai_sum.select('pet_tot_{}'.format(n))

        if n == 'wy_smr':
            ppt = ai_sum.select('prec_tot_{}'.format(n)).rename('ppt')
            input_bands = input_bands.addBands(ppt)

        input_bands = input_bands.addBands([temp, ai_sum, wd_estimate, prec, etr, anom_prec, anom_temp])

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('landcover').reproject(crs=proj['crs'], scale=30).rename('nlcd')
    cdl_cult, cdl_crop, cdl_simple = get_cdl(yr)
    input_bands = input_bands.addBands([nlcd, cdl_cult, cdl_crop, cdl_simple])

    awc = ee.Image('projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite').rename('awc')
    clay = ee.Image('projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite').rename('clay')
    ksat = ee.Image('projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite').rename('ksat')
    sand = ee.Image('projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite').rename('sand')

    et = irr_et_data(yr)
    input_bands = input_bands.addBands([et, ppt, etr, awc, clay, ksat, sand])
    input_bands = input_bands.clip(roi)
    standard_names = []
    temp_ct = 1
    prec_ct = 1
    names = input_bands.bandNames().getInfo()

    for name in names:
        if 'tavg' in name and 'tavg' in standard_names:
            standard_names.append('tavg_{}'.format(temp_ct))
            temp_ct += 1
        elif 'prec' in name and 'prec' in standard_names:
            standard_names.append('prec_{}'.format(prec_ct))
            prec_ct += 1
        else:
            standard_names.append(name)

    input_bands = input_bands.rename(standard_names)
    return input_bands


def irr_et_data(yr):
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)
    bands = None

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

        et_str = 'et_{}_{}'.format(yr, month)
        et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
        et = et_sum.rename(et_str)

        if month == 4:
            bands = et
        else:
            bands = bands.addBands([et])

    return bands


def extract_point_data(tables, bucket, years, description, features=None, min_years=0, debug=False):
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
    # ndvi_diff = ndvi_diff.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

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

    points_ = 'users/dgketchum/expansion/points/sample_pts_29DEC2022_filtered'
    bucket = 'wudr'
    clip = 'users/dgketchum/expansion/study_area_dissolve'
    # clip = 'users/dgketchum/expansion/snake_plain'
    years_ = [x for x in range(1987, 2022)]
    years_.reverse()
    request_band_extract('bands_29DEC2022', points_, clip, years_)

    pts = 'users/dgketchum/expansion/pts_29DEC2022'
    # get_uncultivated_points(pts, 'sample_pts_29DEC2022')

    extract_ = 'users/dgketchum/expansion/tables_28DEC2022/bands_28DEC2022_{}'
    ic = 'users/dgketchum/expansion/eff_ppt_snake'

    years_ = [2020]
    # export_classification(extract_, ic, clip, years_, irr_mask=False)

# ========================= EOF ====================================================================
