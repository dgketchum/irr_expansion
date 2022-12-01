import os
import sys
from datetime import date
from calendar import monthrange

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

sys.setrecursionlimit(2000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
BOUNDARIES = 'users/dgketchum/boundaries'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'


def get_uncultivated_points(tables, file_prefix):
    fc = ee.FeatureCollection(tables)
    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.eq(2))
    remap = remap.sum().rename('uncult')

    plot_sample_regions = remap.sampleRegions(
        collection=fc,
        scale=30,
        tileScale=16)

    desc = '{}'.format(file_prefix)
    task = ee.batch.Export.table.toCloudStorage(
        plot_sample_regions,
        description=desc,
        selectors=['uncult', 'id'],
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


def export_classification(extract, asset_root, region, years, input_props, bag_fraction=0.5, min_irr_years=5):
    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_irr_years)

    for yr in years:
        for m in range(4, 11):
            target = 'et_{}_{}'.format(yr, m)
            cols = input_props + [target]
            extract_path = extract.format(yr)
            fc = ee.FeatureCollection(extract_path).select(cols)
            roi = ee.FeatureCollection(region)

            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=150,
                minLeafPopulation=1,
                bagFraction=bag_fraction).setOutputMode('REGRESSION')

            trained_model = classifier.train(fc, target, input_props)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            input_bands = stack_bands(yr, roi)
            image_stack = input_bands.select(input_props + [target])

            desc = 'nat_et_{}_{}'.format(yr, m)

            classified_img = image_stack.unmask().classify(trained_model).float().set({
                'system:index': ee.Date('{}-{}-01'.format(yr, m)).format('YYYYMMdd'),
                'system:time_start': ee.Date('{}-{}-01'.format(yr, m)).millis(),
                'system:time_end': ee.Date('{}-{}-{}'.format(yr, m, monthrange(yr, m)[1])).millis(),
                'date_ingested': str(date.today()),
                'image_name': desc,
                'training_data': extract,
                'bag_fraction': bag_fraction,
                'target': target})

            classified_img = classified_img.clip(roi.geometry())
            classified_img = classified_img.mask(irr_mask)
            classified_img = classified_img.rename(desc)
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description=desc,
                assetId=os.path.join(asset_root, target),
                scale=30,
                pyramidingPolicy={'.default': 'mean'},
                maxPixels=1e13)

            task.start()
            print(os.path.join(asset_root, desc))


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

        desc = '{}_{}'.format(file_prefix, yr)
        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description=desc,
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        print(desc)
        task.start()


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

    et = irr_et_data(yr)
    input_bands = input_bands.addBands([et])

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
        elif 'nd_cy' in name:
            standard_names.append('nd_max_cy')
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

    points_ = 'users/dgketchum/expansion/points/study_uncult_points'
    bucket = 'wudr'
    clip = 'users/dgketchum/expansion/study_area_dissolve'
    # request_band_extract('domain', points_, clip, [x for x in range(1987, 2019)])

    extract_ = 'users/dgketchum/expansion/tables/domain_{}'
    ic = 'users/dgketchum/expansion/naturalized_et'

    props = ['aspect', 'elevation', 'lat', 'lon', 'slope', 'tpi_1250', 'tpi_150', 'tpi_250']
    export_classification(extract_, ic, clip, [x for x in range(2000, 2016)], input_props=props)

# ========================= EOF ====================================================================
