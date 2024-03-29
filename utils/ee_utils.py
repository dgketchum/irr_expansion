from datetime import datetime, timedelta

import ee

from field_points.crop_codes import remap_cdl


def get_world_climate(months, param='prec'):
    if months[0] > months[1]:
        months = [x for x in range(months[0], 13)] + [x for x in range(1, months[1] + 1)]
    elif months[0] == months[1]:
        months = [x for x in range(1, 13)]
    else:
        months = [x for x in range(months[0], months[1] + 1)]
    months = [str(x).zfill(2) for x in months]
    assert param in ['tavg', 'prec']
    combinations = [(m, param) for m in months]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).select(param) for m, p in combinations]
    i = ee.ImageCollection(l)
    if param == 'prec':
        i = i.sum()
    else:
        i = i.mean().multiply(0.1)
    return i


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


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll).merge(l4_coll))
    return lsSR_masked


def landsat_composites(year, start, end, roi, append_name, composites_only=False):
    start_year = datetime.strptime(start, '%Y-%m-%d').year
    if start_year != year:
        year = start_year

    def evi_(x):
        return x.expression('2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))', {'NIR': x.select('B5'),
                                                                                   'RED': x.select('B4'),
                                                                                   'BLUE': x.select('B2')})

    def gi_(x):
        return x.expression('NIR / GREEN', {'NIR': x.select('B5'),
                                            'GREEN': x.select('B3')})

    bands_means = None
    lsSR_masked = landsat_masked(year, roi)
    if not composites_only:
        bands_means = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'],
                               ['B2_{}'.format(append_name),
                                'B3_{}'.format(append_name),
                                'B4_{}'.format(append_name),
                                'B5_{}'.format(append_name),
                                'B6_{}'.format(append_name),
                                'B7_{}'.format(append_name),
                                'B10_{}'.format(append_name)]
                               )).mean())

    if append_name in ['m2', 'm1', 'gs']:
        ndvi_mx = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_max_{}'.format(append_name))

        ndvi_mean = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).mean()).rename('nd_mean_{}'.format(append_name))

        ndvi = ndvi_mx.addBands([ndvi_mean])

    else:
        ndvi = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_{}'.format(append_name))

    ndwi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.normalizedDifference(['B5', 'B6'])).max()).rename('nw_{}'.format(append_name))
    evi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: evi_(x)).max()).rename('evi_{}'.format(append_name))
    gi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: gi_(x)).max()).rename('gi_{}'.format(append_name))

    if composites_only:
        bands = ndvi.addBands([ndwi, evi, gi])
    else:
        bands = bands_means.addBands([ndvi, ndwi, evi, gi])

    return bands


def get_cdl(yr):
    cultivated, crop = None, None
    cdl_years = [x for x in range(2008, 2021)]
    cultivated_years = [x for x in range(2013, 2019)]

    mode_reduce = ee.Reducer.mode()

    first = True
    for y in cultivated_years:
        image = ee.Image('USDA/NASS/CDL/{}'.format(y))
        cultivated = image.select('cultivated')
        cultivated = cultivated.remap([1, 2], [0, 1])
        if first:
            cultivated = cultivated.rename('clt_{}'.format(y))
            first = False
        else:
            cultivated.addBands(cultivated.rename('clt_{}'.format(y)))

    cultivated = cultivated.reduce(mode_reduce).resample('bilinear').rename('cdl')

    if yr in cdl_years:
        image = ee.Image('USDA/NASS/CDL/{}'.format(yr))
        crop = image.select('cropland')
    else:
        first = True
        for y in cdl_years:
            image = ee.Image('USDA/NASS/CDL/{}'.format(y))
            crop = image.select('cropland')
            if first:
                crop = crop.rename('crop_{}'.format(y))
                first = False
            else:
                crop.addBands(crop.rename('crop_{}'.format(y)))
        crop = crop.reduce(mode_reduce).rename('cropland')

    cdl_keys, our_keys = remap_cdl()
    simple_crop = crop.remap(cdl_keys, our_keys).rename('crop5c').resample('bilinear')
    return cultivated, crop, simple_crop


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
