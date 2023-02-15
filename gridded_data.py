import os
import sys
from calendar import monthrange

import numpy as np
import ee

from call_ee import ee_task_start

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
KLAMATH_CLIP = 'users/dgketchum/boundaries/klamath_rogue_buff'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'
BASIN_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def get_geomteries():
    bozeman = ee.Geometry.Polygon([[-111.19206055457778, 45.587493372544984],
                                   [-110.91946228797622, 45.587493372544984],
                                   [-110.91946228797622, 45.754947053477565],
                                   [-111.19206055457778, 45.754947053477565],
                                   [-111.19206055457778, 45.587493372544984]])

    navajo = ee.Geometry.Polygon([[-108.50192867920967, 36.38701227276218],
                                  [-107.92995297120186, 36.38701227276218],
                                  [-107.92995297120186, 36.78068624960868],
                                  [-108.50192867920967, 36.78068624960868],
                                  [-108.50192867920967, 36.38701227276218]])

    test_point = ee.Geometry.Point(-111.96061568829488, 47.652738197659694)

    western_us = ee.Geometry.Polygon([[-127.00073221292574, 30.011505140554807],
                                      [-100.63354471292574, 30.011505140554807],
                                      [-100.63354471292574, 49.908396143431744],
                                      [-127.00073221292574, 49.908396143431744],
                                      [-127.00073221292574, 30.011505140554807]])

    return bozeman, navajo, test_point, western_us


def export_gridded_data(tables, bucket, years, description, features=None, check_exists=None,
                        min_years=5, debug=False, join_col='STAID', geo_type='Polygon',
                        gs_met=False):
    initialize()
    missing_files = None
    if check_exists:
        l = open(f, 'r').readlines()
        missing_files = [ln.strip() for ln in l]
        missing_files = [f.split('.')[0] for f in missing_files]
        missing_files = [os.path.basename(f) for f in missing_files]

    fc = ee.FeatureCollection(tables)
    fc = fc.randomColumn('rand', seed=1234)

    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)
    klamath_clip = ee.FeatureCollection(KLAMATH_CLIP)

    eff_ppt_coll = ee.ImageCollection('users/dgketchum/expansion/ept')
    eff_ppt_coll = eff_ppt_coll.map(lambda x: x.rename('eff_ppt'))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_years)

    for yr in years:
        for month in range(1, 13):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/klamath').merge(
                ee.ImageCollection('users/dpendergraph/ssebop/klamath'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_klam = et_coll.sum().multiply(0.00001).clip(klamath_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb, et_klam]).mosaic()
            et = et_sum.mask(irr_mask)

            eff_ppt = eff_ppt_coll.filterDate(s, e).select('eff_ppt').mosaic()
            eff_ppt = eff_ppt.mask(irr_mask).rename('eff_ppt')

            ppt, etr = extract_gridmet_monthly(yr, month)
            ietr = extract_corrected_etr(yr, month)
            ietr = ietr.mask(irr_mask)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            eff_ppt = eff_ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ietr = ietr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(eff_ppt)

            area = ee.Image.pixelArea()
            irr = irr_mask.multiply(area).rename('irr')
            et = et.multiply(area).rename('et')
            eff_ppt = eff_ppt.multiply(area).rename('eff_ppt')
            cc = cc.multiply(area).rename('cc')
            ppt = ppt.multiply(area).rename('ppt')
            etr = etr.multiply(area).rename('etr')
            ietr = ietr.multiply(area).rename('ietr')

            if yr > 1986 and month in range(4, 11):
                bands = irr.addBands([et, cc, ppt, etr, eff_ppt, ietr])
                select_ = [join_col, 'irr', 'et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
                if gs_met:
                    select_ = [join_col, 'ppt', 'etr']
                    bands = ppt.addBands(etr)

            else:
                bands = ppt.addBands([etr])
                select_ = [join_col, 'ppt', 'etr']

            if debug:
                pt = bands.sample(region=get_geomteries()[2],
                                  numPixels=1,
                                  scale=30)
                p = pt.first().getInfo()['properties']
                print('propeteries {}'.format(p))

            out_desc = '{}_{}_{}'.format(description, yr, month)

            if geo_type == 'Polygon':

                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)

                task = ee.batch.Export.table.toCloudStorage(
                    data,
                    description=out_desc,
                    bucket=bucket,
                    fileNamePrefix=out_desc,
                    fileFormat='CSV',
                    selectors=select_)

                task.start()
                print(out_desc)

            elif geo_type == 'Point':
                for st in BASIN_STATES:
                    st_points = fc.filterMetadata('STUSPS', 'equals', st)
                    out_desc = '{}_{}_{}_{}'.format(description, st, yr, month)

                    if check_exists and out_desc not in missing_files:
                        continue

                    plot_sample_regions = bands.sampleRegions(
                        collection=st_points,
                        scale=30,
                        tileScale=16)

                    task = ee.batch.Export.table.toCloudStorage(
                        plot_sample_regions,
                        description=out_desc,
                        bucket='wudr',
                        fileNamePrefix=out_desc,
                        fileFormat='CSV',
                        selectors=select_)

                    ee_task_start(task)
                    print(out_desc)


def extract_corrected_etr(year, month):
    m_str = str(month).rjust(2, '0')
    end_day = monthrange(year, month)[1]
    ic = ee.ImageCollection('projects/openet/reference_et/gridmet/monthly')
    band = ic.filterDate('{}-{}-01'.format(year, m_str), '{}-{}-{}'.format(year, m_str, end_day)).select('etr').first()
    return band.multiply(0.001)


def extract_gridmet_monthly(year, month):
    m_str, m_str_next = str(month).rjust(2, '0'), str(month + 1).rjust(2, '0')
    if month == 12:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-31'.format(year, m_str))
    else:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-01'.format(year, m_str_next))
    pet = dataset.select('etr').sum().multiply(0.001).rename('gm_etr')
    ppt = dataset.select('pr').sum().multiply(0.001).rename('gm_ppt')
    return ppt, pet


def initialize():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))


if __name__ == '__main__':
    bucket = 'wudr'
    f = os.path.join(os.getcwd(), 'field_points', 'missing.txt')
    table_ = 'users/dgketchum/expansion/points/field_pts_attr_13FEB2023'
    years_ = list(range(1984, 1987))
    # years_.reverse()
    export_gridded_data(table_, bucket, years_, 'ietr_fields_13FEB2023', min_years=5, gs_met=True,
                        debug=False, join_col='OPENET_ID', geo_type='Point', check_exists=False)
# ========================= EOF ================================================================================
