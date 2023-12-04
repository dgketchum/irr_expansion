import os
import sys
import json
import time
from calendar import monthrange

import numpy as np
import ee
import geopandas as gpd

from utils.ee_utils import get_cdl

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
KLAMATH_CLIP = 'users/dgketchum/boundaries/klamath_rogue_buff'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'


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


def get_field_cdl(tables, bucket, year, description, join_col='OPENET_ID'):
    initialize()
    cdl_cult, cdl_crop, cdl_simple = get_cdl(year)
    cdl_crop = cdl_crop.toInt()
    cdl_crop = cdl_crop.rename('cdl')
    select_ = [join_col, 'MGRS_TILE', 'mode']

    with open('field_points/tiles.json', 'r') as f_obj:
        tiles = json.load(f_obj)

    for st in BASIN_STATES:

        fc = ee.FeatureCollection(os.path.join(tables, st))

        for tile in tiles[st]:
            tile_fields = fc.filterMetadata('MGRS_TILE', 'equals', tile)
            out_desc = '{}_{}_{}_{}'.format(description, st, tile, year)

            data = cdl_crop.reduceRegions(collection=tile_fields,
                                          reducer=ee.Reducer.mode())

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)

            ee_task_start(task)
            print(out_desc)


def get_field_itype(tables, bucket, year, description, join_col='OPENET_ID'):
    initialize()
    fc = ee.FeatureCollection('users/dgketchum/expansion/itype_int')
    label = ee.Image(0).paint(featureCollection=fc, color='itype').rename('itype')
    select_ = [join_col, 'MGRS_TILE', 'mode']

    with open('field_points/tiles.json', 'r') as f_obj:
        tiles = json.load(f_obj)

    for st in ['CO', 'MT', 'NM', 'UT', 'WA', 'WY']:

        fc = ee.FeatureCollection(os.path.join(tables, st))

        for tile in tiles[st]:
            tile_fields = fc.filterMetadata('MGRS_TILE', 'equals', tile)
            out_desc = '{}_{}_{}_{}'.format(description, st, tile, year)

            data = label.reduceRegions(collection=tile_fields,
                                       reducer=ee.Reducer.mode(),
                                       scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)

            ee_task_start(task)
            print(out_desc)


def extract_area_data(tables, bucket, years, description, features=None, check_exists=None,
                      min_years=5, debug=False, join_col='STAID', geo_type='Polygon',
                      gs_met=False, masks=True, volumes=True):
    initialize()
    missing_files = None
    if check_exists:
        l = open(check_exists, 'r').readlines()
        missing_files = [ln.strip() for ln in l]
        missing_files = [os.path.basename(f_) for f_ in missing_files]

    fc = ee.FeatureCollection(tables)
    fc = fc.randomColumn('rand', seed=1234)

    if features:
        fc = fc.filter(ee.Filter.inList(join_col, features))

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
            eff_ppt = eff_ppt_coll.filterDate(s, e).select('eff_ppt').mosaic()

            ppt, etr = extract_gridmet_monthly(yr, month)
            ietr = extract_corrected_etr(yr, month)

            area = ee.Image.pixelArea()

            if masks:
                irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
                et = et_sum.mask(irr_mask)
                eff_ppt = eff_ppt.mask(irr_mask).rename('eff_ppt')
                ietr = ietr.mask(irr_mask)
                irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
                irr = irr_mask.multiply(area).rename('irr')
            else:
                irr = irr.lt(1).rename('irr')
                et = et_sum

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('et')
            eff_ppt = eff_ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('eff_ppt')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ppt')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('etr')
            ietr = ietr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ietr')

            cc = et.subtract(eff_ppt).rename('cc')

            if volumes:
                et = et.multiply(area)
                eff_ppt = eff_ppt.multiply(area)
                cc = cc.multiply(area)
                ppt = ppt.multiply(area)
                etr = etr.multiply(area)
                ietr = ietr.multiply(area)

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

            if geo_type in ['huc', 'basin']:
                assert masks
                assert volumes

                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)

                # debug = data.filterMetadata(join_col, 'equals', '14359000').first().getInfo()

                out_desc = '{}_{}_{}'.format(description, yr, month)
                task = ee.batch.Export.table.toCloudStorage(
                    data,
                    description=out_desc,
                    bucket=bucket,
                    fileNamePrefix=out_desc,
                    fileFormat='CSV',
                    selectors=select_)
                ee_task_start(task)
                print(out_desc)

            if geo_type == 'fields':

                select_.append('MGRS_TILE')

                with open('field_points/tiles.json', 'r') as f_obj:
                    tiles = json.load(f_obj)

                for st in ['PK']:

                    fc = ee.FeatureCollection(os.path.join(tables, st))

                    for tile in tiles[st]:

                        tile_pts = fc.filterMetadata('MGRS_TILE', 'equals', tile)
                        out_desc = '{}_{}_{}_{}_{}'.format(description, st, tile, yr, month)

                        if check_exists and out_desc not in missing_files:
                            continue

                        data = bands.reduceRegions(collection=tile_pts,
                                                   reducer=ee.Reducer.mean(),
                                                   scale=30)

                        # debug = data.filterMetadata('OPENET_ID', 'equals', 'MT_1').first().getInfo()

                        task = ee.batch.Export.table.toCloudStorage(
                            data,
                            description=out_desc,
                            bucket=bucket,
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
    bucket = 'wudr'

    # shp = gpd.read_file('/media/research/IrrigationGIS/expansion/shapefiles/'
    #                     'study_area/study_area_huc8_ucrb_wKlamath.shp')
    # feats = list(shp['huc8'])
    #
    # table_ = 'users/dgketchum/hydrography/huc8'
    # extract_area_data(table_, bucket, list(range(1987, 2022)), 'huc8',
    #                   join_col='huc8', geo_type='huc', features=feats,
    #                   volumes=True, masks=True)

    table_ = 'users/dgketchum/gages/expansion_gage_basins'
    extract_area_data(table_, bucket, list(range(2006, 2007)), 'basins',
                      join_col='GAGE_ID', geo_type='basin',
                      volumes=True, masks=True)

    # get_field_itype(table_, bucket, 2021, 'openet_itype')

    # extract_area_data(table_, bucket, list(range(1987, 2022)), 'park_fields',
    #                   join_col='OPENET_ID', geo_type='huc',
    #                   volumes=True, masks=True)

# ========================= EOF ================================================================================
