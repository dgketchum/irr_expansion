import os
import sys
from calendar import monthrange

import ee

from training_data import ee_task_start, is_authorized

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


def irr_et_data(yr):
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)
    klamath_clip = ee.FeatureCollection(KLAMATH_CLIP)
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

        annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/klamath').merge(
            ee.ImageCollection('users/dpendergraph/ssebop/klamath'))
        et_coll = annual_coll.filter(ee.Filter.date(s, e))
        et_klam = et_coll.sum().clip(klamath_clip.geometry())

        annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
        et_coll = annual_coll_.filter(ee.Filter.date(s, e))
        et_umrb = et_coll.sum().clip(umrb_clip.geometry())

        et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb, et_klam]).mosaic()
        et = et_sum.multiply(0.00001).rename('et_{}'.format(month))

        ept = ee.ImageCollection('users/dgketchum/expansion/ept').filterDate(s, e).sum()
        ept = ept.rename('eff_ppt_{}'.format(month))

        if month == 4:
            bands = et.addBands(ept)
        else:
            bands = bands.addBands([et, ept])

    return bands


def extract_validation_data(file_prefix, points_layer, years, scale):
    ee.Initialize()
    points = ee.FeatureCollection(points_layer)

    for st in BASIN_STATES:
        for yr in years:

            stack = irr_et_data(yr)

            state_bound = ee.FeatureCollection(os.path.join(BOUNDARIES, st))
            stack = stack.clip(state_bound)
            st_points = points.filterMetadata('STUSPS', 'equals', st)

            plot_sample_regions = stack.sampleRegions(
                collection=st_points,
                scale=scale,
                tileScale=16,
                projection=ee.Projection('EPSG:4326'))

            desc = '{}_{}_{}'.format(file_prefix, st, yr)
            task = ee.batch.Export.table.toCloudStorage(
                plot_sample_regions,
                description=desc,
                bucket='wudr',
                fileNamePrefix=desc,
                fileFormat='CSV')

            ee_task_start(task)
            print(desc)


if __name__ == '__main__':
    is_authorized()

    points_ = 'users/dgketchum/expansion/points/validation_pts_28FEB2023'
    bucket = 'wudr'
    years_ = [x for x in range(1987, 2022)]
    clip = 'users/dgketchum/expansion/study_area_klamath'
    desc_ = 'validation_28FEB2023'
    extract_validation_data(desc_, points_, years_, 30)

# ========================= EOF ====================================================================
