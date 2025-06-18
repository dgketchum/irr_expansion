import time

import ee

from openet.refetgee import Daily

# import geopandas as gpd

ee.Initialize()

# test range
# start_date = '2020-01-01'
# end_date = '2020-01-31'
# Full rcp45 and rcp85 date range (2006,2099)
start_date = '2006-01-01'
end_date = '2099-12-31'

# scenario_list = ['rcp45', 'rcp85', 'historical']
future_scenario_list = ['rcp45', 'rcp85']

model_list = [
    'bcc-csm1-1',
    'bcc-csm1-1-m',
    'BNU-ESM',
    'CanESM2',
    'CCSM4',
    'CNRM-CM5',
    'CSIRO-Mk3-6-0',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'HadGEM2-CC365',
    'HadGEM2-ES365',
    'inmcm4',
    'IPSL-CM5A-MR',
    'IPSL-CM5A-LR',
    'IPSL-CM5B-LR',
    'MIROC5',
    'MIROC-ESM',
    'MIROC-ESM-CHEM',
    'MRI-CGCM3',
    'NorESM1-M',
]

# testing combos
# model_list = ['bcc-csm1-1', 'bcc-csm1-1-m']

# ftr from ee asset upload (add .shp to ftr option)
ftr = ee.FeatureCollection('projects/ee-dgketchum/assets/swim/gridmet_selected_points_nevada')
# import itertools
# for i, j in itertools.product(range(x), range(y)):
for scenario in future_scenario_list:
    for model in model_list:
        for yr in range(2006, 2100):

            maca_coll = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA').filterDate(f'{yr}-01-01', f'{yr + 1}-01-01')
            maca_coll = maca_coll.filterMetadata('model', 'equals', model).filterMetadata('scenario', 'equals',
                                                                                          scenario)


            def maca_reduceregions(i):
                dateStr = i.date()
                datenum = ee.Image.constant(ee.Number.parse(dateStr.format("YYYYMMdd")))
                # Add dateNum Band to Image
                dateYear = ee.Number.parse(dateStr.format("YYYY"))
                dateMonth = ee.Number.parse(dateStr.format("MM"))
                dateDay = ee.Number.parse(dateStr.format("dd"))
                i = i.addBands(ee.Image(datenum).rename('datenum'))
                i = i.addBands(ee.Image(dateYear).rename('year'))
                i = i.addBands(ee.Image(dateMonth).rename('month'))
                i = i.addBands(ee.Image(dateDay).rename('day'))
                # geerefet eto calc
                eto = Daily.maca(i).eto
                i = i.addBands(ee.Image(eto).rename('eto_mm'))
                # geerefet etr calc
                etr = Daily.maca(i).etr
                i = i.addBands(ee.Image(etr).rename('etr_mm'))
                # resultant wind speed from vectors
                ws = ee.Image(i.select(['uas'])).pow(2) \
                    .add(ee.Image(i.select(['vas'])).pow(2)) \
                    .sqrt().rename(['ws'])
                i = i.addBands(ee.Image(ws).rename('uz'))
                zw = 10  # maca wind vector measurement height (meters)
                # Wind speed(Eqn 67) adjust to 2m
                u2 = ws.expression(
                    'uz * 4.87 / log(67.8 * zw - 5.42)', {'uz': ws, 'zw': zw})
                i = i.addBands(ee.Image(u2).rename('ws_2m'))
                fc = i.reduceRegions(ftr, ee.Reducer.mean())
                return ee.FeatureCollection(fc).set('date', ee.String(dateStr.format('YYYY-MM-dd')))


            output = maca_coll.map(maca_reduceregions)
            # print(output.getInfo())

            selectors = ['GFID', 'lat', 'lon'] + ['datenum', 'year', 'month', 'day', 'UACE10',
                                                  'tasmax', 'tasmin', 'rsds', 'uz', 'huss', 'pr', 'ws_2m', 'eto_mm',
                                                  'etr_mm']

            desc = f'{scenario}_{model}_{yr}'
            export_name = f'maca/{desc}'

            task = ee.batch.Export.table.toCloudStorage(
                collection=ee.FeatureCollection(output.flatten()),
                selectors=selectors,
                description=desc,
                fileNamePrefix=export_name,
                fileFormat='CSV',
                bucket='wudr')

            try:
                task.start()
                print(desc, flush=True)

            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), desc, '......')
                time.sleep(600)
                task.start()
                print(desc)
