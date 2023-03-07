import json
from copy import copy
from collections import OrderedDict

import ee
import requests

BASIN_STATES = ['CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
KWARGS = dict(source_desc='SURVEY', sector_desc='CROPS', group_desc=None, commodity_desc=None, short_desc=None,
              domain_desc=None, agg_level_desc='STATE', domaincat_desc=None, statisticcat_desc=None,
              state_name=None, asd_desc=None, county_name=None, region_desc=None, zip_5=None,
              watershed_desc=None, year=None, freq_desc=None, reference_period_desc=None)


def ppi_to_cdl_crop():
    d = {'Farm Products': 'WPU01',
         'Alfalfa': 'WPU01810101',
         'Apples': 'WPUSI01102A',
         # 'Asparagus': 'None',
         'Barley': 'WPU01220101',
         'Caneberries': 'WPS011102',
         'Canola': 'WPU01830171',
         'Carrots': 'WPU01130212',
         'Cherries': 'WPS011102',
         'Chick Peas': 'WPU01130109',
         'Corn': 'WPU01220205',
         'Cotton': 'WPU015',
         'Dry Beans': 'WPU02840102',
         'Durum Wheat': 'WPU0121',
         'Grapes': 'WPU01110228',
         'Hops': 'WPU0122',
         'Lentils': 'WPU0122',
         'Lettuce': 'WPU01130215',
         'Millet': 'WPU0122',
         # 'Mint': 'WPS011102',
         'Oats': 'WPU012203',
         'Onions': 'WPU01130216',
         'Other Hay/Non Alfalfa': 'WPU0181',
         'Other Small Grains': 'WPS012',
         # 'Peaches': 'WPS011102',
         'Pears': 'WPU01110221',
         'Peas': 'WPU01130219',
         'Potatoes': 'WPU011306',
         'Rye': 'WPU0122',
         'Safflower': 'WPS012',
         'Sod/Grass Seed': 'WPU0181',
         'Sorghum': 'WPU012205',
         'Soybeans': 'WPU01830131',
         'Spring Wheat': 'WPU01210102',
         'Squash': 'WPU01130231',
         'Sugarbeets': 'WPU02530702',
         'Sunflower': 'WPU01830161',
         'Sweet Corn': 'WPU01130214',
         'Winter Wheat': 'WPU01210101'}
    return d


def ppi_to_cdl_code():
    d = {1: 'WPU01220205',
         2: 'WPU015',
         4: 'WPU012205',
         5: 'WPU01830131',
         6: 'WPU01830161',
         12: 'WPU01130214',
         21: 'WPU01220101',
         22: 'WPU0121',
         23: 'WPU01210102',
         24: 'WPU01210101',
         25: 'WPS012',
         27: 'WPU0122',
         28: 'WPU012203',
         29: 'WPU0122',
         31: 'WPU01830171',
         33: 'WPS012',
         36: 'WPU01810101',
         37: 'WPU0181',
         41: 'WPU02530702',
         42: 'WPU02840102',
         43: 'WPU011306',
         49: 'WPU01130216',
         51: 'WPU01130109',
         52: 'WPU0122',
         53: 'WPU01130219',
         55: 'WPS011102',
         56: 'WPU0122',
         59: 'WPU0181',
         66: 'WPS011102',
         68: 'WPUSI01102A',
         69: 'WPU01110228',
         77: 'WPU01110221',
         206: 'WPU01130212',
         222: 'WPU01130231',
         227: 'WPU01130215'}
    return d


def nass_annual_price_queries():
    kwargs = copy(KWARGS)
    kwargs.update({'freq_desc': 'ANNUAL'})
    queries = OrderedDict([('Rye', dict([('commodity_desc', 'RYE'),
                                         ('short_desc', 'RYE - PRICE RECEIVED, MEASURED IN $ / BU')]))])

    kwargs = {k: v for k, v in kwargs.items() if v}
    keys = list(queries.keys())
    for k in keys:
        queries[k].update(kwargs)

    return queries


def nass_monthly_price_queries():
    kwargs = copy(KWARGS)
    kwargs.update({'freq_desc': 'MONTHLY'})

    queries = OrderedDict([('Alfalfa', dict([('commodity_desc', 'HAY'),
                                             ('short_desc', 'HAY, ALFALFA - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Other Hay/Non Alfalfa', dict([('commodity_desc', 'HAY'),
                                                           ('short_desc',
                                                            'HAY, (EXCL ALFALFA) - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Corn', dict([('commodity_desc', 'CORN'),
                                          ('short_desc', 'CORN, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU')])),

                           ('Barley', dict([('commodity_desc', 'BARLEY'),
                                            ('short_desc', 'BARLEY - PRICE RECEIVED, MEASURED IN $ / BU')])),

                           ('Winter Wheat', dict([('commodity_desc', 'WHEAT'),
                                                  ('short_desc',
                                                   'WHEAT, WINTER - PRICE RECEIVED, MEASURED IN $ / BU')])),

                           ('Spring Wheat', dict([('commodity_desc', 'WHEAT'),
                                                  ('short_desc',
                                                   'WHEAT, SPRING, (EXCL DURUM) - PRICE RECEIVED, MEASURED IN $ / BU')])),

                           ('Apples', dict([('commodity_desc', 'APPLES'),
                                            ('short_desc',
                                             'APPLES, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / LB')])),

                           ('Potatoes', dict([('commodity_desc', 'POTATOES'),
                                              ('short_desc',
                                               'POTATOES, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Lettuce', dict([('commodity_desc', 'LETTUCE'),
                                             ('short_desc',
                                              'LETTUCE, HEAD, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Sugarbeets', dict([('commodity_desc', 'SUGARBEETS'),
                                                ('short_desc',
                                                 'SUGARBEETS - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Dry Beans', dict([('commodity_desc', 'BEANS'),
                                               ('short_desc',
                                                'BEANS, DRY EDIBLE, INCL CHICKPEAS - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Grapes', dict([('commodity_desc', 'GRAPES'),
                                            ('short_desc',
                                             'GRAPES, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Pears', dict([('commodity_desc', 'PEARS'),
                                           ('short_desc',
                                            'PEARS, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Carrots', dict([('commodity_desc', 'CARROTS'),
                                             ('short_desc',
                                              'CARROTS, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Hops', dict([('commodity_desc', 'HOPS'),
                                          ('short_desc', 'HOPS - PRICE RECEIVED, MEASURED IN $ / LB')])),

                           ('Sweet Corn', dict([('commodity_desc', 'SWEET CORN'),
                                                ('short_desc',
                                                 'SWEET CORN, FRESH MARKET - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Sorghum', dict([('commodity_desc', 'SORGHUM'),
                                             ('short_desc', 'SORGHUM, GRAIN - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Squash', dict([('commodity_desc', 'SQUASH'),
                                            ('short_desc', 'SQUASH - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Soybeans', dict([('commodity_desc', 'SOYBEANS'),
                                              ('short_desc', 'SOYBEANS - PRICE RECEIVED, MEASURED IN $ / BU')])),

                           ('Potatoes', dict([('commodity_desc', 'POTATOES'),
                                              ('short_desc', 'POTATOES - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           # ('Canola', dict([('commodity_desc', 'HAY'),
                           #                  ('short_desc', 'HAY, ALFALFA - PRICE RECEIVED, MEASURED IN $ / TON')])),

                           ('Sunflower', dict([('commodity_desc', 'SUNFLOWER'),
                                               ('short_desc', 'SUNFLOWER - PRICE RECEIVED, MEASURED IN $ / CWT')])),

                           ('Onions', dict([('commodity_desc', 'ONIONS'),
                                            ('short_desc',
                                             'ONIONS, DRY, SPRING - PRICE RECEIVED, MEASURED IN $ / CWT')]))])

    kwargs = {k: v for k, v in kwargs.items() if v}
    keys = list(queries.keys())
    for k in keys:
        queries[k].update(kwargs)

    return queries


def study_area_crops():
    return OrderedDict([(36, ('Alfalfa', 2238577)),
                        (176, ('Grassland/Pasture', 1034576)),
                        (37, ('Other Hay/Non Alfalfa', 426737)),
                        (1, ('Corn', 403791)),
                        (21, ('Barley', 381224)),
                        (24, ('Winter Wheat', 304483)),
                        (23, ('Spring Wheat', 276591)),
                        (68, ('Apples', 245186)),
                        (152, ('Shrubland', 227123)),
                        (43, ('Potatoes', 202253)),
                        (121, ('Developed/Open Space', 183478)),
                        (61, ('Fallow/Idle Cropland', 147398)),
                        # (59, ('Sod/Grass Seed', 132879)),
                        (41, ('Sugarbeets', 110129)),
                        (42, ('Dry Beans', 101632)),
                        (69, ('Grapes', 70741)),
                        (66, ('Cherries', 52967)),
                        (190, ('Woody Wetlands', 39425)),
                        (77, ('Pears', 33875)),
                        (71, ('Other Tree Crops', 31946)),
                        (47, ('Misc Vegs & Fruits', 30843)),
                        (49, ('Onions', 26715)),
                        (122, ('Developed/Low Intensity', 26281)),
                        (56, ('Hops', 23803)),
                        (12, ('Sweet Corn', 22323)),
                        (53, ('Peas', 21980)),
                        (28, ('Oats', 18614)),
                        (195, ('Herbaceous Wetlands', 18493)),
                        (57, ('Herbs', 16449)),
                        (58, ('Clover/Wildflowers', 13852)),
                        (205, ('Triticale', 10365)),
                        (44, ('Other Crops', 9553)),
                        (14, ('Mint', 7835)),
                        (35, ('Mustard', 7586)),
                        (31, ('Canola', 7131)),
                        (67, ('Peaches', 6799)),
                        (70, ('Christmas Trees', 6765)),
                        (142, ('Evergreen Forest', 5964)),
                        (242, ('Blueberries', 5585)),
                        (22, ('Durum Wheat', 4379)),
                        (175, ('', 3908)),
                        (123, ('Developed/Med Intensity', 3677)),
                        (206, ('Carrots', 3520)),
                        (141, ('Deciduous Forest', 2536)),
                        (111, ('Open Water', 2244)),
                        (246, ('Radishes', 2118)),
                        (228, ('', 1947)),
                        (5, ('Soybeans', 1730)),
                        (25, ('Other Small Grains', 1611)),
                        (2, ('Cotton', 1440)),
                        (27, ('Rye', 1399)),
                        (4, ('Sorghum', 1332)),
                        (222, ('Squash', 1323)),
                        (6, ('Sunflower', 1273)),
                        (55, ('Caneberries', 1217)),
                        (207, ('Asparagus', 1114)),
                        (52, ('Lentils', 1113)),
                        (151, ('', 1101)),
                        (227, ('Lettuce', 1101)),
                        (33, ('Safflower', 1007)),
                        (29, ('Millet', 872)),
                        (48, ('Watermelons', 830)),
                        (72, ('Citrus', 812)),
                        (247, ('Turnips', 702)),
                        (229, ('Pumpkins', 636)),
                        (13, ('Pop or Orn Corn', 612)),
                        (208, ('Garlic', 551)),
                        (60, ('Switchgrass', 537)),
                        (214, ('Broccoli', 491)),
                        (250, ('Cranberries', 442)),
                        (221, ('Strawberries', 395)),
                        (124, ('Developed/High Intensity', 394)),
                        (220, ('Plums', 392)),
                        (30, ('Speltz', 373)),
                        (39, ('Buckwheat', 354)),
                        (40, ('', 326)),
                        (219, ('Greens', 305)),
                        (32, ('Flaxseed', 304)),
                        (216, ('Peppers', 303)),
                        (143, ('Mixed Forest', 300)),
                        (224, ('Vetch', 283)),
                        (131, ('Barren', 279)),
                        (223, ('Apricots', 273)),
                        (218, ('Nectarines', 221)),
                        (225, ('Dbl Crop WinWht/Corn', 204)),
                        (244, ('Cauliflower', 201)),
                        (51, ('Chick Peas', 186)),
                        (194, ('', 179)),
                        (11, ('Tobacco', 161)),
                        (76, ('Walnuts', 157)),
                        (74, ('Pecans', 126)),
                        (212, ('Oranges', 114)),
                        (20, ('', 106)),
                        (120, ('', 100)),
                        (237, ('Dbl Crop Barley/Corn', 98)),
                        (243, ('Cabbage', 98)),
                        (87, ('Wetlands', 64)),
                        (38, ('Camelina', 63)),
                        (50, ('Cucumbers', 58)),
                        (46, ('Sweet Potatoes', 46)),
                        (209, ('Cantaloupes', 44)),
                        (34, ('Rape Seed', 41)),
                        (65, ('Barren', 40)),
                        (204, ('Pistachios', 39)),
                        (189, ('', 31)),
                        (226, ('Dbl Crop Oats/Corn', 24)),
                        (63, ('Forest', 21)),
                        (210, ('Prunes', 18)),
                        (54, ('Tomatoes', 15)),
                        (140, ('', 13)),
                        (249, ('Gourds', 11)),
                        (236, ('Dbl Crop WinWht/Sorghum', 10)),
                        (241, ('Dbl Crop Corn/Soybeans', 10)),
                        (245, ('Celery', 10)),
                        (213, ('Honeydew Melons', 9)),
                        (211, ('Olives', 8)),
                        (26, ('Dbl Crop WinWht/Soybeans', 7)),
                        (110, ('', 6)),
                        (75, ('Almonds', 3)),
                        (73, ('', 2)),
                        (112, ('Perennial Ice/Snow', 1)),
                        (235, ('Dbl Crop Barley/Sorghum', 1)),
                        (45, ('Sugarcane', 1))])


def cdl_key():
    """Four-class system (1: grain, 2: vegetable, 3: forage, 4: orchard. Plus 5: non-ag/undefined"""
    key = {0: ('None', 5),
           1: ('Corn', 1),
           2: ('Cotton', 1),
           3: ('Rice', 1),
           4: ('Sorghum', 1),
           5: ('Soybeans', 1),
           6: ('Sunflower', 1),
           7: ('', 5),
           8: ('', 5),
           9: ('', 5),
           10: ('Peanuts', 1),
           11: ('Tobacco', 2),
           12: ('Sweet Corn', 1),
           13: ('Pop or Orn Corn', 1),
           14: ('Mint', 2),
           15: ('', 5),
           16: ('', 5),
           17: ('', 5),
           18: ('', 5),
           19: ('', 5),
           20: ('', 5),
           21: ('Barley', 1),
           22: ('Durum Wheat', 1),
           23: ('Spring Wheat', 1),
           24: ('Winter Wheat', 1),
           25: ('Other Small Grains', 1),
           26: ('Dbl Crop WinWht/Soybeans', 1),
           27: ('Rye', 1),
           28: ('Oats', 1),
           29: ('Millet', 1),
           30: ('Speltz', 1),
           31: ('Canola', 1),
           32: ('Flaxseed', 1),
           33: ('Safflower', 1),
           34: ('Rape Seed', 1),
           35: ('Mustard', 1),
           36: ('Alfalfa', 3),
           37: ('Other Hay/Non Alfalfa', 3),
           38: ('Camelina', 1),
           39: ('Buckwheat', 1),
           40: ('', 5),
           41: ('Sugarbeets', 2),
           42: ('Dry Beans', 2),
           43: ('Potatoes', 2),
           44: ('Other Crops', 2),
           45: ('Sugarcane', 2),
           46: ('Sweet Potatoes', 2),
           47: ('Misc Vegs & Fruits', 2),
           48: ('Watermelons', 2),
           49: ('Onions', 2),
           50: ('Cucumbers', 2),
           51: ('Chick Peas', 2),
           52: ('Lentils', 2),
           53: ('Peas', 2),
           54: ('Tomatoes', 2),
           55: ('Caneberries', 2),
           56: ('Hops', 2),
           57: ('Herbs', 2),
           58: ('Clover/Wildflowers', 3),
           59: ('Sod/Grass Seed', 3),
           60: ('Switchgrass', 3),
           61: ('Fallow/Idle Cropland', 6),
           62: ('Pasture/Grass', 3),
           63: ('Forest', 5),
           64: ('Shrubland', 5),
           65: ('Barren', 5),
           66: ('Cherries', 4),
           67: ('Peaches', 4),
           68: ('Apples', 4),
           69: ('Grapes', 4),
           70: ('Christmas Trees', 4),
           71: ('Other Tree Crops', 4),
           72: ('Citrus', 4),
           73: ('', 5),
           74: ('Pecans', 4),
           75: ('Almonds', 4),
           76: ('Walnuts', 4),
           77: ('Pears', 4),
           78: ('', 5),
           79: ('', 5),
           80: ('', 5),
           81: ('Clouds/No Data', 5),
           82: ('Developed', 5),
           83: ('Water', 5),
           84: ('', 5),
           85: ('', 5),
           86: ('', 5),
           87: ('Wetlands', 5),
           88: ('Nonag/Undefined', 5),
           89: ('', 5),
           90: ('', 5),
           91: ('', 5),
           92: ('Aquaculture', 5),
           93: ('', 5),
           94: ('', 5),
           95: ('', 5),
           96: ('', 5),
           97: ('', 5),
           98: ('', 5),
           99: ('', 5),
           100: ('', 5),
           101: ('', 5),
           102: ('', 5),
           103: ('', 5),
           104: ('', 5),
           105: ('', 5),
           106: ('', 5),
           107: ('', 5),
           108: ('', 5),
           109: ('', 5),
           110: ('', 5),
           111: ('Open Water', 5),
           112: ('Perennial Ice/Snow', 5),
           113: ('', 5),
           114: ('', 5),
           115: ('', 5),
           116: ('', 5),
           117: ('', 5),
           118: ('', 5),
           119: ('', 5),
           120: ('', 5),
           121: ('Developed/Open Space', 5),
           122: ('Developed/Low Intensity', 5),
           123: ('Developed/Med Intensity', 5),
           124: ('Developed/High Intensity', 5),
           125: ('', 5),
           126: ('', 5),
           127: ('', 5),
           128: ('', 5),
           129: ('', 5),
           130: ('', 5),
           131: ('Barren', 5),
           132: ('', 5),
           133: ('', 5),
           134: ('', 5),
           135: ('', 5),
           136: ('', 5),
           137: ('', 5),
           138: ('', 5),
           139: ('', 5),
           140: ('', 5),
           141: ('Deciduous Forest', 5),
           142: ('Evergreen Forest', 5),
           143: ('Mixed Forest', 5),
           144: ('', 5),
           145: ('', 5),
           146: ('', 5),
           147: ('', 5),
           148: ('', 5),
           149: ('', 5),
           150: ('', 5),
           151: ('', 5),
           152: ('Shrubland', 5),
           153: ('', 5),
           154: ('', 5),
           155: ('', 5),
           156: ('', 5),
           157: ('', 5),
           158: ('', 5),
           159: ('', 5),
           160: ('', 5),
           161: ('', 5),
           162: ('', 5),
           163: ('', 5),
           164: ('', 5),
           165: ('', 5),
           166: ('', 5),
           167: ('', 5),
           168: ('', 5),
           169: ('', 5),
           170: ('', 5),
           171: ('', 5),
           172: ('', 5),
           173: ('', 5),
           174: ('', 5),
           175: ('', 5),
           176: ('Grassland/Pasture', 5),
           177: ('', 5),
           178: ('', 5),
           179: ('', 5),
           180: ('', 5),
           181: ('', 5),
           182: ('', 5),
           183: ('', 5),
           184: ('', 5),
           185: ('', 5),
           186: ('', 5),
           187: ('', 5),
           188: ('', 5),
           189: ('', 5),
           190: ('Woody Wetlands', 5),
           191: ('', 5),
           192: ('', 5),
           193: ('', 5),
           194: ('', 5),
           195: ('Herbaceous Wetlands', 5),
           196: ('', 5),
           197: ('', 5),
           198: ('', 5),
           199: ('', 5),
           200: ('', 5),
           201: ('', 5),
           202: ('', 5),
           203: ('', 5),
           204: ('Pistachios', 4),
           205: ('Triticale', 1),
           206: ('Carrots', 2),
           207: ('Asparagus', 2),
           208: ('Garlic', 2),
           209: ('Cantaloupes', 2),
           210: ('Prunes', 2),
           211: ('Olives', 2),
           212: ('Oranges', 3),
           213: ('Honeydew Melons', 2),
           214: ('Broccoli', 2),
           215: ('Avocados', 2),
           216: ('Peppers', 2),
           217: ('Pomegranates', 4),
           218: ('Nectarines', 4),
           219: ('Greens', 2),
           220: ('Plums', 4),
           221: ('Strawberries', 2),
           222: ('Squash', 2),
           223: ('Apricots', 4),
           224: ('Vetch', 3),
           225: ('Dbl Crop WinWht/Corn', 1),
           226: ('Dbl Crop Oats/Corn', 1),
           227: ('Lettuce', 2),
           228: ('', 1),
           229: ('Pumpkins', 2),
           230: ('Dbl Crop Lettuce/Durum Wht', 2),
           231: ('Dbl Crop Lettuce/Cantaloupe', 2),
           232: ('Dbl Crop Lettuce/Cotton', 2),
           233: ('Dbl Crop Lettuce/Barley', 2),
           234: ('Dbl Crop Durum Wht/Sorghum', 1),
           235: ('Dbl Crop Barley/Sorghum', 1),
           236: ('Dbl Crop WinWht/Sorghum', 1),
           237: ('Dbl Crop Barley/Corn', 1),
           238: ('Dbl Crop WinWht/Cotton', 1),
           239: ('Dbl Crop Soybeans/Cotton', 1),
           240: ('Dbl Crop Soybeans/Oats', 1),
           241: ('Dbl Crop Corn/Soybeans', 1),
           242: ('Blueberries', 2),
           243: ('Cabbage', 2),
           244: ('Cauliflower', 2),
           245: ('Celery', 2),
           246: ('Radishes', 2),
           247: ('Turnips', 2),
           248: ('Eggplants', 2),
           249: ('Gourds', 2),
           250: ('Cranberries', 2),
           251: ('', 5),
           252: ('', 5),
           253: ('', 5),
           254: ('Dbl Crop Barley/Soybeans', 1),
           255: ('', 5)}
    return key


def remap_cdl():
    """remap cdl to alternative class system, return tuple (original, remapped)"""
    key = cdl_key()
    map_ = list(key.keys())
    remap = [v[1] for k, v in key.items()]
    return map_, remap


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


def cdl_accuracy(out_js):
    dct = {s: [] for s in BASIN_STATES}
    for y in range(2008, 2022):
        for s in BASIN_STATES:
            url = 'https://www.nass.usda.gov/Research_and_Science/' \
                  'Cropland/metadata/metadata_{}{}.htm'.format(s.lower(), str(y)[-2:])
            resp = requests.get(url).content.decode('utf-8')
            for i in resp.split('\n'):
                txt = i.strip('\r')
                if txt.startswith('OVERALL'):
                    l = txt.split(' ')
                    k = float(l[-1])
            dct[s].append(k)
            print(s, y, '{:.3f}'.format(k))

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
