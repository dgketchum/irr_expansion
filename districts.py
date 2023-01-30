import os
import json

import geopandas as gpd


def write_metadata(shp, meta_out):
    df = gpd.read_file(shp)
    dct = {}
    for i, r in df.iterrows():
        dct[str(r['FID']).rjust(8, '0')] = {'NAME': str(r['FID']).rjust(8, '0'), 'AREA': r['geometry'].area / 1e6}
    with open(meta_out, 'w') as f:
        json.dump(dct, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/expansion'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/expansion'

    s = '/media/research/IrrigationGIS/expansion/shapefiles/reclamation/usbr_districts_north_5071_gt_1sqkm.shp'
    # s = '/media/research/IrrigationGIS/expansion/shapefiles/reclamation/irr_usbr_clipped_gt40sqkm.shp'
    data_ = os.path.join(root, 'gages', 'usbr_metadata_north.json')
    # data_ = os.path.join(root, 'gages', 'nonreclamation_metadata_north.json')
    write_metadata(s, data_)
# ========================= EOF ====================================================================
