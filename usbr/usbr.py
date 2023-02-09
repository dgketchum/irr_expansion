import os
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString


def join_districts(_dir, out_shp):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.shp')]
    odf = gpd.GeoDataFrame(columns=['id', 'name', 'geometry'])
    ct = 0
    for s in l:
        df = gpd.read_file(s)

        bname = os.path.basename(s).split('.')[0].replace(' ', '_')
        print(bname, df.crs)
        geo = df.loc[0, 'geometry']

        assert not isinstance(geo, LineString) and not isinstance(geo, MultiLineString)

        df = df.explode(ignore_index=True)
        cols = [x if x == 'geometry' else x.upper() for x in df.columns]
        df.columns = cols

        if 'NAME' in cols:
            name = True
        else:
            name = False

        if df.shape[0] > 1:
            for i, r in df.iterrows():
                geo = r['geometry']

                if not name:
                    _name = '{}_{}'.format(bname, i + 1)
                else:
                    _name = r['NAME']

                try:
                    _name = _name.replace(' ', '_').lower()
                except AttributeError:
                    _name = '{}_{}'.format(bname, i + 1)

                row = {'geometry': geo, 'name': _name, 'id': ct}
                odf.loc[ct] = row
                ct += 1
                print(_name)
        else:
            if not name:
                _name = bname
            else:
                _name = df.loc[0, 'NAME']
                _name = _name.replace(' ', '_').lower()
            row = {'geometry': df.loc[0, 'geometry'], 'name': _name, 'id': ct}
            odf.loc[ct] = row
            ct += 1
            print(_name)

    odf.to_file(out_shp, crs='EPSG:4326')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    br = os.path.join(root, '')
    singles = os.path.join(br, 'to_merge')
    joined = os.path.join(br, 'merged')
    out_file = os.path.join(joined, 'usbr_districts_north_7FEB2023.shp')
    join_districts(singles, out_file)
# ========================= EOF ====================================================================
