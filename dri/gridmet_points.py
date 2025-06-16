import os
import geopandas as gpd


def join_gfid_to_polygons(polygon_path, point_path, shp_output_path, joined_points_output_path):
    polygons = gpd.read_file(polygon_path)
    points = gpd.read_file(point_path)
    points = points.to_crs(polygons.crs)
    joined_gdf = gpd.sjoin_nearest(polygons, points[['GFID', 'geometry']], how='left')
    joined_gdf = joined_gdf.drop(columns='index_right')
    joined_gdf.to_file(shp_output_path)

    joined_gfid_list = joined_gdf['GFID'].dropna().unique()
    output_points = points[points['GFID'].isin(joined_gfid_list)]
    output_points.to_file(joined_points_output_path)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    fields = os.path.join(root, 'fields')
    nv_fields = 'Nevada_Agricultural_Field_Boundaries_20250214'

    polygon_shapefile = os.path.join(fields, 'Nevada_Agricultural_Field_Boundaries_20250214_5071.shp')
    point_shapefile = os.path.join(root, 'swim/gridmet/gridmet_centroids.shp')
    output_shapefile_polygons = os.path.join(nv_fields, 'Nevada_Fields_with_Nearest_GFID.shp')
    output_shapefile_points = os.path.join(nv_fields, 'Joined_Points.shp')

    join_gfid_to_polygons(polygon_shapefile, point_shapefile, output_shapefile_polygons, output_shapefile_points)
# ========================= EOF ====================================================================
