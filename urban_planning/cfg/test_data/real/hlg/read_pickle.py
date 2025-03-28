import pickle
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

# pickle_file = '/Users/manasvivarma/Downloads/working code /data1/hlg/111/models/best.p'
pickle_file = '/Users/manasvivarma/Downloads/working code 2 (1)/working code /urban_planning/cfg/test_data/real/hlg/init_plan_hlg.pickle'

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)


print(data)

# gdf = data['gdf']
# polygons_gdf = gdf[gdf.geometry.apply(lambda geom: geom.geom_type) == 'Polygon']
# polygons_gdf['area'] = polygons_gdf.geometry.area
# polygons_gdf['edge_length'] = polygons_gdf.geometry.length
# average_stats = polygons_gdf.groupby('type').agg(
#     average_area=('area', 'mean'),
#     average_edge_length=('edge_length', 'mean')
# )
# print("Average area and edge length for each polygon type:")
# print(average_stats)
# print(gdf.crs)

# # print(gdf)
# office_gdf = gdf[(gdf['type'] == 7) & (gdf['geometry'].geom_type == 'Polygon')]
# def get_edge_lengths(polygon):
#     coords = list(polygon.exterior.coords)
#     edges = [LineString([coords[i], coords[i+1]]).length for i in range(len(coords) - 1)]
#     return edges
# office_gdf['min_edge_length'] = office_gdf['geometry'].apply(lambda poly: min(get_edge_lengths(poly)) if poly.is_valid else None)
# office_gdf['max_edge_length'] = office_gdf['geometry'].apply(lambda poly: max(get_edge_lengths(poly)) if poly.is_valid else None)
# print(office_gdf[['geometry', 'min_edge_length', 'max_edge_length']])


# gdf = data["best_plans"][0]["gdf"]




# # gdf['traffic'] = np.nan
# # gdf.loc[gdf['geometry'].geom_type == 'Point', 'traffic'] = np.random.randint(0, 31, size=gdf[gdf['geometry'].geom_type == 'Point'].shape[0])
# # print(gdf[gdf['geometry'].geom_type == 'LineString'])

# # Filter for LineString geometries with type == 2 and population == 5
# # new_roads = gdf[(gdf['geometry'].geom_type == 'LineString') & 
# #                 (gdf['type'] == 2) & 
# #                 (gdf['population'] == 5)]

# # new_roads['length'] = new_roads['geometry'].length
# # mean_length = new_roads['length'].mean()
# # print(f"The mean length of the new roads is: {mean_length}")


# print(gdf)
# # gdf.loc[183, 'type'] = 15
# # print(gdf[(gdf['geometry'].geom_type == 'Polygon') & (gdf['population'].isna())])

# # print(gdf['population'].isna().sum())

# # ********************************************
# print("Before modification:", gdf['type'].unique())
# # print(gdf)

# # unique_geometries = gdf['geometry'].apply(lambda geom: geom.geom_type).unique()
# # print(unique_geometries)


# # gdf['type'] = gdf['type'].replace(15, 16)
# # gdf['type'] = gdf['type'].replace(14, 15)
# # gdf['type'] = gdf['type'].replace(16, 14)
# # # print("After modification:", gdf['type'].unique())
# # # print(gdf[gdf['geometry'].geom_type == 'Point'])
# # data['gdf'] = gdf
# # with open(pickle_file, 'wb') as file:
# #     pickle.dump(data, file)

# # print("Data saved back to pickle file.")

# # *********************************************

# # type_population_mapping = {
# #     1: (500,1000) ,     # FEASIBILE
# #     4: (1000, 5000),   # RESIDENTIAL
# #     5: (500, 1000),    # BUSINESS
# #     6: (10, 50),       # WASTEMGMT
# #     7: (50, 200),      # GREEN_L
# #     8: (10, 50),       # GREEN_S
# #     9: (300, 1000),    # SCHOOL
# #     10: (500, 2000),   # HOSPITAL_L
# #     11: (100, 500),    # HOSPITAL_S
# #     12: (50, 300),     # RECREATION
# #     13: (200, 1000)    # OFFICE
# # }


# # def generate_population(row):
# #     # Only process if geometry is a polygon and 'type' is in the mapping
# #     if row.geometry.geom_type == 'Polygon' and row['type'] in type_population_mapping:
# #         population_range = type_population_mapping[row['type']]
# #         return np.random.randint(*population_range)
# #     return np.nan


# # gdf['population'] = gdf.apply(generate_population, axis=1)

# # # Step 4: Display the GeoDataFrame with the new population column for polygons
# # #print(gdf[['geometry', 'type', 'population']].head())

# # filtered_gdf = gdf[gdf['type'].isin(type_population_mapping.keys())]
# # print(filtered_gdf[['geometry', 'type', 'population']].head())

# # data['gdf'] = gdf
# # with open(pickle_file, 'wb') as file:
# #     pickle.dump(data, file)

# # print("Data saved back to pickle file.")



# def save_gdf_to_geojson(gdf, file_path):
#     """
#     Save a GeoDataFrame to a GeoJSON file.
    
#     Parameters:
#         gdf (GeoDataFrame): The GeoDataFrame to save.
#         file_path (str): The path where the GeoJSON file will be saved.
#     """
#     # Ensure the GeoDataFrame is valid
#     if not isinstance(gdf, gpd.GeoDataFrame):
#         raise ValueError("The input must be a GeoDataFrame.")
    
#     # Save the GeoDataFrame to GeoJSON format
#     gdf.to_file(file_path, driver='GPKG')
#     print(f"GeoDataFrame saved to {file_path} as GeoJSON.")

# # Example usage:
# save_gdf_to_geojson(gdf, 'map1.gpkg')
