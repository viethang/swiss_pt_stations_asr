import shapely.geometry as sg
import geopandas as gpd
import pandas as pd

def polygon_contains(polygon, stop):
  point = sg.Point(stop["stop_lon"], stop["stop_lat"])
  return polygon.contains(point)

def extract_stops(geojson_file, stops_file, output_file):
  gdf = gpd.read_file(geojson_file)

  polygon = gdf.iloc[0].geometry

  stops_data = pd.read_csv(stops_file, delimiter=",")
  filtered_stops = stops_data.iloc[[index for index,stop in stops_data.iterrows() if (polygon_contains(polygon, stop))]]
  stop_names = filtered_stops["stop_name"].drop_duplicates()
  print("Extracted stop names", stop_names.size)
  stop_names.to_csv(output_file)

if __name__ == "__main__":
  extract_stops("data/lausanne.geojson", "data/stops.txt", "data/lausanne_stops.csv")