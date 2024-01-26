import pyproj
import json
from functools import partial
from shapely.geometry import Point
from shapely.geometry import MultiPolygon as mPoly
from shapely.ops import transform

import polygon_utilities as pu
import plot_satellite_beams


def __load_json_data(filename: str):
    # {'duration': int, 'start_time': float, 'eve_amount': int, 'inter_eve_distances': int,
    #  'victim_type': int, 'victim_movement_radius': float, 'victim_speed': float, 'RoI_area': float,
    #  'distance_outside': float, 'eve_locations': [[float, float, float]], 'victim_locations': [[float, float, float]],
    #  'polygon': [[[float, float]]]}
    # only convert the polygon to a real polygon, keep the rest as it is
    with open(filename, 'r') as file_input:
        all_lines = file_input.readlines()
    # parse the header-line
    main_obj = json.loads(all_lines[0])
    polygons_loaded = main_obj['polygon']
    single_polygons_lst = []
    for polygon_points_lst in polygons_loaded:
        # load each polygon (each polygon is a list of 2D-coordinates)
        temp_poly = pu.create_shPoly(polygon_points_lst)
        single_polygons_lst.append(temp_poly)
    if len(single_polygons_lst) > 1:
        multi_poly = mPoly(single_polygons_lst)
        main_obj['polygon'] = multi_poly
    else:
        main_obj['polygon'] = single_polygons_lst[0]
    return main_obj


def __circle_around_point(lat: float, lon: float, circle_radius: float):
    # circle_radius in meter
    point = Point(lon, lat)
    local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
        pyproj.Proj(local_azimuthal_projection),
    )

    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
    )

    point_transformed = transform(wgs84_to_aeqd, point)

    buffer = point_transformed.buffer(circle_radius)

    buffer_wgs84 = transform(aeqd_to_wgs84, buffer)

    return buffer_wgs84


def generate_RoI_map(file_name: str, uplink_measurements: dict):
    loaded_data = __load_json_data(filename=file_name)
    if loaded_data['distance_outside'] == 0:
        figure_name = f"valid estimation ({loaded_data['RoI_area']}km²)"
    else:
        figure_name = f"invalid estimation ({loaded_data['RoI_area']}km², {loaded_data['distance_outside']}km)"
    plot_satellite_beams.plot_polygon_manyEves_manyVic(figure_name=figure_name,
                                                       eves_lolah=loaded_data['eve_locations'],
                                                       victims_lolah=loaded_data['victim_locations'],
                                                       poly=loaded_data['polygon'],
                                                       ul_measurements=uplink_measurements)


if __name__ == '__main__':
    # json file with the resulting RoI from the real world measurements
    json_file = "realworld_measurements/realworld_measurements_result_polygon.json"

    # locations where uplink measurements were performed
    eulenbis_lola = (7.623257778, 49.516863333)  # nr 1: eulenbis
    morlautern_lola = (7.7557358335, 49.464428333)  # nr 2: morlautern
    morlautern_radius = 4180
    morlautern_circle = __circle_around_point(morlautern_lola[1], morlautern_lola[0], morlautern_radius)
    bann_lola = (7.602617222, 49.383333333)  # nr 3: bann
    pirmasens_lola = (7.527523889, 49.203052778)  # nr 4: primasens
    humbergturm_lola = (7.779892222, 49.415112778)  # nr 5: humbergturm
    humbergturm_radius = 2862
    humbergturm_circle = __circle_around_point(humbergturm_lola[1], humbergturm_lola[0], humbergturm_radius)

    ul_measurements_dict = {'Eulenbis': (eulenbis_lola, None),
                            'Morlautern': (morlautern_lola, morlautern_circle),
                            'Bann': (bann_lola, None),
                            'Pirmasens': (pirmasens_lola, None),
                            'Humbergturm': (humbergturm_lola, humbergturm_circle)}

    generate_RoI_map(json_file, ul_measurements_dict)
