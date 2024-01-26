# The goal is to provide a method to evaluate a given solution-polygon, based on the area of the polygon.
# To use the methods provided by the shapely-library, the points have to be converted from a 3D geographic coordinate
# system (lon, lat, height) to a 2D area-preserving projection.
# https://www.researchgate.net/publication/324252646_A_bevy_of_area-preserving_transforms_for_map_projection_designers
# We use Lamberts Azimuthal Equal-Area Projection:
# https://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html


import numpy as np
import polygon_utilities as pu
from shapely.geometry import Polygon as shPoly
from shapely.geometry import MultiPolygon as shMPoly
from shapely.geometry import Point as shPoint
import shapely.validation as shVal


def __lola_2_lambert(center_lolah: (float, float, float), points_lola: [(float, float)]) -> [(float, float)]:
    c_lon = np.deg2rad(center_lolah[0])
    c_lat = np.deg2rad(center_lolah[1])
    points_lola = np.array(points_lola)
    p_lon = np.deg2rad(points_lola[:, 0])
    p_lat = np.deg2rad(points_lola[:, 1])
    k = np.sqrt(2 / (1 + np.sin(c_lat) * np.sin(p_lat) + np.cos(c_lat) * np.cos(p_lat) * np.cos(p_lon - c_lon)))
    x_lambert = k * np.cos(p_lat) * np.sin(p_lon - c_lon)
    y_lambert = k * (np.cos(c_lat) * np.sin(p_lat) - np.sin(c_lat) * np.cos(p_lat) * np.cos(p_lon - c_lon))
    lambert = np.array([x_lambert, y_lambert]).T
    return lambert


def __estimationPoly_2_lambert(polygon: shMPoly, center_lolah: (float, float, float)) -> shMPoly:
    if polygon is None:
        return None
    elif type(polygon) is shMPoly:
        lambert_polygons = []
        for geometry in polygon.geoms:
            temp_lambert_poly = __estimationPoly_2_lambert(geometry, center_lolah)
            lambert_polygons.append(temp_lambert_poly)
        lambert_mPoly = shMPoly(lambert_polygons)
        return lambert_mPoly
    elif type(polygon) is shPoly:
        ext_points_est = polygon.exterior.coords
        if ext_points_est == None or len(ext_points_est) == 0:
            return None
        ext_points_lolah = pu.map_2_lolah(center_lolah, ext_points_est)
        ext_points_lambert = __lola_2_lambert(center_lolah, ext_points_lolah)
        int_points_lambert_list = []
        for inner_hole in polygon.interiors:
            hole_points_est = inner_hole.coords
            hole_points_lolah = pu.map_2_lolah(center_lolah, hole_points_est)
            hole_points_lambert = __lola_2_lambert(center_lolah, hole_points_lolah)
            int_points_lambert_list.append(hole_points_lambert)
        lambert_poly = shPoly(shell=ext_points_lambert, holes=int_points_lambert_list)
        return lambert_poly
    else:
        corrected = pu.solve_polygon_type_error(polygon)
        return __estimationPoly_2_lambert(corrected, center_lolah)


def evaluate_polygon_area(polygon: shMPoly, eve_lolah: (float, float, float), earth_mean_radius: float = 6371) -> float:
    # transform from estimation-map to Lambert
    area_polygon = __estimationPoly_2_lambert(polygon, eve_lolah)
    # get the area of the hammer_polygon and roughly scale it back to km²
    if area_polygon is not None:
        estimated_area = area_polygon.area
        scaling_factor = np.power(earth_mean_radius, 2)
        scaled_area = estimated_area * scaling_factor
        return scaled_area
    else:
        return 0


def evaluate_polygon(polygon: shMPoly, eve_lolah: (float, float, float), victim_lolah: (float, float, float),
                     earth_mean_radius: float = 6371) -> (bool, float):
    # transform from estimation-map to Lambert
    lambert_polygon = __estimationPoly_2_lambert(polygon, eve_lolah)
    # get the area of the hammer_polygon and roughly scale it back to km²
    victim_lambert = __lola_2_lambert(eve_lolah, [victim_lolah])
    victim_point = shPoint(victim_lambert[0, :])
    if lambert_polygon is not None:
        if not lambert_polygon.is_valid:
            lambert_polygon = shVal.make_valid(lambert_polygon)
        if lambert_polygon.is_valid:
            estimated_area = lambert_polygon.area
            scaling_factor = np.power(earth_mean_radius, 2)
            scaled_area = estimated_area * scaling_factor
            is_inside = lambert_polygon.contains(victim_point)
            if not is_inside:
                victim_itrs = pu.lonLatHeight_2_ITRS(victim_lolah[0], victim_lolah[1], victim_lolah[2])
                eve_itrs = pu.lonLatHeight_2_ITRS(eve_lolah[0], eve_lolah[1], eve_lolah[2])
                victim_stereo_coord = pu.__itrs_2_stereographic(eve_itrs, [victim_itrs])
                victim_stereo_point = shPoint(victim_stereo_coord[0])
                outside_distance = victim_stereo_point.distance(polygon)
                return is_inside, scaled_area, outside_distance
            return is_inside, scaled_area, 0
        else:
            print(f"ERROR: User_Position_Evaluator.evaluate_polygon(): still invalid lambert_polygon!")
            return False, 0, -2
    else:
        print(f"ERROR: User_Position_Evaluator.evaluate_polygon(): lambert_polygon is None!")
        return False, 0, -1


# from: https://stackoverflow.com/questions/11710972/great-circle-distance-plus-altitude-change
# even better: https://www.ngs.noaa.gov/TOOLS/Inv_Fwd/Inv_Fwd.html
def great_circle_distance(point1_lolah: (float, float, float), point2_lolah: (float, float, float),
                          earth_mean_radius: float = 6371) -> float:
    lon1 = np.deg2rad(point1_lolah[0])
    lat1 = np.deg2rad(point1_lolah[1])
    lon2 = np.deg2rad(point2_lolah[0])
    lat2 = np.deg2rad(point2_lolah[1])
    delta_lon = np.abs(lon1 - lon2)
    central_angle = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))
    distance = earth_mean_radius * central_angle
    return distance


# not as precise as NASA, but much simpler (no altitude included): https://de.wikipedia.org/wiki/Orthodrome
def orthodrome_distance(point1_lolah: (float, float, float), point2_lolah: (float, float, float)) -> float:
    # factors for the WGS84-ellipsoid
    f = 1.0 / 298.257223563  # flattening of the earth
    a = 6378.137  # equator radius [km]
    # rough distance
    lon1 = np.deg2rad(point1_lolah[0])  # lambda
    lat1 = np.deg2rad(point1_lolah[1])  # phi
    lon2 = np.deg2rad(point2_lolah[0])
    lat2 = np.deg2rad(point2_lolah[1])
    F = (lat1 + lat2) / 2
    G = (lat1 - lat2) / 2
    l = (lon1 - lon2) / 2
    S = np.power(np.sin(G), 2) * np.power(np.cos(l), 2) + np.power(np.cos(F), 2) * np.power(np.sin(l), 2)
    C = np.power(np.cos(G), 2) * np.power(np.cos(l), 2) + np.power(np.sin(F), 2) * np.power(np.sin(l), 2)
    omega = np.arctan(np.sqrt(S / C))
    D = 2 * omega * a
    # improve distance
    T = np.sqrt(S * C) / omega
    H1 = (3 * T - 1) / (2 * C)
    H2 = (3 * T + 1) / (2 * S)
    s = D * (1 + f * H1 * np.power(np.sin(F), 2) * np.power(np.cos(G), 2)
             - f * H2 * np.power(np.cos(F), 2) * np.power(np.sin(G), 2))
    return s


def evaluate_point_estimator(polygon: shMPoly, eve_lolah: (float, float, float), victim_lolah: (float, float, float),
                             earth_mean_radius: float = 6371) -> float:
    # get the centroid as point-estimator
    if polygon is None:
        return -1
    centroid_estimate = polygon.centroid  # this centroid may be not in the middle of the area, since it depends on the edge-points
    centroid_estimate = list(centroid_estimate.coords)
    centroid_lolah = pu.map_2_lolah(center_lolah=eve_lolah, points_map=centroid_estimate)
    if len(centroid_lolah) == 0:
        return -1
    centroid_lolah = centroid_lolah[0]
    distance = great_circle_distance(victim_lolah, centroid_lolah, earth_mean_radius)
    return distance
