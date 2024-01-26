import numpy as np
import shapely.geometry
from astropy import coordinates as coord
from astropy import units as u
from shapely.geometry import Polygon as shPoly
from shapely.geometry import MultiPolygon as shMPoly
from shapely.validation import make_valid


map_radius = 6371


def lonLatHeight_2_ITRS(lon: float, lat: float, height: float) -> (float, float, float):
    # lon is +east-west, lat is +north-south
    # lon in deg, lat in deg, height in km
    location = coord.EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=height * u.km)
    return location.itrs.x.value, location.itrs.y.value, location.itrs.z.value


def __ITRS_2_LonLatHeight(x: float, y: float, z: float) -> (float, float, float):
    itrs = coord.ITRS(x * u.km, y * u.km, z * u.km, 0 * u.km / u.s, 0 * u.km / u.s, 0 * u.km / u.s)
    location = itrs.earth_location
    lon = location.geodetic.lon
    lat = location.geodetic.lat
    height = location.geodetic.height
    # lon in deg, lat in deg, height in km
    return lon.value, lat.value, height.value


def __itrs_2_stereographic(center: (float, float, float), points: [(float, float, float)]) -> [(float, float)]:
    # https://en.wikipedia.org/wiki/Stereographic_map_projection
    center = np.array(center)
    points = np.array(points)
    distance1 = np.linalg.norm(center)
    normal_vector = center / distance1
    proj_start = center * (-1.0)
    # use the hessian normal form to calculate the intersection point between map plane and vector_(proj_start - point)
    f1 = distance1 - np.dot(proj_start, normal_vector)
    f2 = np.dot((points - proj_start), normal_vector)
    scaling_factors = f1 / f2
    i_points = (points - proj_start) * scaling_factors[:, None] + proj_start
    # now the intersection points are on a 2D plane, but not scaled... but this still should work
    # https://math.stackexchange.com/questions/4339940/rotating-3d-coordinates-to-2d-plane
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    i_points_relative = i_points - center[None, :]
    z_vector = (0, 0, 1)
    rot_angle = np.arccos((np.dot(normal_vector, z_vector)) / np.linalg.norm(normal_vector))
    rot_vector = np.cross(normal_vector, z_vector)
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    rot_matrix = __one_rotation_to_rule_them_all(rot_angle, rot_vector)
    rot_matrix = rot_matrix.T
    result = np.dot(i_points_relative, rot_matrix)
    return result[:, :2]


def __one_rotation_to_rule_them_all(angle: float, axis: (float, float, float)) -> np.ndarray:
    r_cos = np.cos(angle)
    r_sin = np.sin(angle)
    r_x = axis[0]
    r_y = axis[1]
    r_z = axis[2]
    rot_matrix = np.array([
        r_cos + np.power(r_x, 2) * (1 - r_cos),
        r_x * r_y * (1 - r_cos) - r_z * r_sin,
        r_x * r_z * (1 - r_cos) + r_y * r_sin,
        r_y * r_x * (1 - r_cos) + r_z * r_sin,
        r_cos + np.power(r_y, 2) * (1 - r_cos),
        r_y * r_z * (1 - r_cos) - r_x * r_sin,
        r_z * r_x * (1 - r_cos) - r_y * r_sin,
        r_z * r_y * (1 - r_cos) + r_x * r_sin,
        r_cos + np.power(r_z, 2) * (1 - r_cos)
    ]).reshape(3, 3)
    return rot_matrix


def __stereographic_2_itrs(center_itrs: (float, float, float), points: [(float, float)]) -> [(float, float, float)]:
    # center has to be in ITRS!
    ee_a = 6378.137  # semi-major axis of earth-ellipsoid in WSG84
    ee_b = 6356.752  # semi-minor axis of earth-ellipsoid in WSG84
    center_itrs = np.array(center_itrs)
    distance1 = np.linalg.norm(center_itrs)
    normal_vector = center_itrs / distance1
    proj_start = center_itrs * (-1.0)
    points = np.array(points)
    points_length = np.shape(points)[0]
    if points_length == 0:
        return []
    points = np.c_[points, np.zeros(points_length)]

    z_vector = (0, 0, 1)
    rot_angle = -1.0 * np.arccos((np.dot(normal_vector, z_vector)) / np.linalg.norm(normal_vector))
    rot_vector = np.cross(normal_vector, z_vector)
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    rot_matrix = __one_rotation_to_rule_them_all(rot_angle, rot_vector)
    rot_matrix = rot_matrix.T
    i_points_relative = np.dot(points, rot_matrix)
    i_points = i_points_relative + center_itrs[None, :]

    sp_vector = i_points - proj_start
    a = ee_b ** 2 * sp_vector[:, 0] ** 2 + ee_b ** 2 * sp_vector[:, 1] ** 2 + ee_a ** 2 * sp_vector[:, 2] ** 2
    b = 2 * (ee_b ** 2 * proj_start[0, None] * sp_vector[:, 0] + ee_b ** 2 * proj_start[1, None] * sp_vector[:, 1] +
             ee_a ** 2 * proj_start[2] * sp_vector[:, 2])
    c = ee_b ** 2 * proj_start[0, None] ** 2 + ee_b ** 2 * proj_start[1, None] ** 2 + ee_a ** 2 * proj_start[
        2, None] ** 2 - \
        ee_a ** 2 * ee_b ** 2
    D = b ** 2 - 4 * a * c
    if (D < 0).any():
        print(f"WARNING: __stereographic_2_itrs: negative determinant")
        D = np.abs(D)
    x_1 = (-b + np.sqrt(D)) / (2 * a)
    # x_2 = (-b - np.sqrt(D)) / (2 * a)
    scaling_factors = x_1
    original_points = (i_points - proj_start) * scaling_factors[:, None] + proj_start
    return original_points


def map_2_lolah(center_lolah: (float, float, float), points_map: [(float, float)]) -> [(float, float, float)]:
    center_itrs = lonLatHeight_2_ITRS(center_lolah[0], center_lolah[1], center_lolah[2])
    points_itrs = __stereographic_2_itrs(center_itrs, points_map)
    points_lolah = []
    for temp_point in points_itrs:
        temp_lolah = __ITRS_2_LonLatHeight(temp_point[0], temp_point[1], temp_point[2])
        points_lolah.append(temp_lolah)
    return points_lolah


def points_itrs_2_poly(eve_itrs: (float, float, float), lolah_points: [(float, float, float)]) -> shPoly:
    map_points = __itrs_2_stereographic(eve_itrs, lolah_points)
    poly = shPoly(map_points)
    poly = make_valid(poly)
    return poly


def poly_get_points_itrs(eve_itrs: (float, float, float), polygon: shPoly) -> [[(float, float, float)]]:
    if polygon is None:
        return None
    elif type(polygon) is shMPoly:
        mPoly_points_list = []
        geometries = polygon.geoms
        for i in range(len(geometries)):
            temp_geometry_poly = geometries[i]
            temp_geometry_poly_point_lists = poly_get_points_itrs(eve_itrs, temp_geometry_poly)
            for temp_list in temp_geometry_poly_point_lists:
                mPoly_points_list.append(temp_list)
        return mPoly_points_list  # [(lon, lat, h)]
    elif type(polygon) is shPoly:
        polygons_points_list = []
        temp_points = polygon.exterior.coords
        temp_points2 = np.array(temp_points)  # [(map_x, map_y)]
        polygons_points_list.append(__stereographic_2_itrs(eve_itrs, temp_points2))
        # also add inner holes, if they are there
        for inner_hole in polygon.interiors:
            hole_points = inner_hole.coords
            hole_points2 = np.array(hole_points)  # [(map_x, map_y)]
            polygons_points_list.append(__stereographic_2_itrs(eve_itrs, hole_points2))
        return polygons_points_list
    else:
        corrected = solve_polygon_type_error(polygon)
        return poly_get_points_itrs(eve_itrs, corrected)


def get_shPoly_points_2D(polygon: shPoly) -> [[(float, float)]]:
    # return a list of [numpy-arrays with coordinates (one polygon)] many polygons
    if polygon is None:
        return []
    elif type(polygon) is list:
        final_list = []
        for element in polygon:
            temp_list = get_shPoly_points_2D(element)
            final_list.extend(temp_list)
        return final_list
    elif type(polygon) is shMPoly:
        mPoly_points_list = []
        geometries = polygon.geoms
        for i in range(len(geometries)):
            temp_geometry_poly = geometries[i]
            temp_geometry_poly_point_lists = get_shPoly_points_2D(temp_geometry_poly)
            for temp_list in temp_geometry_poly_point_lists:
                mPoly_points_list.append(temp_list)
        return mPoly_points_list
    elif type(polygon) is shPoly:
        polygons_points_list = []
        temp_points = polygon.exterior.coords
        temp_points2 = np.array(temp_points)  # [(map_x, map_y)]
        polygons_points_list.append(temp_points2)
        # also add inner holes, if they are there
        for inner_hole in polygon.interiors:
            hole_points = inner_hole.coords
            hole_points2 = np.array(hole_points)  # [(map_x, map_y)]
            polygons_points_list.append(hole_points2)
        return polygons_points_list
    else:
        return []


def get_shPoly_outerHull_points_2D(polygon: shPoly) -> [(float, float)]:
    # return a list of [numpy-arrays with coordinates (one polygon)] many polygons
    if polygon is None:
        return []
    elif type(polygon) is list:
        print(f"ERROR: get_shPoly_outerHull_points_2D is intended to work on a single polygon, not on a polygon list")
        return []
    elif type(polygon) is shMPoly:
        # select the largest polygon (in terms of covered area)
        selected_poly_points = None
        selected_size = 0
        geometries = polygon.geoms
        for i in range(len(geometries)):
            temp_geometry_poly = geometries[i]
            temp_geometry_area = temp_geometry_poly.area
            if temp_geometry_area > selected_size:
                selected_size = temp_geometry_area
                selected_poly_points = get_shPoly_outerHull_points_2D(temp_geometry_poly)
        return selected_poly_points
    elif type(polygon) is shPoly:
        temp_points = polygon.exterior.coords
        temp_points2 = np.array(temp_points)  # [(map_x, map_y)]
        return temp_points2
    else:
        return []


def recenter_poly(from_itrs: (float, float, float), to_itrs: (float, float, float), polygon: shPoly) -> shPoly:
    if polygon is None:
        return None
    elif type(polygon) is shMPoly:
        poly_list = []
        geometries = polygon.geoms
        for i in range(len(geometries)):
            temp_geometry_poly = geometries[i]
            recentered_poly = recenter_poly(from_itrs, to_itrs, temp_geometry_poly)
            poly_list.append(recentered_poly)
        return shMPoly(poly_list)
    elif type(polygon) is shPoly:
        temp_points = np.array(polygon.exterior.coords)  # [(map_x, map_y)]
        temp_points2 = __stereographic_2_itrs(from_itrs, temp_points)
        temp_points3 = __itrs_2_stereographic(to_itrs, temp_points2)
        holes_list = []
        for inner_hole in polygon.interiors:
            hole_points = np.array(inner_hole.coords)  # [(map_x, map_y)]
            hole_points2 = __stereographic_2_itrs(from_itrs, hole_points)
            hole_points3 = __itrs_2_stereographic(to_itrs, hole_points2)
            holes_list.append(hole_points3)
        result_poly = shPoly(shell=temp_points3, holes=holes_list)
        result_poly = make_valid(result_poly)
        return result_poly
    else:
        corrected = solve_polygon_type_error(polygon)
        return recenter_poly(from_itrs, to_itrs, corrected)


def create_shPoly(data_points: [(float, float)]) -> shPoly:
    my_poly = shPoly(data_points)
    if not my_poly.is_valid:
        my_poly = make_valid(my_poly)
    return my_poly


def solve_polygon_type_error(input) -> shPoly:
    if input is None:
        return None
    elif (type(input) is shPoly) or (type(input) is shMPoly):
        return input
    elif type(input) is shapely.geometry.GeometryCollection:
        # extract the polygons
        geometries = list(input.geoms)
        polys_list = []
        for temp_geo in geometries:
            if type(temp_geo) is shPoly:
                polys_list.append(temp_geo)
            elif type(temp_geo) is shMPoly:
                polys_list.extend(list(temp_geo.geoms))
        if len(polys_list) == 0:
            return None
        elif len(polys_list) == 1:
            return polys_list[0]
        else:
            return shMPoly(polys_list)
        pass
    elif type(input) is shapely.geometry.LinearRing:
        return shPoly(input)
    elif type(input) is shapely.geometry.LineString:
        return None
    elif type(input) is shapely.geometry.Point:
        return None
    else:
        print(f"ERROR: Polygon_Utilities.solve_polygon_type_error(): unknown data-type: {type(input)}! (data:{input})")


def polygons_union(polygon_list: [shPoly]) -> shPoly:
    if len(polygon_list) == 0:
        return None
    result_poly = None
    for i in range(len(polygon_list)):
        temp_shPoly = polygon_list[i]
        if temp_shPoly is not None:
            temp_shPoly = make_valid(temp_shPoly)
            if result_poly is None:
                result_poly = temp_shPoly
            else:
                result_poly = result_poly.union(temp_shPoly)
    if result_poly is not None:
        return make_valid(result_poly)
    else:
        return None


def polygons_intersection(polygon_list: [shPoly]) -> shPoly:
    if len(polygon_list) == 0:
        return None
    result_poly = None
    for i in range(len(polygon_list)):
        temp_shPoly = polygon_list[i]
        if temp_shPoly is not None:
            temp_shPoly = make_valid(temp_shPoly)
            if result_poly is None:
                result_poly = temp_shPoly
            else:
                result_poly = result_poly.intersection(temp_shPoly)
    if result_poly is not None:
        return make_valid(result_poly)
    else:
        return None


def polygons_subtract(startPoly: shPoly, sub_polygons_list: [shPoly]) -> shPoly:
    if startPoly is None:
        return None
    startPoly = make_valid(startPoly)
    result_poly = startPoly
    for i in range(len(sub_polygons_list)):
        temp_shPoly = sub_polygons_list[i]
        if temp_shPoly is not None:
            temp_shPoly = make_valid(temp_shPoly)
            result_poly = result_poly.difference(temp_shPoly)
    if result_poly is not None:
        return make_valid(result_poly)
    else:
        return None
