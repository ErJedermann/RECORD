import shapely
from shapely.geometry import Polygon as shPoly
from shapely.geometry import MultiPolygon as shMPoly
from shapely.validation import make_valid

import numpy as np
import matplotlib.path as mpltPath

import plot_satellite_beams
import beam_model.beamModel_coordinate_transformations as bct


# goal is to reduce number of points to be more efficient
# Internally the beam model works with two dimensions, not a 3-dimensional. In previous processing the collected
# data points are treated as points on a surface of an imaginary sphere around the satellite, based on the angles phi
# and theta. The internal coordinate system converts phi and theta to a new 2D cartesian space:
#       before: phi = angle to the velocity-axis, theta = angle to nadir (towards-earth-axis)
#       new 2d space: center (0,0) is the nadir,
#                     treat it as complex numbers: phi (number) is phi (coord) and r (number) is theta (coord)
#                     x_new = Real(z) = r * cos(phi) = theta * cos(phi)
#                     y_new = Imaginary(z) = r * sin(phi) = theta * sin(phi)


# small helper class 'polygon' (during model creation it was encapsulating several functions)
class Polygon:
    def __init__(self, edge_point_list: [([float, float], [float, float])]):
        self.ordered_edge_points = edge_point_list

    def get_points_line(self):
        return self.ordered_edge_points  # list of 2d-points, shape=(n,2)


class GenericRecordedProcessedModel:
    def __init__(self, sat_name: str = None):
        if sat_name is None:
            self.sat_name = "generic_proc_model"
        else:
            self.sat_name = sat_name
        self.distance_sat = 700  # km
        self.earth_radius = 6371  # km
        self.cluster_list = []  # (beam-number, edge-points-list[(x, y)])
        self.beam_polygon_dict = {}  # {beam-nr: [polygons]} (the outer hull of each beam)
        self.beam_center_dict = {}  # {beam-nr: (x, y)} (beam-centers in internal 2d space)

    def spherical_2_flat(self, phi: float, theta: float):
        phi = np.deg2rad(phi)
        x_new = theta * np.sin(phi)
        y_new = theta * np.cos(phi)
        return x_new, y_new

    def flat_2_spherical(self, x, y):
        theta = np.sqrt(np.power(x, 2) + np.power(y, 2))
        phi = np.arctan2(x, y)
        phi = np.rad2deg(phi)
        return phi, theta

    def __solve_polygon_type_error(self, input, depth_counter: int = 0, depth_max: int = 5) -> [shPoly]:
        if depth_counter >= depth_max:
            return []
        if input is None:
            return []
        elif type(input) is shPoly:
            return [input]
        elif type(input) is shMPoly:
            return list(input.geoms)
        elif type(input) is shapely.geometry.GeometryCollection:
            # extract the polygons
            geometries = list(input.geoms)
            polys_list = []
            for temp_geo in geometries:
                temp_geo = self.__solve_polygon_type_error(temp_geo, depth_counter + 1)
                for t2_geo in temp_geo:
                    if t2_geo != None:
                        if type(t2_geo) is shPoly:
                            polys_list.append(t2_geo)
                        elif type(t2_geo) is shMPoly:
                            polys_list.extend(list(t2_geo.geoms))
            return polys_list
        elif type(input) is shapely.geometry.LinearRing:
            return [shPoly(input)]
        elif type(input) is shapely.geometry.LineString:
            return []
        elif type(input) is shapely.geometry.Point:
            return []
        elif type(input) is shapely.geometry.MultiLineString:
            resultstring = []
            for linestring in input.geoms:
                resultstring.extend(linestring.coords)
            resultPoly = shPoly(resultstring)
            resultPoly = make_valid(resultPoly)
            return self.__solve_polygon_type_error(resultPoly, depth_counter + 1)
        else:
            print(f"ERROR: solve_polygon_type_error(): unknown data-type: {type(input)}! (data:{input})")

    def __check_inside_polygon(self, point: [float, float], polygon_points: [[float, float]]) -> bool:
        polygon_points = np.array(polygon_points)
        path = mpltPath.Path(polygon_points)
        inside1 = path.contains_point(point)
        return inside1


    def load_processed_model(self, filepath: str):
        temp_data_arr = np.load(filepath, allow_pickle=True)
        temp_sat_name, temp_data_dict, temp_center_dict = temp_data_arr
        self.sat_name = temp_sat_name
        self.beam_polygon_dict = {}
        self.beam_polygon_dict.update(temp_data_dict)
        self.beam_center_dict = {}
        self.beam_center_dict.update(temp_center_dict)

    # returns all beams, where the given receiver-position is in
    def get_beams(self, rec_pos_lolah: (float, float, float), sat_pos_ITRS: (float, float, float),
                  sat_vel_ITRS: (float, float, float), threshold: int = 1, sat_name: str = "") -> [str]:
        # get the relative rec-position in ITRS
        rec_pos_ITRS = bct.loLaH_2_ITRS([rec_pos_lolah])[0]
        rec_pos_ITRS = np.array(rec_pos_ITRS)
        sat_pos_ITRS = np.array(sat_pos_ITRS)
        rec_pos_ITRS_rel = rec_pos_ITRS - sat_pos_ITRS  # relative vector sat->rec
        # transform the relative rec-pos to local (spherical) system
        rec_phi, rec_theta = bct.ITRS_relative_2_local_spherical(rec_pos_ITRS_rel, sat_pos_ITRS, sat_vel_ITRS)
        # convert local (spherical) to flat-2d
        rec_flat = self.spherical_2_flat(rec_phi, rec_theta)
        # go through all polygons and check if the point is inside one of them, then select the whole beam
        beam_list = []
        for beam_name in list(self.beam_polygon_dict.keys()):
            polygon_list = self.beam_polygon_dict[beam_name]
            for polygon in polygon_list:
                poly_points = polygon.get_points_line()
                if self.__check_inside_polygon(rec_flat, poly_points):
                    beam_list.append(sat_name + beam_name)
                    break
        return beam_list

    # returns the beam, where the given receiver-position is closest to the center and ensures that it is visible
    def get_beam(self, rec_pos_lolah: (float, float, float), sat_pos_ITRS: (float, float, float),
                 sat_vel_ITRS: (float, float, float), sat_name: str = "") -> str:
        # get the relative rec-position in ITRS
        rec_pos_ITRS = bct.loLaH_2_ITRS([rec_pos_lolah])[0]
        rec_pos_ITRS = np.array(rec_pos_ITRS)
        sat_pos_ITRS = np.array(sat_pos_ITRS)
        rec_pos_ITRS_rel = rec_pos_ITRS - sat_pos_ITRS  # relative vector sat->rec
        # transform the relative rec-pos to local (spherical) system
        rec_phi, rec_theta = bct.ITRS_relative_2_local_spherical(rec_pos_ITRS_rel, sat_pos_ITRS, sat_vel_ITRS)
        # convert local (spherical) to flat-2d
        rec_flat = np.array(self.spherical_2_flat(rec_phi, rec_theta))
        # check which beam-center is closest
        beam_centers = np.array(list(self.beam_center_dict.values()))
        beam_distances = np.sqrt(np.sum(np.power(beam_centers - rec_flat[None, :], 2), axis=1))
        beam_names = np.array(list(self.beam_center_dict.keys()))
        sorting_order = np.argsort(beam_distances)
        beam_sorted_names = beam_names[sorting_order]
        beam_sorted_names = list(beam_sorted_names)

        # ensure that the beams is really available by returning the closest available beam
        available_beams = self.get_beams(rec_pos_lolah, sat_pos_ITRS, sat_vel_ITRS)
        available_beams = list(available_beams)
        # sorted_av_beams = sorted(available_beams, key=lambda i: beam_sorted_names.index(i[0]))  # somehow doesn't work
        sorted_av_beams = [b_name for x in beam_sorted_names for b_name in available_beams if b_name == x]

        if len(sorted_av_beams) > 0:
            return sat_name + sorted_av_beams[0]
        else:
            return "-no-beam-"

    def print_flat_beam_polygons(self):
        plot_satellite_beams.plot_beam_polygons(self.beam_polygon_dict)

    # Improved method that does two additional rounds, to archive height-accuracy in +3 m to -8 mm (earth ellipsoid)
    # https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html#astropy.coordinates.EarthLocation
    # Had to be adapted to return itrs coordinates instead of lolah.
    def calculate_footprint(self,
                            sat_pos_itrs: (float, float, float) = (4669.37662199, 226.40705157, 5410.80234396),
                            sat_vel_itrs: (float, float, float) = (-5.64170821, 0.10273897, 4.85158888),
                            beams_list: [str] = None, plot_footprint: bool = False) -> [[(float, float, float)]]:
        ee_a = 6378.137  # semi-major axis of earth-ellipsoid in WSG84
        ee_b = 6356.752  # semi-minor axis of earth-ellipsoid in WSG84
        sat_pos_itrs = np.array(sat_pos_itrs)
        sat_vel_itrs = np.array(sat_vel_itrs)
        if beams_list is None:
            beams_list = list(self.beam_polygon_dict.keys())
        beams_print_dict = {}  # key=beam_name, value=[[lo,la,h]] (list of polygon-footprints [list of lolah-points])

        for beam_index in beams_list:
            beam_polygons_list = self.beam_polygon_dict[beam_index]
            polygons_print_list = []
            for polygon in beam_polygons_list:
                polygon_points_flat = polygon.get_points_line()
                polygon_points_spherical = []
                for point in polygon_points_flat:
                    point_s = self.flat_2_spherical(point[0], point[1])
                    polygon_points_spherical.append(point_s)
                polygon_points_itrs_relative = bct.local_spherical_2_ITRS_relative(polygon_points_spherical,
                                                                                   sat_pos_itrs, sat_vel_itrs)
                # calculate the points, where the beam-points hit the surface of the elliptical earth therefore find
                # an r where 'sat_pos + r * beam_point = surface' (for each beam-point).
                # The surface of an ellipse is x²/a² + y²/a² + z²/b² = 1; a and b are the semi-major and semi-minor axis
                # for i in range(len(polygon_points_itrs_relative)):
                #     # (use abc-formula to solve it analytic)
                #     beam_pos = polygon_points_itrs_relative[i, :]
                #     a = ee_b**2 * beam_pos[0]**2 + ee_b**2 * beam_pos[1]**2 + ee_a**2 * beam_pos[2]**2
                #     b = 2 * (ee_b**2 * sat_pos_itrs[0] * beam_pos[0] + ee_b**2 * sat_pos_itrs[1] * beam_pos[1] +
                #              ee_a**2 * sat_pos_itrs[2] * beam_pos[2])
                #     c = ee_b**2 * sat_pos_itrs[0]**2 + ee_b**2 * sat_pos_itrs[1]**2 + ee_a**2 * sat_pos_itrs[2]**2 - \
                #         ee_a**2 * ee_b**2
                #     D = b**2 - 4 * a * c
                #     if D < 0:
                #         print(f"WARNING: GRPBM.calc_footprint: negative determinant: {D}")
                #         D = abs(D)
                #     x_1 = (-b + np.sqrt(D)) / (2 * a)
                #     x_2 = (-b - np.sqrt(D)) / (2 * a)
                #     if abs(x_1) < abs(x_2):
                #         scale_factors.append(x_1)
                #     else:
                #         scale_factors.append(x_2)

                # use abc-formula to solve it analytic + vectorize the stuff
                beam_pos = polygon_points_itrs_relative
                a = ee_b ** 2 * beam_pos[:, 0] ** 2 + ee_b ** 2 * beam_pos[:, 1] ** 2 + ee_a ** 2 * beam_pos[:, 2] ** 2
                b = 2 * (ee_b ** 2 * sat_pos_itrs[0, None] * beam_pos[:, 0] +
                         ee_b ** 2 * sat_pos_itrs[1, None] * beam_pos[:, 1] +
                         ee_a ** 2 * sat_pos_itrs[2, None] * beam_pos[:, 2])
                c = ee_b ** 2 * sat_pos_itrs[0, None] ** 2 + ee_b ** 2 * sat_pos_itrs[1, None] ** 2 + \
                    ee_a ** 2 * sat_pos_itrs[2, None] ** 2 - ee_a ** 2 * ee_b ** 2
                D = b ** 2 - 4 * a * c
                D = np.where(D >= 0, D, np.NaN)

                x_1 = (-b + np.sqrt(D)) / (2 * a)
                x_2 = (-b - np.sqrt(D)) / (2 * a)
                x_1 = np.abs(x_1)
                x_2 = np.abs(x_2)
                pre_factors = np.array([x_1, x_2]).T
                scale_factors = pre_factors.min(axis=1)

                # calculate the footprint-points
                beam_points_itrs_surface_rel = np.array(polygon_points_itrs_relative) * np.array(scale_factors)[:, None]
                beam_points_itrs_surface1 = beam_points_itrs_surface_rel + np.array(sat_pos_itrs)

                # calculate the horizon point for all beams (closest point to surface when it does not hit the surface)
                # https://math.stackexchange.com/questions/13176/how-to-find-a-point-on-a-line-closest-to-another-given-point
                scale_fac_lop = -(sat_pos_itrs[0, None] * beam_pos[:, 0] + sat_pos_itrs[1, None] * beam_pos[:, 1] +
                                  sat_pos_itrs[2, None] * beam_pos[:, 2]) / (
                                        np.power(beam_pos[:, 0], 2) +
                                        np.power(beam_pos[:, 1], 2) +
                                        np.power(beam_pos[:, 2], 2))
                point_on_line = sat_pos_itrs + np.array(scale_fac_lop)[:, None] * beam_pos
                scale_fac_horizon = np.sqrt(np.power(ee_a * ee_b, 2) / (
                        np.power(ee_b * point_on_line[:, 0], 2) +
                        np.power(ee_b * point_on_line[:, 1], 2) +
                        np.power(ee_a * point_on_line[:, 2], 2)))
                horizon_point = np.array(scale_fac_horizon)[:, None] * point_on_line

                # when surface-points == NaN, use the horizon points
                beam_points_itrs_surface1[:, 0] = np.where(np.isnan(beam_points_itrs_surface1[:, 0]),
                                                           horizon_point[:, 0], beam_points_itrs_surface1[:, 0])
                beam_points_itrs_surface1[:, 1] = np.where(np.isnan(beam_points_itrs_surface1[:, 1]),
                                                           horizon_point[:, 1], beam_points_itrs_surface1[:, 1])
                beam_points_itrs_surface1[:, 2] = np.where(np.isnan(beam_points_itrs_surface1[:, 2]),
                                                           horizon_point[:, 2], beam_points_itrs_surface1[:, 2])
                polygons_print_list.append(beam_points_itrs_surface1)
            beams_print_dict[beam_index] = polygons_print_list
        # print it
        if plot_footprint:
            sat_lolah = bct.ITRS_2_LonLatHeight([sat_pos_itrs])[0]
            plot_satellite_beams.plot_polygon_footprints(sat_lolah, beams_print_dict)
        return beams_print_dict


if __name__ == '__main__':
    bm = GenericRecordedProcessedModel()
    bm.load_processed_model("beamModel_iridium.npy")
    bm.print_flat_beam_polygons()
    bm.calculate_footprint(plot_footprint=True)