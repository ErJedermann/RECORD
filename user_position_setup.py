import numpy as np

import user_position_evaluator as eval
import user_position_setup_sunflower as sunflower


def __estimate_pos(start: (float, float, float), direction: float, distance: float,
                   earth_mean_radius: float = 6371) -> (float, float, float):
    # calculate the shift in north and east (in km)
    shift_north = np.cos(direction) * distance
    shift_east = np.sin(direction) * distance
    # calculate the shift angle
    angle_north = np.arccos((2 * earth_mean_radius ** 2 - shift_north ** 2) / (2 * earth_mean_radius ** 2))
    angle_east = np.arccos((2 * earth_mean_radius ** 2 - shift_east ** 2) / (2 * earth_mean_radius ** 2))
    angle_north = np.rad2deg(angle_north)
    angle_east = np.rad2deg(angle_east)
    # add the shift angle to the start point
    lon2 = start[0] + angle_east
    lat2 = start[1] + angle_north
    v_pos = (lon2, lat2, 0)
    return v_pos


def __generate_new_position(start: (float, float, float), dist: float, angle: float = None) -> (float, float, float):
    if angle is None:  # select random the direction of the victim
        angle = np.random.random() * 2 * np.pi
    v_pos = __estimate_pos(start, angle, dist)
    exact_dist = eval.orthodrome_distance(start, v_pos)
    # print(f"DEBUG: distance guess: {exact_dist} km")
    scaling_factor = dist / exact_dist
    v_pos2 = __estimate_pos(start, angle, dist * scaling_factor)
    # exact_dist2 = eval.orthodrome_distance(start, v_pos2)
    # print(f"DEBUG: better distance: {exact_dist2} km")
    return v_pos2


def __generate_fully_random_position(latitude_max_value_deg: float = 65, latitude_min_value_deg: float = -65) -> (
        float, float, float):
    # return (lon, lat, height) fully random generated
    latitude_range = latitude_max_value_deg - latitude_min_value_deg
    lat_value = np.random.random() * latitude_range + latitude_min_value_deg
    lon_value = np.random.random() * 360
    height_value = np.random.random() * 0.01  # max 10 meter above sea-level
    return (lon_value, lat_value, height_value)


def get_large_scale_setup(eve_amounts: int, inter_eve_dist: float, victim_loc: int = 1) -> \
        ((float, float, float), [(float, float, float)]):
    # Creates a large-scale setup with many equally distributed (with inter_eve_dist km apart) receivers.
    # Victim_loc: 1: Victim is inside the covered area, up to inter_eve_dist km away from the center (used in the paper)
    #             2: Victim is at the border of the covered area (up to inter_eve_dist km inside the covered area).
    #             3: Victim is outside the covered area (up to inter_eve_dist km outside).
    # get the setup
    loLaH_start = __generate_fully_random_position()
    phiTheta, covered_radius = sunflower.sunflower_pattern(eve_amounts, inter_eve_dist)
    # add a randomized offset to the theta-angel (the direction) to avoid a setup-bias
    phiTheta = phiTheta % 360
    theta_offset = np.random.random() * 360
    phiTheta[:, 1] = (phiTheta[:, 1] + theta_offset) % 360
    lat_lon_start = (loLaH_start[1], loLaH_start[0])
    latLon = sunflower.shift_points(lat_lon_start, phiTheta)
    lolah_eves = np.array([latLon[:, 1], latLon[:, 0], np.zeros(len(latLon))]).T  # convert LatLon to LonLatH
    lolah_eves = list(lolah_eves)
    # do the victim-location
    if victim_loc == 1:  # close to the center
        vic_dist = np.random.random() * inter_eve_dist
    elif victim_loc == 2:  # close to the border of the covered area, but inside
        vic_dist = covered_radius - (np.random.random() * inter_eve_dist)
    elif victim_loc == 3:  # close to the border of the covered area, outside
        vic_dist = covered_radius + (np.random.random() * inter_eve_dist)
    else:
        raise Exception(f"User_Position_Setup.get_large_scale_setup: invalid victim_location: {victim_loc}")
    v_loc = __generate_new_position(loLaH_start, vic_dist)
    return v_loc, lolah_eves


def get_many_victims_setup(amounts: int, center_lolah: (float, float, float), dist: float = None,
                           circle_radius: float = None) -> [(float, float, float)]:
    # Creates a large-scale setup with many equally distributed (with dist km apart) locations (victims).
    phiTheta, covered_radius = sunflower.sunflower_pattern(amounts, point_distances=dist, circle_radius=circle_radius)
    phiTheta = phiTheta % 360
    lat_lon_start = (center_lolah[1], center_lolah[0])
    latLon = sunflower.shift_points(lat_lon_start, phiTheta)
    lolah_locations = np.array([latLon[:, 1], latLon[:, 0], np.zeros(len(latLon))]).T  # convert LatLon to LonLatH
    lolah_locations = list(lolah_locations)
    return lolah_locations
