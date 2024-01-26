import numpy as np
from astropy import units as u
from astropy import coordinates as coord


def __local_spherical_2_cartesian(phi: float, theta: float, distance: float) -> (float, float, float):
    # physical convention: inclination=theta (to x axis), azimuth=phi (to z axis)
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    x = distance * np.sin(theta) * np.cos(phi)
    y = distance * np.sin(theta) * np.sin(phi)
    z = distance * np.cos(theta)
    return x, y, z


def __local_cartesian_2_spherical(x: int, y: float, z: float) -> (float, float, float):
    # physical convention: inclination=theta (to x axis), azimuth=phi (to z axis)
    distance = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    theta = np.arccos(z / distance)
    theta = np.rad2deg(theta)
    phi = np.arctan2(y, x)
    phi = np.rad2deg(phi)
    return phi, theta, distance


def __lonLatHeight_2_ITRS(lon: float, lat: float, height: float) -> (float, float, float):
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


def __local_2_intermediate(local_beam_pos: [(float, float, float)], sat_pos_itrs: (float, float, float),
                           sat_vel_itrs: (float, float, float)) -> [(float, float, float)]:
    # local_pos.shape = (n,3)  all points of one beam
    # Local 'satellite' coordinate system:
    #       cartesian: x = direction of movement, z = direction of earth center (down), y = depending
    #       spherical: phi = angle to x-axis, theta = angle to the z-axis, r = -not important-
    # Intermediate satellite coordinate system:
    #       cartesian: z = direction of earth center (down), x = tangential vector towards the z-axis, y = depending
    # with the position vector, a tangential plant in hessian normal-form is given: p dot n_0 - dist = 0
    sat_pos_itrs = np.array(sat_pos_itrs)
    dist = np.linalg.norm(sat_pos_itrs)
    n_0 = sat_pos_itrs / dist
    # find intersection of tangential plane with z-axis (in ITRS)
    z_p = dist / n_0[2]
    # vector_s in tangential plane from position to z-axis
    vec_s = np.array([0, 0, z_p]) - sat_pos_itrs
    vel_tang = __get_tangential_velocity(sat_pos_itrs, sat_vel_itrs)
    # angle alpha_intermediate between vector_s and (tangential) velocity
    dot_prod = np.dot(vec_s, vel_tang)
    mul_norms = np.linalg.norm(vec_s) * np.linalg.norm(vel_tang)
    alpha_i = np.arccos(dot_prod / mul_norms)
    # For Iridium the inclination is 86,4° (the orbits inclination is towards the east).
    if vel_tang[2] > 0:
        # if the z-component of the velocity is positive, the satellite is going upwards
        if vec_s[2] < 0:
            # if the intersection with the z-axis is in negative area, the satellite is south of the equator
            alpha_i = alpha_i - np.pi  # correct 170° to -10°
        else:
            # if the z-intersection is positive, the satellite is on the north half of the earth
            alpha_i = - alpha_i  # correct 10° to -10° (towards east)
    else:
        # if the z-component of the velocity is negative, the satellite is going downwards
        if vec_s[2] < 0:
            # if the z-intersection is negative, the satellite is in the south half
            alpha_i = np.pi + alpha_i  # correct 10° to (180°+10°) = +190°
        else:
            # if the z-intersection is positive, the satellite is still in the north half
            alpha_i = - alpha_i  # correct 170° to -170°
    rot_z = [[np.cos(alpha_i), np.sin(alpha_i), 0],
             [-np.sin(alpha_i), np.cos(alpha_i), 0],
             [0, 0, 1]]
    rot_z = np.array(rot_z)
    beam_points3a = []
    # a,b,c = np.shape(beam_points2)
    local_beam_pos = np.array(local_beam_pos)
    original_shape = np.shape(local_beam_pos)
    local_beam_pos = np.reshape(local_beam_pos, (np.product(original_shape[:-1]), 3))  # reduce to [x,3] dimensions
    for element_nr in range(len(local_beam_pos)):
        temp_point = local_beam_pos[element_nr, :]
        temp_point = np.dot(rot_z, temp_point)
        beam_points3a.append(temp_point)
    return np.array(beam_points3a).reshape(original_shape)


def __intermediate_2_local(int_beam_pos: [(float, float, float)], sat_pos_itrs: (float, float, float),
                           sat_vel_itrs: (float, float, float)) -> [(float, float, float)]:
    # local_pos.shape = (n,3)  all points of one beam
    # Local 'satellite' coordinate system:
    #       cartesian: x = direction of movement, z = direction of earth center (nadir), y = depending
    #       spherical: phi = angle to x-axis, theta = angle to the z-axis, r = -not important-
    # Intermediate satellite coordinate system:
    #       cartesian: z = direction of earth center (down), x = tangential vector towards the z-axis, y = depending
    # with the position vector, a tangential plant in hessian normal-form is given: p dot n_0 - dist = 0
    sat_itrs = np.array(sat_pos_itrs)
    dist = np.linalg.norm(sat_itrs)
    n_0 = sat_itrs / dist
    # find intersection of tangential plane with z-axis (in ITRS)
    z_p = dist / n_0[2]
    # vector_s in tangential plane from position to z-axis
    vec_s = np.array([0, 0, z_p]) - sat_itrs
    vel_tang = __get_tangential_velocity(sat_pos_itrs, sat_vel_itrs)
    # angle alpha_intermediate between vector_s and (tangential) velocity
    dot_prod = np.dot(vec_s, vel_tang)
    mul_norms = np.linalg.norm(vec_s) * np.linalg.norm(vel_tang)
    alpha_i = np.arccos(dot_prod / mul_norms)
    # For Iridium the inclination is 86,4° (the orbits inclination is towards the east).
    if vel_tang[2] > 0:
        # if the z-component of the velocity is positive, the satellite is going upwards
        if vec_s[2] < 0:
            # if the intersection with the z-axis is in negative area, the satellite is south of the equator
            alpha_i = alpha_i - np.pi  # correct 170° to -10°
        else:
            # if the z-intersection is positive, the satellite is on the north half of the earth
            alpha_i = - alpha_i  # correct 10° to -10° (towards east)
    else:
        # if the z-component of the velocity is negative, the satellite is going downwards
        if vec_s[2] < 0:
            # if the z-intersection is negative, the satellite is in the south half
            alpha_i = np.pi + alpha_i  # correct 10° to (180°+10°) = +190°
        else:
            # if the z-intersection is positive, the satellite is still in the north half
            alpha_i = - alpha_i  # correct 170° to -170°
    rot_z = [[np.cos(alpha_i), np.sin(alpha_i), 0],
             [-np.sin(alpha_i), np.cos(alpha_i), 0],
             [0, 0, 1]]
    rot_z = np.array(rot_z)
    rot_z_inv = np.linalg.inv(rot_z)
    beam_points3a = []
    # a,b,c = np.shape(beam_points2)
    int_beam_pos = np.array(int_beam_pos)
    original_shape = np.shape(int_beam_pos)
    int_beam_pos = np.reshape(int_beam_pos, (int(np.product(original_shape[:-1])), 3))  # reduce to [x,3] dimensions
    for element_nr in range(len(int_beam_pos)):
        temp_point = int_beam_pos[element_nr, :]
        temp_point = np.dot(rot_z_inv, temp_point)
        beam_points3a.append(temp_point)
    return np.array(beam_points3a).reshape(original_shape)


def __intermediate_2_ITRS(int_beam_pos: [(float, float, float)], sat_pos_lolah: (float, float, float)) \
        -> [(float, float, float)]:
    # local_pos.shape = (n,3)  all points of one beam
    # Intermediate satellite coordinate system:
    #       cartesian: z = direction of earth center (down), x = tangential vector towards the z-axis, y = depends
    # Global coordinate system (ITRS):
    #       cartesian: x: equator in greenwich-meridian, z = geographic north-pole, y = depending
    int_beam_pos = np.array(int_beam_pos)
    original_shape = np.shape(int_beam_pos)
    elements = np.product(original_shape[:-1])  # reduce the dimensions to [x,3]
    int_beam_pos = np.reshape(int_beam_pos, (elements, 3))
    lon, lat, h = sat_pos_lolah
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    rot_matrix = [[-np.cos(lon) * np.sin(lat), -np.sin(lon), -np.cos(lon) * np.cos(lat)],
                  [-np.sin(lon) * np.sin(lat), np.cos(lon), -np.sin(lon) * np.cos(lat)],
                  [np.cos(lat), 0, -np.sin(lat)]]
    rot_matrix = np.array(rot_matrix)
    beam_points3 = []
    for element_nr in range(len(int_beam_pos)):
        temp_point = int_beam_pos[element_nr, :]
        temp_point = np.dot(rot_matrix, temp_point)
        beam_points3.append(temp_point)
    return np.array(beam_points3).reshape(original_shape)


def __ITRS_2_intermediate(itrs_beam_pos: [(float, float, float)], sat_pos_lolah: (float, float, float)) -> \
        [(float, float, float)]:
    # Intermediate satellite coordinate system:
    #       cartesian: z = direction of earth center (down), x = tangential vector towards the z-axis, y = depending
    # Global coordinate system (ITRS):
    #       cartesian: x: equator in greenwich-meridian, z = geographic north-pole, y = depending
    itrs_beam_pos = np.array(itrs_beam_pos)
    original_shape = np.shape(itrs_beam_pos)
    elements = int(np.product(original_shape[:-1]))  # reduce the dimensions to [x,3]
    itrs_beam_pos = np.reshape(itrs_beam_pos, (elements, 3))
    lon, lat, h = sat_pos_lolah
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    rot_matrix = [[-np.cos(lon) * np.sin(lat), -np.sin(lon), -np.cos(lon) * np.cos(lat)],
                  [-np.sin(lon) * np.sin(lat), np.cos(lon), -np.sin(lon) * np.cos(lat)],
                  [np.cos(lat), 0, -np.sin(lat)]]
    rot_matrix = np.array(rot_matrix)
    rot_matrix_inv = np.linalg.inv(rot_matrix)
    beam_points3 = []
    for element_nr in range(len(itrs_beam_pos)):
        temp_point = itrs_beam_pos[element_nr, :]
        temp_point = np.dot(rot_matrix_inv, temp_point)
        beam_points3.append(temp_point)
    return np.array(beam_points3).reshape(original_shape)


def __get_tangential_velocity(sat_pos_itrs: (float, float, float), sat_vel_itrs: (float, float, float)) -> \
        (float, float, float):
    # return a vector (in ITRS) of the tangential velocity
    sat_pos = np.array(sat_pos_itrs)
    sat_vel = np.array(sat_vel_itrs)
    projection_vel_on_pos = (np.dot(sat_vel, sat_pos) / np.power(np.linalg.norm(sat_pos), 2)) * sat_pos
    tangential_vel = sat_vel - projection_vel_on_pos
    return tangential_vel


# added for Generic_rec_processed_beam_model
def local_spherical_2_ITRS_relative(local_spherical: [(float, float)], sat_pos_itrs: (float, float, float),
                                    sat_vel_itrs: (float, float, float)) -> [(float, float, float)]:
    # transform local beam points to itrs
    local_cat_list = []
    for point in local_spherical:
        local_cat = __local_spherical_2_cartesian(point[0], point[1], 874.5)
        local_cat_list.append(local_cat)
    local_cat_list = np.array(local_cat_list)
    beam_points_int = __local_2_intermediate(local_cat_list, sat_pos_itrs, sat_vel_itrs)
    sat_pos_lolah = __ITRS_2_LonLatHeight(sat_pos_itrs[0], sat_pos_itrs[1], sat_pos_itrs[2])
    beam_points_itrs_relative = __intermediate_2_ITRS(beam_points_int, sat_pos_lolah)
    return beam_points_itrs_relative


# added for Generic_rec_processed_beam_model
def ITRS_relative_2_local_spherical(rec_pos_ITRS_rel: (float, float, float),
                                    sat_pos_itrs: (float, float, float),
                                    sat_vel_itrs: (float, float, float)) -> [(float, float)]:
    sat_pos_lolah = __ITRS_2_LonLatHeight(sat_pos_itrs[0], sat_pos_itrs[1], sat_pos_itrs[2])
    rec_pos_int = __ITRS_2_intermediate(itrs_beam_pos=rec_pos_ITRS_rel, sat_pos_lolah=sat_pos_lolah)
    rec_pos_loc_cat = __intermediate_2_local(rec_pos_int, sat_pos_itrs, sat_vel_itrs)
    rec_pos_loc_sp = __local_cartesian_2_spherical(rec_pos_loc_cat[0], rec_pos_loc_cat[1], rec_pos_loc_cat[2])
    rec_phi, rec_theta, rec_dist = rec_pos_loc_sp
    return rec_phi, rec_theta


# added for Generic_rec_processed_beam_model
def ITRS_2_LonLatHeight(pos_itrs: [(float, float, float)]) -> [(float, float, float)]:
    lolah_list = []
    for temp_pos in pos_itrs:
        temp_lolah = __ITRS_2_LonLatHeight(temp_pos[0], temp_pos[1], temp_pos[2])
        lolah_list.append(temp_lolah)
    # lon in deg, lat in deg, height in km
    return lolah_list


# added for Generic_rec_processed_beam_model
def loLaH_2_ITRS(pos_lolah: [(float, float, float)]):
    itrs_list = []
    for temp_lolah in pos_lolah:
        temp_itrs = __lonLatHeight_2_ITRS(temp_lolah[0], temp_lolah[1], temp_lolah[2])
        itrs_list.append(temp_itrs)
    return itrs_list
