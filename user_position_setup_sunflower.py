import numpy as np
import plotly.graph_objects as go

# Idea for the placement strategy: https://rmets.onlinelibrary.wiley.com/doi/abs/10.1256/qj.05.227
# code-src: https://stackoverflow.com/a/44164075

earth_radius = 6371  # km

def __number_and_distance_to_circle(points: int, point_distance: float) -> float:
    # Converts the number of points and distance between points to the radius of the covered circular area.
    magic_factor = 2.75  # ratio between circular area and cumulative area per point
    foo = 1 - (np.power(point_distance, 2) * np.pi * points) / (magic_factor * 8 * np.pi * np.power(earth_radius, 2))
    return np.rad2deg(np.arccos(foo)) * 2 * np.pi * earth_radius / 360 * 2


def __radius_2_opening_angle(radius: float) -> float:
    # Input: radius [km] (great-circle distance) of a circle on the earth surface.
    # Output: opening angle [°] (between center and circle line) at the earth center of the given circle radius.
    circ_earth = 2 * np.pi * earth_radius
    if radius > circ_earth / 2:
        radius = circ_earth / 2
    return radius / (2 * np.pi * earth_radius) * 360


def __opening_angle_2_area_percent(angle: float) -> float:
    # Input: opening angle [°] of the used area.
    # Output: percentage of how much percent of the earth surface is used.
    # The area change is sinus shaped (small diff at 0°, max diff at 90°, small diff at 180°).
    # Integral from 0 to angle over sin(angle) is the area.
    area_circle = 1 - np.cos(np.deg2rad(angle))
    area_max = 2.0
    return area_circle / area_max


def __area_percent_2_skipped_points(area_percent: float, used_points: int) -> int:
    # Points are equally distributed on the surface -> used area [%] = used points [%].
    all_points = used_points / area_percent
    skipped_points = all_points - used_points
    return int(skipped_points)


def __sunflower_sphere_segment(points: int = 100, points_skipped: int = 0) -> [(float, float)]:
    # places the points in sunflower-seed arrangement (variation of fibonacci) on the unit sphere
    indices = np.arange(0, points, dtype=float) + 0.5
    points_all = points + points_skipped

    phi = np.arccos(1 - 2 * indices / points_all)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    phi = np.rad2deg(phi)
    theta = np.rad2deg(theta)
    return np.array([phi, theta]).T


def spherical_2_cartesian_coordinates(phiTheta: [(float, float)]):
    # good for debugging purposes
    phi = np.deg2rad(phiTheta[:, 0])
    theta = np.deg2rad(phiTheta[:, 1])
    x, y, z = np.cos(theta) * np.sin(phi) * earth_radius, np.sin(theta) * np.sin(phi) * earth_radius, np.cos(
        phi) * earth_radius
    coords = np.array([x, y, z])
    return coords.T


def latLon_2_cartesian_coordinates(latLon: [(float, float)]):
    # good for debugging purposes
    latLon = np.array(latLon)
    lat = np.deg2rad(latLon[:, 0])
    lon = np.deg2rad(latLon[:, 1])
    x = np.cos(lon) * np.cos(lat) * earth_radius
    y = np.sin(lon) * np.cos(lat) * earth_radius
    z = np.sin(lat) * earth_radius
    coords = np.array([x, y, z])
    return coords.T


def sunflower_pattern(points: int, point_distances: float = None, circle_radius: float = None, verbose: bool = False):
    if point_distances is None and circle_radius is None:
        print("ERROR: sunflower_pattern: both 'distance between points' and 'radius of the coverage circle' are None!")
    if point_distances:
        circle_radius = __number_and_distance_to_circle(points, point_distances)
    opening_angle = __radius_2_opening_angle(circle_radius)
    area_percent = __opening_angle_2_area_percent(opening_angle)
    if verbose:
        earth_surface = 4 * np.pi * np.power(earth_radius, 2)
        print(f"circle_radius [km]: {circle_radius}; opening angle [°]: {opening_angle}; "
              f"surface [km²]: {area_percent * earth_surface}")
    skipped_points = __area_percent_2_skipped_points(area_percent, points)
    phiTheta = __sunflower_sphere_segment(points, skipped_points)
    return phiTheta, circle_radius


def shift_points(latLon_start: (float, float), phiTheta: [(float, float)]) -> (float, float):
    # source: https://www.movable-type.co.uk/scripts/latlong.html
    lat1 = np.deg2rad(latLon_start[0])
    lon1 = np.deg2rad(latLon_start[1])
    if type(phiTheta) == list:
        phiTheta = np.array(phiTheta)
    if type(phiTheta) == tuple:
        phiTheta = np.array([phiTheta])
    theta = np.deg2rad(phiTheta[:, 1])
    phi = np.deg2rad(phiTheta[:, 0])
    # φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
    lat2 = np.arcsin(np.sin(lat1) * np.cos(phi) + np.cos(lat1) * np.sin(phi) * np.cos(theta))
    # λ2 = λ1 + atan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )
    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(lat1), np.cos(phi) - np.sin(lat1) * np.sin(lat2))
    lat2 = np.rad2deg(lat2)
    lon2 = np.rad2deg(lon2)
    return np.array([lat2, lon2]).T


def analyze_distances_one_sphere(points: [(float, float, float)], show: bool = False, verbose: bool = False,
                                 figure_name: str = "distribution of min-distances") -> [float]:
    points = np.array(points)
    pointsX = points[:, 0]
    pointsY = points[:, 1]
    pointsZ = points[:, 2]
    diffX = np.abs(pointsX[:, np.newaxis] - pointsX)
    diffY = np.abs(pointsY[:, np.newaxis] - pointsY)
    diffZ = np.abs(pointsZ[:, np.newaxis] - pointsZ)
    diff = np.sqrt(np.power(diffX, 2) + np.power(diffY, 2) + np.power(diffZ, 2))
    # diff is a [x,x]-sized array with the diffs from all to all points,
    # the diagonal is 0 (from itself to itself), so add a diagonal matrix, to remove the zeros
    diff = diff + np.eye(len(points), len(points)) * diff.max()
    mins = diff.min(axis=0)

    if show:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mins))
        fig.update_layout(
            title=figure_name, xaxis_title='distances', yaxis_title='count'
        )
        fig.show()
    if verbose:
        print(f"distance mean: {np.mean(mins)}, std: {np.std(mins)} ({np.std(mins) / np.mean(mins) * 100}% deviation)")
    return mins


if __name__ == '__main__':
    # tests
    points_phi_theta, circ_radius = sunflower_pattern(points=100, point_distances=400, verbose=True)
    points_cartesian = spherical_2_cartesian_coordinates(points_phi_theta)
    analyze_distances_one_sphere(points_cartesian, show=True, verbose=True)
