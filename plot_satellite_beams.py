import plotly.graph_objects as go
import plotly.express as px
from astropy import units as u
from astropy import coordinates as coord
import numpy as np
import polygon_utilities as pu

from shapely.geometry import Polygon as shPoly
from shapely.geometry import MultiPolygon as shMPoly

# next color: px.colors.qualitative.Light24[14]
beam_color_dict = {'1': px.colors.qualitative.Plotly[0],
                   '2': px.colors.qualitative.Plotly[1],
                   '3': px.colors.qualitative.Plotly[2],
                   '4': px.colors.qualitative.Plotly[3],
                   '5': px.colors.qualitative.Plotly[4],
                   '6': px.colors.qualitative.Plotly[5],
                   '7': px.colors.qualitative.Plotly[6],
                   '8': px.colors.qualitative.Plotly[7],
                   '9': px.colors.qualitative.Plotly[8],
                   '10': px.colors.qualitative.Plotly[9],
                   '11': px.colors.qualitative.Alphabet[0],
                   '12': px.colors.qualitative.Alphabet[1],
                   '13': px.colors.qualitative.Alphabet[2],
                   '14': px.colors.qualitative.Alphabet[3],
                   '15': px.colors.qualitative.Alphabet[4],
                   '16': px.colors.qualitative.Alphabet[5],
                   '17': px.colors.qualitative.Alphabet[6],
                   '18': px.colors.qualitative.Alphabet[7],
                   '19': px.colors.qualitative.Alphabet[8],
                   '20': px.colors.qualitative.Alphabet[9],
                   '21': px.colors.qualitative.Alphabet[10],
                   '22': px.colors.qualitative.Alphabet[11],
                   '23': px.colors.qualitative.Alphabet[12],
                   '24': px.colors.qualitative.Alphabet[13],
                   '25': px.colors.qualitative.Alphabet[14],
                   '26': px.colors.qualitative.Alphabet[15],
                   '27': px.colors.qualitative.Alphabet[16],
                   '28': px.colors.qualitative.Alphabet[17],
                   '29': px.colors.qualitative.Alphabet[18],
                   '30': px.colors.qualitative.Alphabet[19],
                   '31': px.colors.qualitative.Alphabet[20],
                   '32': px.colors.qualitative.Alphabet[21],
                   '33': px.colors.qualitative.Light24[13],
                   '34': px.colors.qualitative.Alphabet[23],
                   '35': px.colors.qualitative.Alphabet[24],
                   '36': px.colors.qualitative.Alphabet[25],
                   '37': px.colors.qualitative.Light24[0],
                   '38': px.colors.qualitative.Light24[1],
                   '39': px.colors.qualitative.Light24[2],
                   '40': px.colors.qualitative.Light24[3],
                   '41': px.colors.qualitative.Light24[12],
                   '42': px.colors.qualitative.Light24[5],
                   '43': px.colors.qualitative.Light24[6],
                   '44': px.colors.qualitative.Light24[7],
                   '45': px.colors.qualitative.Light24[8],
                   '46': px.colors.qualitative.Light24[9],
                   '47': px.colors.qualitative.Light24[10],
                   '48': px.colors.qualitative.Light24[11],
                   }


def __ITRS_2_LonLatHeight(pos_itrs: [(float, float, float)]) -> [(float, float, float)]:
    lolah_list = []
    for temp_pos in pos_itrs:
        x, y, z = temp_pos
        itrs = coord.ITRS(x * u.km, y * u.km, z * u.km, 0 * u.km / u.s, 0 * u.km / u.s, 0 * u.km / u.s)
        location = itrs.earth_location
        lon = location.geodetic.lon.value
        lat = location.geodetic.lat.value
        height = location.geodetic.height.value
        temp_lolah = (lon, lat, height)
        lolah_list.append(temp_lolah)
    # lon in deg, lat in deg, height in km
    return lolah_list


def __LonLat_2_ITRS(self, pos_lola: [(float, float)]) -> [(float, float, float)]:
    itrs_list = []
    for temp_pos in pos_lola:
        location = coord.EarthLocation(lon=temp_pos[0] * u.deg, lat=temp_pos[1] * u.deg, height=0 * u.km)
        itrs_list.append((location.itrs.x.value, location.itrs.y.value, location.itrs.z.value))
    # x,y,z in km
    return itrs_list


# plot the 2d representation of beam-points
def plot_beam_points2d(beam_points: [(float, float)], beam_name: str,
                       figure_name: str = "Single 2D converted sat-beam"):
    fig = go.Figure()
    beam_points = np.array(beam_points)  # [(x, y)]
    fig.add_trace(go.Scatter(x=beam_points[:, 0], y=beam_points[:, 1],
                             mode='markers', hovertext=beam_name,
                             name=beam_name
                             ))
    # add custom layout
    fig.update_layout(
        title_text=figure_name,
        width=1500,
        height=1000,
        # scene=dict(aspectmode='data'),
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


# plot the 2d representation of the beam-point clusters
def plot_beam_clusters(beam_clusters: {}, beam_name: str, figure_name: str = "beam-clusters"):
    # beam_clusters = {key = cluster; value=[(x,y)]}
    fig = go.Figure()
    # use px.colors.qualitative.Alphabet
    entry_max = max(list(beam_clusters.keys()))
    if entry_max > 23:
        print("ERROR: plot_processed_rec_beams.plot_bam_clusters: too many clusters for color-scheme")
        exit(1)
    for index in list(beam_clusters.keys()):
        point_color = px.colors.qualitative.Alphabet[index]
        point_label = f"{beam_name}; cluster:{index}"
        cluster_points = beam_clusters[index]
        cluster_points = np.array(cluster_points)
        fig.add_trace(go.Scatter(x=cluster_points[:, 0], y=cluster_points[:, 1],
                                 mode='markers', hovertext=point_label, marker=dict(color=point_color),
                                 name=point_label
                                 ))
    # add custom layout
    fig.update_layout(
        title_text=figure_name,
        width=1500,
        height=1000,
        showlegend=False,
        # scene=dict(aspectmode='data'),
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


# plots a 2d map of the extracted polygons
def plot_beam_polygons(beams_polygons: {}, figure_name: str = "beam-clusters"):
    # beam_clusters = {key = cluster; value=[(x,y)]}
    fig = go.Figure()
    for beam_index in list(beams_polygons.keys()):
        polygon_list = beams_polygons[beam_index]
        point_label = f"beam: {beam_index}"
        point_color = beam_color_dict[str(beam_index)]
        # make the first polygon-plot by hand to create the legend-group and add the others to this group
        poly_points = np.array(polygon_list[0].get_points_line())
        fig.add_trace(go.Scatter(x=poly_points[:, 0], y=poly_points[:, 1],
                                 mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                 name=point_label, legendgroup=point_label,
                                 ))
        for i in range(1, len(polygon_list)):
            temp_poly = polygon_list[i]
            poly_points = np.array(temp_poly.get_points_line())
            fig.add_trace(go.Scatter(x=poly_points[:, 0], y=poly_points[:, 1],
                                     mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                     name=point_label, legendgroup=point_label, showlegend=False,
                                     ))
    # add custom layout
    fig.update_layout(
        title_text=figure_name,
        width=1500,
        height=1000,
        # showlegend=False,
        # scene=dict(aspectmode='data'),
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


def __get_beam_color(beam_index):
    beam_index = str(beam_index)
    if beam_index in list(beam_color_dict.keys()):
        return beam_color_dict[str(beam_index)]
    else:
        color_keys = list(beam_color_dict.keys())
        rnd_index = int(np.random.random(1) * len(color_keys))
        used_key = color_keys[rnd_index]
        return beam_color_dict[str(used_key)]


# plots a 2d map of the extracted polygons
def plot_shapely_polygons(beams_polygons: {}, figure_name: str = "beam-clusters"):
    # beam_clusters = {key = cluster; value=[(x,y)]}
    fig = go.Figure()
    for beam_index in list(beams_polygons.keys()):
        point_label = f"beam: {beam_index}"
        point_color = __get_beam_color(beam_index)
        poly_point_list_of_lists = pu.get_shPoly_points_2D(beams_polygons[beam_index])
        # make the first polygon-plot by hand to create the legend-group and add the others to this group
        first_poly = poly_point_list_of_lists[0]
        fig.add_trace(go.Scatter(x=first_poly[:, 0], y=first_poly[:, 1],
                                 mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                 name=point_label, legendgroup=point_label,
                                 ))
        for i in range(1, len(poly_point_list_of_lists)):
            temp_poly = poly_point_list_of_lists[i]
            fig.add_trace(go.Scatter(x=temp_poly[:, 0], y=temp_poly[:, 1],
                                     mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                     name=point_label, legendgroup=point_label, showlegend=False,
                                     ))
    # add custom layout
    fig.update_layout(
        title_text=figure_name,
        width=1500,
        height=1000,
        # showlegend=False,
        # scene=dict(aspectmode='data'),
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


def __check_for_wraparound(lola_points: [(float, float, float)]) -> [(float, float, float)]:
    lola_points = np.array(lola_points)
    for i in range(1, len(lola_points)):
        p1 = lola_points[i - 1]
        p2 = lola_points[i]
        if abs(p1[0] - p2[0]) > 180:
            if p1[0] > 0:
                lola_points[i, 0] = p2[0] + 360
                # help_point = [p2[0]+360, p2[1], p2[2]]
            else:
                lola_points[i, 0] = p2[0] - 360
                # help_point = [p2[0]-360, p2[1], p2[2]]
            # lola_points = np.insert(lola_points, i, help_point, axis=0)
    return lola_points


def plot_polygon_footprints(sat_pos_lolah: (float, float, float), beams_poly_dict: dict,
                            target_pos_lolah: (float, float, float) = None,
                            figure_name: str = f"Satellite beam-clusters footprint", sat_name: str = "satellite"):
    # all values in km
    fig = go.Figure()
    # add the beam-polygons
    for beam_name in list(beams_poly_dict.keys()):
        point_label = f"beam: {beam_name}"
        if beam_name in list(beam_color_dict.keys()):
            point_color = beam_color_dict[str(beam_name)]
        else:
            point_color = beam_color_dict[list(beam_color_dict.keys())[0]]
        polygons_list = beams_poly_dict[beam_name]
        if (polygons_list is not None) and (len(polygons_list) != 0) and (len(polygons_list[0]) != 0):
            # make the first polygon-plot by hand to create the legend-group and add the others to this group
            poly_points_lolah = np.array(__ITRS_2_LonLatHeight(polygons_list[0]))
            poly_points_lolah = __check_for_wraparound(poly_points_lolah)
            fig.add_trace(go.Scattermapbox(lon=poly_points_lolah[:, 0], lat=poly_points_lolah[:, 1],
                                           mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                           name=point_label, legendgroup=point_label,
                                           ))
            for i in range(1, len(polygons_list)):
                temp_poly = polygons_list[i]
                poly_points_lolah = np.array(__ITRS_2_LonLatHeight(temp_poly))
                poly_points_lolah = __check_for_wraparound(poly_points_lolah)
                poly_points_lolah = __check_for_wraparound(poly_points_lolah)
                fig.add_trace(go.Scattermapbox(lon=poly_points_lolah[:, 0], lat=poly_points_lolah[:, 1],
                                               mode='markers+lines', hovertext=point_label,
                                               marker=dict(color=point_color),
                                               name=point_label, legendgroup=point_label, showlegend=False,
                                               ))
    # add the satellite
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[sat_pos_lolah[0]],
        lat=[sat_pos_lolah[1]],
        name=sat_name,
        marker=dict(color='red', size=20)))
    # add the target
    if target_pos_lolah is not None:
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[target_pos_lolah[0]],
            lat=[target_pos_lolah[1]],
            name="target",
            marker=dict(color='green', size=20)))
    # add custom layout
    fig.update_layout(
        margin={'l': 0, 't': 30, 'b': 0, 'r': 0},
        mapbox={
            'style': "open-street-map",
            'center': {'lon': sat_pos_lolah[0], 'lat': sat_pos_lolah[1]},
            'zoom': 1},
        title_text=figure_name,
        width=1800,
        height=910,
        # showlegend=False,
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


def plot_polygon_itrs(beams_poly_dict_itrs: dict, figure_name: str = f"Satellite beam-clusters footprint"):
    # all values in km
    fig = go.Figure()
    # add the beam-polygons
    for beam_name in list(beams_poly_dict_itrs.keys()):
        point_label = f"beam: {beam_name}"
        if beam_name in list(beam_color_dict.keys()):
            point_color = beam_color_dict[str(beam_name)]
        else:
            point_color = beam_color_dict[list(beam_color_dict.keys())[0]]
        polygons_list = beams_poly_dict_itrs[beam_name]
        # make the first polygon-plot by hand to create the legend-group and add the others to this group
        # poly_points_itrs = __LonLat_2_ITRS(polygons_list[0])
        poly_points_itrs = np.array(polygons_list[0])  # [(x,y,z)] ITRS
        fig.add_trace(go.Scatter3d(x=poly_points_itrs[:, 0], y=poly_points_itrs[:, 1], z=poly_points_itrs[:, 2],
                                   mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                   name=point_label, legendgroup=point_label,
                                   ))
        for i in range(1, len(polygons_list)):
            # temp_poly = __LonLat_2_ITRS(polygons_list[i])
            poly_points_itrs = np.array(polygons_list[i])  # [(x,y,z)] ITRS
            fig.add_trace(go.Scatter3d(x=poly_points_itrs[:, 0], y=poly_points_itrs[:, 1], z=poly_points_itrs[:, 2],
                                       mode='markers+lines', hovertext=point_label, marker=dict(color=point_color),
                                       name=point_label, legendgroup=point_label,
                                       ))

    # add custom layout
    fig.update_layout(
        margin={'l': 0, 't': 30, 'b': 0, 'r': 0},
        title_text=figure_name,
        width=1800,
        height=910,
        # showlegend=False,
        # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis, aspectmode='data'),
    )
    fig.show()


def plot_polygon_manyEves_manyVic(eves_lolah: [(float, float, float)], figure_name: str,
                                  victims_lolah: [(float, float, float)], poly: shPoly = None,
                                  ul_measurements: dict = None):
    eve1_itrs = pu.lonLatHeight_2_ITRS(eves_lolah[0][0], eves_lolah[0][1], eves_lolah[0][2])
    fig = go.Figure()
    # add the beam-polygons
    if poly:
        point_color = beam_color_dict[list(beam_color_dict.keys())[0]]
        point_label = "RoI"
        showlegend = True
        polygons_list = pu.poly_get_points_itrs(eve1_itrs, poly)
        for i in range(0, len(polygons_list)):
            temp_poly = polygons_list[i]
            poly_points_lolah = np.array(__ITRS_2_LonLatHeight(temp_poly))
            poly_points_lolah = __check_for_wraparound(poly_points_lolah)
            fig.add_trace(go.Scattermapbox(lon=poly_points_lolah[:, 0], lat=poly_points_lolah[:, 1],
                                           mode='markers+lines', hovertext=point_label,
                                           marker=dict(color=point_color),
                                           name=point_label, legendgroup=point_label, showlegend=showlegend,
                                           ))
            showlegend = False
    # add the eavesdroppers
    eves_lolah = np.array(eves_lolah)
    fig.add_trace(go.Scattermapbox(mode="markers", lon=eves_lolah[:, 0], lat=eves_lolah[:, 1], name="DL observers",
                                   marker=dict(color='orange', size=20)))
    # add uplink_measurements
    if ul_measurements is not None:
        ul_names = list(ul_measurements.keys())
        ul_locations = []
        for temp_name in ul_names:
            temp_loc, temp_circle = ul_measurements[temp_name]
            ul_locations.append(list(temp_loc))
            if temp_circle is not None:
                circ_loc = temp_circle.exterior.coords
                circ_loc = np.array(circ_loc)
                fig.add_trace(go.Scattermapbox(mode="lines", lon=circ_loc[:, 0], lat=circ_loc[:, 1], name=temp_name,
                                               line={'width': 3, 'color': 'red'}))
        ul_locations = np.array(ul_locations)
        fig.add_trace(
            go.Scattermapbox(mode="markers", lon=ul_locations[:, 0], lat=ul_locations[:, 1], name="UL observers",
                             marker={'size': 10, 'color': 'red'}, hovertext=ul_names))
    # add the victims
    victims_lolah = np.array(victims_lolah)
    fig.add_trace(go.Scattermapbox(mode="markers", lon=victims_lolah[:, 0], lat=victims_lolah[:, 1], name="targets",
                                   marker=dict(color='green', size=20)))
    # add custom layout
    fig.update_layout(
        margin={'l': 0, 't': 30, 'b': 0, 'r': 0},
        mapbox={'style': "open-street-map",
                'center': {'lon': eves_lolah[0, 0], 'lat': eves_lolah[0, 1]},
                'zoom': 1},
        title_text=figure_name,
        width=1800,
        height=910,
    )
    fig.show()
