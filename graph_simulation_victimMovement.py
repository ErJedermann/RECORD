import json
import numpy as np
import os
import csv
from shapely.geometry import MultiPolygon as mPoly
import pandas as pd
import plotly.express as px

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


def generate_RoI_map(folder_name: str, file_name: str):
    loaded_data = __load_json_data(filename=folder_name + file_name)
    if loaded_data['distance_outside'] == 0:
        figure_name = f"valid estimation ({loaded_data['RoI_area']}km²)"
    else:
        figure_name = f"invalid estimation ({loaded_data['RoI_area']}km², {loaded_data['distance_outside']}km)"
    plot_satellite_beams.plot_polygon_manyEves_manyVic(figure_name=figure_name,
                                                       eves_lolah=loaded_data['eve_locations'],
                                                       victims_lolah=loaded_data['victim_locations'],
                                                       poly=loaded_data['polygon'])


def __read_output_csv(filename: str):
    if os.path.exists(filename):
        all_rows = []  # [[area1_s, area1_c, dist1_c_mid, dist1_c_end], [area2_s, area2_c, dist2_c_mid, dist2_c_end]]
        with open(filename, 'r') as csvinput:
            reader = csv.reader(csvinput)
            for row in reader:
                # row = next(reader)
                if row[0][0] == '[':
                    row[0] = row[0][1:]
                if row[-1][-1] == ']':
                    row[-1] = row[-1][:-1]
                all_rows.append(row)
        csvinput.close()
        all_colums = np.array(all_rows, dtype=float)  # .T
        return all_colums
    else:
        print(f"ERROR: read_output_csv: file ({filename}) does not exist!")


def __process_csv_2_df(csv_data: [[float]], movement_diameter: str):
    csv_data = np.array(csv_data)
    max_dist = np.maximum.reduce([csv_data[:, 2], csv_data[:, 3]])
    csv_data = list(np.array([csv_data[:, 0], csv_data[:, 1], max_dist]).T)
    df = pd.DataFrame(data=csv_data, columns=['area_single', 'area_combined', 'dist_outside'])
    df['⌀'] = movement_diameter
    is_outside = df['dist_outside'] > 0
    df['is_outside'] = is_outside
    # print(df)
    return df


def __bigDf_2_graph(bigDf: pd.DataFrame,
                    figure_name: str = "histogram of (in-)valid RoI estimations"):
    x_label = "RoI size in km²"
    fig = px.histogram(data_frame=bigDf, x='area_combined', color='is_outside', facet_row='⌀', )
    fig.update_traces(xbins=dict(
        start=0,
        end=300,
        size=10,
    ))
    fig.update_layout(
        title_text=figure_name,
        xaxis_title=x_label,
        # yaxis_title="region of interest [km²]",
        width=750,  # 1500 or 750
        height=450,  # 900 or 450
        barmode='stack',
    )
    fig.show()


def generate_hist(file_names: [str], movement_diameters: [str], folder_name: str = "", plot: bool = True):
    if len(file_names) != len(movement_diameters):
        raise Exception("Number of files to load != number of figure labels!")
    df_list = []
    for i in range(len(file_names)):
        temp_csv_data = __read_output_csv(filename=folder_name + file_names[i])
        temp_df = __process_csv_2_df(temp_csv_data, movement_diameter=movement_diameters[i])
        df_list.append(temp_df)
    big_df = pd.concat(df_list)
    if plot:
        __bigDf_2_graph(bigDf=big_df)
    return big_df


def static_analyze_df(df: pd.DataFrame):
    my_values = df['⌀'].tolist()
    my_values = list(dict.fromkeys(my_values))
    for diameter in my_values:
        all_out_values = df.loc[(df['⌀'] == diameter)]['is_outside'].tolist()
        is_outside = sum(all_out_values)  # True == 1 -> sum = number of true-values
        rate_outside = is_outside / len(all_out_values)
        rate_outside_str = '%.3f' % (rate_outside * 100)
        all_area_values = df.loc[(df['⌀'] == diameter)]['area_combined'].tolist()
        median_area = np.median(all_area_values)
        median_area_str = '%.3f' % median_area
        print(
            f"{diameter} ⌀: outside={rate_outside_str}% \t median RoI: {median_area_str}km² \t repetitions: {len(all_area_values)}")


if __name__ == '__main__':
    folder_name = "simulation_data/victim_movement/"
    movement_diameters = ["1km", "2km", "4km", "8km"]
    bucket_files = ["0.5km_RWP_400km_cont_3600sec_6_sun_eves_1VicType_realWorld_noisyPrediction.csv",
                    "1km_RWP_400km_cont_3600sec_6_sun_eves_1VicType_realWorld_noisyPrediction.csv",
                    "2km_RWP_400km_cont_3600sec_6_sun_eves_1VicType_realWorld_noisyPrediction.csv",
                    "4km_RWP_400km_cont_3600sec_6_sun_eves_1VicType_realWorld_noisyPrediction.csv"]
    data = generate_hist(file_names=bucket_files, movement_diameters=movement_diameters, folder_name=folder_name,
                         plot=True)
    static_analyze_df(data)
