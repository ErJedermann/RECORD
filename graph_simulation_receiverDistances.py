import numpy as np
import os
import csv

import plot_measurement_row

def read_output_csv(filename: str):
    filename = filename
    if os.path.exists(filename):
        all_rows = []
        with open(filename, 'r') as csvinput:
            reader = csv.reader(csvinput)
            for row in reader:
                if row[0][0] == '[':
                    row[0] = row[0][1:]
                if row[-1][-1] == ']':
                    row[-1] = row[-1][:-1]
                all_rows.append(row)
        csvinput.close()
        all_colums = np.array(all_rows, dtype=float).T
        return all_colums
    else:
        print(f"ERROR: read_output_csv: file ({filename}) does not exist!")


def combine_rec_outputs_to_plot(x_axis: [float], x_label: str, figure_name: str, output_folder: str,
                                output_mapping: [(int, int, str)], atk_types: [str]):
    if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
        print(f"ERROR: Ms_RN.combine_output_to_plot: given folder ({output_folder}) doesn't exist or is not a folder!")
        exit(1)
    if output_folder[-1] is not os.sep:
        output_folder = f"{output_folder}{os.sep}"
    # prepare the data-structure to fill in all the data-points
    temp_data_list = []  # [[atk1_data[[x1], [x2], ...], [atk2_data[[x1], [x2], ...], ...]
    for j in range(len(atk_types)):
        type_list = []
        for k in range(len(x_axis)):
            type_list.append([])
        temp_data_list.append(type_list)
    # get each file and fill the data-structure
    for entry in output_mapping:
        temp_x_index, temp_atk_type, temp_file = entry
        print(f"open: {temp_file}")
        some_data = read_output_csv(output_folder + temp_file)
        some_data[some_data == 0] = np.nan  # remove 0s in the data (no event at the first receiver -> no estimation)
        for i in range(len(some_data)):
            loaded_column = some_data[i, :]
            temp_data_list[temp_atk_type + i][temp_x_index] = loaded_column
    # create the plot
    print_dict = {}  # {atk_type: [[x1], [x2], ...] }
    for i in range(len(atk_types)):
        print_dict[atk_types[i]] = temp_data_list[i]
    # customization
    del print_dict[atk_types[6]]
    del print_dict[atk_types[4]]
    plot_measurement_row.plot_attack_area_analysis_selfmade_quantiles(print_dict, x_axis, x_label=x_label, figure_name=figure_name, is_interObsDist=True)


if __name__ == '__main__':
    folder_name = "simulation_data/observer_distances_and_amount/"

    x_values = ['100 km', '200 km', '300 km', '400 km', '500 km', '600 km', '700 km', '800 km']

    x_text = f"inter-observer-distance"
    fig_name = f"Areas (5th percentile, Q1, median, Q3, 95th percentile) of different rec-setups (1 min recording, 6 receivers)"

    attack_types = [
        "weak noisy (1/3rec)", "weak noisy (3rec)",
        "weak noisy1 (1/6rec)", "weak noisy1 (6rec)",
        "weak noisy (1/6rec)", "weak noisy (6rec)",  # to get new colors for the 6 & 12 rec attacker
        "weak noisy (1/12rec)", "weak noisy (12rec)",
    ]
    mapping_2 = [
        (0, 0, "100kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (1, 0, "200kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (2, 0, "300kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (3, 0, "400kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (4, 0, "500kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (5, 0, "600kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (6, 0, "700kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (7, 0, "800kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (0, 2, "100kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (1, 2, "200kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (2, 2, "300kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (3, 2, "400kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (4, 2, "500kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (5, 2, "600kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (6, 2, "700kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (7, 2, "800kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (0, 4, "100kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (1, 4, "200kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (2, 4, "300kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (3, 4, "400kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (4, 4, "500kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (5, 4, "600kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (6, 4, "700kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (7, 4, "800kmFibo_cont_3600sec_6eves_weakEvents_noisyPrediction.csv"),
        (0, 6, "100kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (1, 6, "200kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (2, 6, "300kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (3, 6, "400kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (4, 6, "500kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (5, 6, "600kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (6, 6, "700kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
        (7, 6, "800kmFibo_cont_3600sec_12eves_weakEvents_noisyPrediction.csv"),
    ]

    combine_rec_outputs_to_plot(x_axis=x_values, x_label=x_text, figure_name=fig_name, output_folder=folder_name,
                                output_mapping=mapping_2, atk_types=attack_types)
