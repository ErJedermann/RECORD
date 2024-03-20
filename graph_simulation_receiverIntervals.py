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
        return all_colums  # [[l1_atk1, l2_atk1], [l1_atk2, l2_atk2], [l1_atk3, l2_atk3]]
    else:
        print(f"ERROR: read_output_csv: file ({filename}) does not exist!")


def combine_rec_outputs_to_plot(x_axis: [float], x_label: str, figure_name: str, output_folder: str,
                                output_mapping: [(int, int, str)], atk_types: [str]):
    # output_mapping [(x_index1, atk_type1, file1), (x_index2, atk_type2, file2),...]
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
        for i in range(len(some_data)):
            loaded_column = some_data[i, :]
            temp_data_list[temp_atk_type + i][temp_x_index] = loaded_column
    # create the plot
    print_dict = {}  # {atk_type: [[x1], [x2], ...] }
    for i in range(len(atk_types)):
        print_dict[atk_types[i]] = temp_data_list[i]
    # customization
    #del print_dict[atk_types[4]]
    #del print_dict[atk_types[2]]
    plot_measurement_row.plot_attack_area_analysis_selfmade_quantiles(print_dict, x_axis, x_label=x_label, figure_name=figure_name)


if __name__ == '__main__':
    fig_name = f"Areas (5th percentile, Q1, median, Q3, 95th percentile) of different rec-intervalls"
    folder_name = "simulation_data/intervall_sniffing/"
    x_values = ['400 km']
    x_text = f"inter-observer-distance"

    attack_types = [
        "weak noisy (1/6rec)", "weak noisy (6rec)",
        "weak noisy interval 120x30 (1/6rec)", "weak noisy interval 120x30 (6rec)",
    ]
    mapping = [
        (0, 0, "400kmFibo_cont_3600sec_3eves_weakEvents_noisyPrediction.csv"),
        (0, 2, "400kmFibo_120x_30sec_rec_600sec_pause_6eves_weakEvents_noisyPrediction.csv")
    ]

    combine_rec_outputs_to_plot(x_axis=x_values, x_label=x_text, figure_name=fig_name, output_folder=folder_name,
                                output_mapping=mapping, atk_types=attack_types)
